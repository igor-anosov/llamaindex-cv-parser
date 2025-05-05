from fastapi import FastAPI, HTTPException
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import TextNode
from fastapi.middleware.cors import CORSMiddleware
import chromadb
import os
import logging  
import json

# Load environment variables
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
BASE_DIR = Path(__file__).resolve().parent
CV_FOLDER = BASE_DIR / "cv_folder"
PERSIST_DIR = BASE_DIR / "chroma_db"

app = FastAPI()
index = None
query_engine = None

# React app CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_or_create_index():
    global index

    if PERSIST_DIR.exists():
        logger.info("🔄 Loading index from existing Chroma collection...")
        chroma_client = chromadb.PersistentClient(path=str(PERSIST_DIR))
        chroma_collection = chroma_client.get_or_create_collection("candidates")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
        return index

    logger.info("🆕 Creating new index from CVs...")

    if not CV_FOLDER.exists():
        raise FileNotFoundError("❌ cv_folder not found.")

    documents = SimpleDirectoryReader(str(CV_FOLDER)).load_data()
    if not documents:
        raise ValueError("❌ No documents found in cv_folder.")

    splitter = SentenceSplitter(chunk_size=512)
    nodes = []
    for doc in documents:
        first_line = doc.text.strip().splitlines()[0]
        chunks = splitter.get_nodes_from_documents([doc])
        for node in chunks:
            node.metadata = {"intro": first_line}
            nodes.append(node)
    logger.info(f"📄 Split into {len(nodes)} chunks.")

    embed_model = OpenAIEmbedding()
    texts = [node.text for node in nodes]
    embeddings = embed_model.get_text_embedding_batch(texts)
    for node, emb in zip(nodes, embeddings):
        node.embedding = emb

    chroma_client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    chroma_collection = chroma_client.get_or_create_collection("candidates")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    storage_context.persist()

    logger.info("✅ Index created and stored.")
    return index

# Initialize at startup
try:
    index = get_or_create_index()
    query_engine = index.as_query_engine()
except Exception as e:
    logger.error(f"🚨 Failed to initialize index: {str(e)}")

@app.get("/")
def root():
    return {"message": "LlamaIndex CV Explorer is running 🚀"}

@app.get("/candidates")
async def list_candidates():
    if not query_engine:
        raise HTTPException(status_code=500, detail="Query engine not ready")
    try:
        # Get all nodes from index to extract metadata
        all_nodes = []
        if hasattr(index, "docstore"):
            all_nodes = list(index.docstore.docs.values())
        elif hasattr(index, "_docstore"):
            all_nodes = list(index._docstore.docs.values())
        
        # If we have nodes with metadata, extract the profession directly
        candidates_map = {}
        for node in all_nodes:
            if hasattr(node, "metadata") and "intro" in node.metadata:
                first_line = node.metadata["intro"]
                # Extract document ID or use a placeholder
                doc_id = node.id_or_repr if hasattr(node, "id_or_repr") else "unknown"
                
                if doc_id not in candidates_map:
                    # Try to extract name from text
                    text_sample = node.text[:500] if hasattr(node, "text") else ""
                    candidates_map[doc_id] = {
                        "node": node,
                        "intro": first_line,
                        "text_sample": text_sample
                    }
        
        if candidates_map:
            # Use query engine to extract structured data using metadata
            prompt = (
                "Based on the intro line and text samples I'll provide, create a JSON array of candidates. "
                "For each candidate, extract the full name, profession from the INTRO line, and estimated years of experience. "
                "Format: [{\"name\": ..., \"profession\": ..., \"experience\": ...}]. "
                "Here are the candidates:\n\n" + 
                "\n---\n".join([
                    f"INTRO: {data['intro']}\nSAMPLE: {data['text_sample'][:200]}..." 
                    for data in candidates_map.values()
                ])
            )
        else:
            # Fallback to original approach
            prompt = (
                "Return a JSON array where each item contains the full name, profession, and years of experience "
                "of each candidate found in the dataset. The profession should be extracted only from the first line "
                "or first paragraph of the candidate's resume. Format: [{\"name\": ..., \"profession\": ..., \"experience\": ...}]"
                "If name is not known, use the value 'Candidate {letter starting from A}'."
                "The response must contain at least 5 candidates."
            )

        response = query_engine.query(prompt)
        try:
            return json.loads(str(response))
        except json.JSONDecodeError:
            logger.warning("Failed to parse response as JSON.")
            return {"raw": str(response)}
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Query failed")

@app.get("/candidates/{name}")
async def get_candidate_summary(name: str):
    if not query_engine:
        raise HTTPException(status_code=500, detail="Query engine not ready")
    try:
        prompt = f"Summarize the skills and experience of candidate named {name}."
        response = query_engine.query(prompt)
        return {"summary": str(response)}
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Query failed")
