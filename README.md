# 📄 LlamaIndex CV Explorer

A simple web application for uploading, indexing, and reviewing candidate resumes using LlamaIndex + OpenAI.

## ⚙️ Technologies

- [LlamaIndex](https://llamaindex.ai/)
- [OpenAI Embedding](https://platform.openai.com/)
- [ChromaDB](https://www.trychroma.com/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://reactjs.org/)

---

## 🚀 How to Run

1. **Clone or copy the project.**

2. **Create a virtual environment and activate it (recommended):**

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

3. **Install the required packages:**

```bash
pip install -r requirements.txt
```

4. **Create a `.env` file with your OpenAI API key:**

```
OPENAI_API_KEY=your_openai_api_key_here
```

5. **Start the backend server:**

```bash
uvicorn main:app --reload
```

6. **Start the frontend:**

```bash
cd frontend
npm install
npm start
```

7. **Open in your browser:**

Frontend will be available at: http://localhost:3000
Backend API will be available at: http://localhost:8000

## 📋 API Endpoints

- `GET /` - API Status
- `GET /candidates` - Get a list of all candidates
- `GET /candidates/{name}` - Get details of a specific candidate
