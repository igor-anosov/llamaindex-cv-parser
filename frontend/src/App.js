import React, { useState, useEffect } from "react";
import "./App.css";

function App() {
  const [candidates, setCandidates] = useState([]);
  const [selectedCandidate, setSelectedCandidate] = useState(null);
  const [candidateSummary, setCandidateSummary] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchCandidates = async () => {
    setLoading(true);
    try {
      const response = await fetch("http://localhost:8000/candidates");
      if (!response.ok) {
        throw new Error("Failed to fetch candidates");
      }
      const data = await response.json();
      setCandidates(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Fetch all candidates on initial load
  useEffect(() => {
    fetchCandidates();
  }, []);

  // Fetch candidate summary when a candidate is selected
  const handleCandidateClick = async (candidate) => {
    setSelectedCandidate(candidate);
    setLoading(true);

    try {
      const response = await fetch(
        `http://localhost:8000/candidates/${encodeURIComponent(candidate.name)}`
      );
      if (!response.ok) {
        throw new Error("Failed to fetch candidate details");
      }
      const data = await response.json();
      setCandidateSummary(data.summary);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header>
        <h1>CV Explorer</h1>
      </header>

      <div className="main-content">
        {error && <div className="error-message">Error: {error}</div>}

        <div className="candidates-list">
          <h2>Candidates</h2>
          {loading && !selectedCandidate ? (
            <div className="loading">Loading candidates...</div>
          ) : candidates.length === 0 ? (
            <div>No candidates found</div>
          ) : (
            <ul>
              {candidates.map((candidate, index) => (
                <li
                  key={index}
                  className={
                    selectedCandidate?.name === candidate.name ? "selected" : ""
                  }
                  onClick={() => handleCandidateClick(candidate)}
                >
                  <h3>{candidate.name}</h3>
                  <div className="candidate-brief">
                    <p>
                      <strong>Profession:</strong> {candidate.profession}
                    </p>
                    <p>
                      <strong>Experience:</strong> {candidate.experience} years
                    </p>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="candidate-details">
          <h2>Candidate Details</h2>
          {selectedCandidate ? (
            <div>
              <h3>{selectedCandidate.name}</h3>
              <div className="candidate-info">
                <p>
                  <strong>Profession:</strong> {selectedCandidate.profession}
                </p>
                <p>
                  <strong>Experience:</strong> {selectedCandidate.experience}{" "}
                  years
                </p>
              </div>
              <div className="candidate-summary">
                <h4>Summary</h4>
                {loading ? (
                  <div className="loading">Loading summary...</div>
                ) : (
                  <p>{candidateSummary || "No summary available"}</p>
                )}
              </div>
            </div>
          ) : (
            <div className="no-selection">
              Select a candidate to view details
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
