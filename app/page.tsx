"use client";

import { useEffect, useState } from "react";

type Source = {
  chapter?: string;
  page?: string | number;
  source?: string;
};

type ChatMessage = {
  id: number;
  question: string;
  answer: string;
  sources?: Source[];
};

const API_URL = "http://127.0.0.1:8000";

export default function Home() {
  const [question, setQuestion] = useState("");
  const [history, setHistory] = useState<ChatMessage[]>([]);
  const [selectedChat, setSelectedChat] = useState<ChatMessage | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Load history from browser storage
  useEffect(() => {
    const savedHistory = localStorage.getItem("dsa-chat-history");

    if (savedHistory) {
      try {
        setHistory(JSON.parse(savedHistory));
      } catch {
        localStorage.removeItem("dsa-chat-history");
      }
    }
  }, []);

  // Save history whenever it changes
  useEffect(() => {
    localStorage.setItem("dsa-chat-history", JSON.stringify(history));
  }, [history]);

  async function askQuestion() {
    const trimmedQuestion = question.trim();

    if (!trimmedQuestion || loading) {
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: trimmedQuestion,
        }),
      });

      if (!response.ok) {
        throw new Error(`Request failed: ${response.status}`);
      }

      const data = await response.json();

      const newChat: ChatMessage = {
        id: Date.now(),
        question: trimmedQuestion,
        answer: data.answer || "No answer received.",
        sources: data.sources || [],
      };

      setHistory((previous) => [newChat, ...previous]);
      setSelectedChat(newChat);

      // IMPORTANT:
      // Clear the textbox after successful submission
      setQuestion("");
    } catch (err) {
      console.error(err);
      setError(
        "Unable to connect to the DSA chatbot. Make sure FastAPI is running."
      );
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(
    event: React.KeyboardEvent<HTMLTextAreaElement>
  ) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      askQuestion();
    }
  }

  function startNewChat() {
    setSelectedChat(null);
    setQuestion("");
    setError("");
  }

  function clearHistory() {
    const confirmed = window.confirm(
      "Are you sure you want to clear all chat history?"
    );

    if (!confirmed) {
      return;
    }

    setHistory([]);
    setSelectedChat(null);
    localStorage.removeItem("dsa-chat-history");
  }

  return (
    <main className="app-container">
      {/* SIDEBAR */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="logo">
            <div className="logo-icon">D</div>

            <div>
              <h2>DSA RAG</h2>
              <span>Chatbot</span>
            </div>
          </div>

          <button
            className="new-chat-button"
            onClick={startNewChat}
          >
            + New Chat
          </button>
        </div>

        <div className="history-section">
          <div className="history-title">
            <span>Chat History</span>

            {history.length > 0 && (
              <button
                className="clear-history"
                onClick={clearHistory}
              >
                Clear
              </button>
            )}
          </div>

          {history.length === 0 ? (
            <div className="empty-history">
              <div className="empty-history-icon">💬</div>
              <p>No conversations yet</p>
              <span>Ask your first DSA question.</span>
            </div>
          ) : (
            <div className="history-list">
              {history.map((chat) => (
                <button
                  key={chat.id}
                  className={`history-item ${
                    selectedChat?.id === chat.id
                      ? "active"
                      : ""
                  }`}
                  onClick={() => setSelectedChat(chat)}
                >
                  <span className="history-item-icon">▹</span>

                  <span className="history-question">
                    {chat.question}
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="sidebar-footer">
          <div className="knowledge-badge">
            <span className="status-dot"></span>
            DSA Knowledge Base
          </div>
        </div>
      </aside>

      {/* MAIN CHAT AREA */}
      <section className="chat-area">
        {/* HEADER */}
        <header className="chat-header">
          <div>
            <h1>DSA RAG Chatbot</h1>
            <p>
              Ask questions from your Data Structures & Algorithms
              knowledge base.
            </p>
          </div>

          <div className="connection-status">
            <span className="status-dot"></span>
            Online
          </div>
        </header>

        {/* CONTENT */}
        <div className="chat-content">
          {!selectedChat ? (
            <div className="welcome-screen">
              <div className="welcome-icon">🧠</div>

              <h2>Learn DSA with RAG</h2>

              <p>
                Ask a question and get an answer grounded in your
                DSA knowledge base.
              </p>

              <div className="suggestions">
                <button
                  onClick={() =>
                    setQuestion("What is a stack?")
                  }
                >
                  What is a stack?
                </button>

                <button
                  onClick={() =>
                    setQuestion("Explain binary search")
                  }
                >
                  Explain binary search
                </button>

                <button
                  onClick={() =>
                    setQuestion(
                      "What is the time complexity of merge sort?"
                    )
                  }
                >
                  Merge sort complexity
                </button>

                <button
                  onClick={() =>
                    setQuestion(
                      "What is the difference between Prim's and Kruskal's algorithms?"
                    )
                  }
                >
                  Prim vs Kruskal
                </button>
              </div>
            </div>
          ) : (
            <div className="conversation">
              {/* USER QUESTION */}
              <div className="question-card">
                <div className="message-label">
                  <div className="avatar user-avatar">U</div>
                  <span>You</span>
                </div>

                <p>{selectedChat.question}</p>
              </div>

              {/* BOT ANSWER */}
              <div className="answer-card">
                <div className="message-label">
                  <div className="avatar bot-avatar">D</div>
                  <span>DSA RAG Assistant</span>
                </div>

                <div className="answer-text">
                  {selectedChat.answer}
                </div>

                {/* SOURCES */}
                {selectedChat.sources &&
                  selectedChat.sources.length > 0 && (
                    <div className="sources-section">
                      <h3>📚 Sources</h3>

                      <div className="sources-grid">
                        {selectedChat.sources.map(
                          (source, index) => (
                            <div
                              className="source-card"
                              key={index}
                            >
                              <div className="source-number">
                                {index + 1}
                              </div>

                              <div>
                                <strong>
                                  {source.chapter ||
                                    "DSA Knowledge Base"}
                                </strong>

                                <p>
                                  {source.page
                                    ? `Page ${source.page}`
                                    : "Source document"}
                                </p>
                              </div>
                            </div>
                          )
                        )}
                      </div>
                    </div>
                  )}
              </div>
            </div>
          )}

          {error && (
            <div className="error-message">
              ⚠️ {error}
            </div>
          )}
        </div>

        {/* INPUT AREA */}
        <div className="input-section">
          <div className="input-box">
            <textarea
              value={question}
              onChange={(event) =>
                setQuestion(event.target.value)
              }
              onKeyDown={handleKeyDown}
              placeholder="Ask a DSA question..."
              rows={1}
              disabled={loading}
            />

            <button
              className="send-button"
              onClick={askQuestion}
              disabled={!question.trim() || loading}
            >
              {loading ? (
                <span className="spinner"></span>
              ) : (
                "↑"
              )}
            </button>
          </div>

          <p className="input-hint">
            Press Enter to ask • Shift + Enter for a new line
          </p>
        </div>
      </section>
    </main>
  );
}