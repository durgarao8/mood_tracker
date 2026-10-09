from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import QuestionRequest, QuestionResponse
from src.rag.pipeline import answer_question


app = FastAPI(
    title="DSA RAG Chatbot API",
    description="RAG-powered Data Structures and Algorithms chatbot",
    version="1.0.0",
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():
    return {
        "message": "DSA RAG Chatbot API is running"
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# =========================================================
# ASK QUESTION
# =========================================================

@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):

    try:

        result = answer_question(
            question=request.question,
            top_k=request.top_k
        )

        return {
            "question": result["question"],
            "answer": result["answer"],
            "sources": result["sources"]
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )