from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)


class Source(BaseModel):
    chapter: str
    page: str
    source: str


class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]