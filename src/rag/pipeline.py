from src.retrieval.retriever import retrieve
from src.rag.context import build_context
from src.llm.prompt import build_prompt
from src.llm.groq_client import generate_response


def answer_question(question: str, top_k: int = 5):

    # ---------------------------------------------
    # 1. Retrieve documents
    # ---------------------------------------------

    results = retrieve(
        query=question,
        top_k=top_k
    )

    # ---------------------------------------------
    # 2. Build context
    # ---------------------------------------------

    context = build_context(results)

    # ---------------------------------------------
    # 3. Build prompt
    # ---------------------------------------------

    prompt = build_prompt(
        question=question,
        context=context
    )

    # ---------------------------------------------
    # 4. Generate answer
    # ---------------------------------------------

    answer = generate_response(
        prompt=prompt
    )

    # ---------------------------------------------
    # 5. Extract sources
    # ---------------------------------------------

    sources = []

    metadatas = results.get("metadatas", [[]])[0]

    for metadata in metadatas:

        metadata = metadata or {}

        sources.append(
            {
                "chapter": str(
                    metadata.get("chapter", "Unknown")
                ),
                "page": str(
                    metadata.get("page", "Unknown")
                ),
                "source": str(
                    metadata.get("source", "Unknown")
                )
            }
        )

    # ---------------------------------------------
    # 6. Return final result
    # ---------------------------------------------

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }