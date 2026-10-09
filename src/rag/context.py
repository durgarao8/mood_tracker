def build_context(results):
    """
    Convert ChromaDB retrieval results into
    a clean text context for the LLM.
    """

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    context_parts = []

    for i, document in enumerate(documents):

        metadata = {}

        if metadatas and i < len(metadatas):
            metadata = metadatas[i] or {}

        source = metadata.get("source", "Unknown")
        page = metadata.get("page", "Unknown")
        chapter = metadata.get("chapter", "Unknown")

        context_parts.append(
            f"""
--- Source {i + 1} ---
Chapter: {chapter}
Page: {page}
Source: {source}

Content:
{document}
"""
        )

    return "\n".join(context_parts)