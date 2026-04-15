"""
RAG vector store for the financial knowledge base.

Builds a ChromaDB in memory collection over the curated financial education
content in data.py, embedded with Google text-embedding-004. The Advisory
Agent uses this to answer concept and strategy questions.
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import embeddings, logger
from .data import KNOWLEDGE_BASE


def _build_documents() -> list[Document]:
    """Convert the knowledge base entries into LangChain Documents."""
    return [
        Document(
            page_content=f"{entry['title']}\n\n{entry['content']}",
            metadata={"title": entry["title"], "category": entry["category"]},
        )
        for entry in KNOWLEDGE_BASE
    ]


def _build_vectorstore() -> Chroma:
    docs = _build_documents()
    store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="financial_knowledge",
    )
    logger.info(f"Indexed {len(docs)} financial knowledge entries in ChromaDB")
    return store


knowledge_vectorstore: Chroma = _build_vectorstore()
