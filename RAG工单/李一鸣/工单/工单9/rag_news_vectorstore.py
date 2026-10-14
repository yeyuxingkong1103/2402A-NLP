"""PDF loading, chunking and Chroma retrieval pipeline."""

from __future__ import annotations

from pathlib import Path

from config import config


def build_vectorstore(pdf_path: str | None = None):
    from langchain_chroma import Chroma
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    source = pdf_path or config.source_pdf
    pages = PyPDFLoader(source).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120).split_documents(pages)
    embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)
    return Chroma.from_documents(chunks, embeddings, collection_name=config.collection_name, persist_directory=config.persist_directory)


def search(question: str, pdf_path: str | None = None, k: int = 4):
    store = build_vectorstore(pdf_path)
    return store.similarity_search_with_score(question, k=k)
