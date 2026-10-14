"""Runnable-shaped RAG demo; concrete provider setup is intentionally injectable."""

from llm_chains import build_chain


def ask(question: str, retriever, llm) -> dict:
    chain = build_chain(retriever, llm)
    return chain.invoke({"query": question})


if __name__ == "__main__":
    print("Inject a configured retriever and LLM, then call ask(question, retriever, llm).")
