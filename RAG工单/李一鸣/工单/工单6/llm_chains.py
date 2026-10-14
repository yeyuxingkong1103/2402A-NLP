"""Basic LangChain prompt and retrieval chain construction."""

from __future__ import annotations

from config import config


PROMPT = """你是新闻问答助手。只依据给出的上下文回答问题。
如果上下文没有答案，请明确说明证据不足。回答末尾列出来源页码。

上下文：
{context}

问题：{question}
"""


def build_chain(retriever, llm):
    from langchain_core.prompts import PromptTemplate
    from langchain.chains import RetrievalQA

    prompt = PromptTemplate(template=PROMPT, input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
