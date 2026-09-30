from backend.app.rag.pipeline import RetrievalDecision


def validate_answer(answer: str, decision: RetrievalDecision) -> bool:
    # 没有 RAG 授权时，任何模型回答都不得通过。
    if not decision.can_answer:
        return False
    # 空回答不可展示。
    if not answer.strip():
        return False
    # MVP 引用校验以 can_answer 为硬门槛，后续任务可扩展逐条引用映射。
    return True
