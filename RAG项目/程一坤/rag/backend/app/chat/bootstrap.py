"""默认问答服务装配（LLM/检索/记忆的实例组装，自 chat/service.py 拆出）。

职责只有一件事：把真实依赖（检索服务、LLM 客户端、长期记忆存储与开关）
组装成一个 ChatService。全部放函数内延迟导入——模块导入本文件不触发
任何外部连接；长期记忆装配失败也不抛错，只降级为"本轮不读写记忆"。

为什么坚持函数内延迟导入：本文件会被评测脚本（evaluation/run_eval.py）与
API 启动路径共同引用。若把 Milvus / Embedding 客户端提到模块顶层，
任何"只想导入一个装配函数"的使用者都会顺带建立外部连接（单测环境尤其致命）。
"""
from app.chat.service import ChatService


def build_default_chat_service() -> ChatService:
    """构建默认问答服务（连接真实检索和 LLM）。

    参数：无（全部配置自 app.core.config.settings 读取）。
    返回：装配完成的 ChatService。长期记忆部分失败时为 None（功能降级，不抛错）。

    装配顺序：检索 → LLM → 长期记忆（可失败）。
    把记忆放最后，是因为它是唯一需要建 Milvus 集合的环节、失败面最大，
    且失败不影响问答主链路。
    """
    from app.core.config import settings
    from app.models.embedding import SiliconFlowEmbeddingClient
    from app.models.llm import build_chat_client_from_settings
    from app.retrieval.assembly import build_default_retrieval_service

    # 检索与 LLM 是问答的硬依赖：拿不到就应当直接抛错（放在下面的 try 之外）
    retrieval_service = build_default_retrieval_service()
    llm_client = build_chat_client_from_settings()

    # 长期记忆（批次 14）：独立集合 + 用户开关门卫；任一环节失败都不影响问答主流程
    long_term_memory = None
    memory_gate = None
    try:
        from pymilvus import MilvusClient

        from app.memory.long_term import LongTermMemoryStore
        from app.memory.memory_settings import MemorySettingsStore

        store = LongTermMemoryStore(
            milvus_client=MilvusClient(
                uri=f"http://{settings.milvus_host}:{settings.milvus_port}"
            ),
            # 记忆集合与法条集合用同一个 embedding 模型 + 同一维度，才能复用同一套
            # 向量与距离口径（法条检索与记忆检索的相似度阈值可直接沿用）
            embedding_client=SiliconFlowEmbeddingClient(
                api_url=settings.embedding_api_base_url,
                api_key=settings.embedding_api_key,
                model=settings.embedding_model,
                dimension=settings.embedding_dimension,
            ),
            collection_name=settings.milvus_long_term_collection_name,
            dimension=settings.embedding_dimension,
            dedup_threshold=settings.long_term_memory_dedup_threshold,
        )
        # ensure_collection 幂等：集合已存在时是空操作，所以每次装配都调是安全的
        store.ensure_collection()
        long_term_memory = store
        from app.chat.chat_runtime import build_default_session_factory

        # 开关门卫：是否允许写入长期记忆由用户在设置页决定；
        # 这里传的是"可调用对象"而非当期取值，使运行中改设置能立即生效
        # （否则每次开关都要重启服务）。
        settings_store = MemorySettingsStore(build_default_session_factory())
        memory_gate = settings_store.is_enabled
    except Exception as error:  # noqa: BLE001
        import logging

        # 只告警不抛出：记忆属于增强功能，不能让"记忆服务抖动"升级成整站不可用
        logging.getLogger("app.chat.memory").warning(
            "长期记忆装配失败，本轮问答不读写记忆：%s: %s", type(error).__name__, error
        )

    return ChatService(
        retrieval_service=retrieval_service,
        llm_client=llm_client,
        # 拒答阈值来自配置（阶段 8.4 校准值；未配置为 0 = 不启用）
        refusal_min_vector_score=settings.refusal_min_vector_score,
        long_term_memory=long_term_memory,
        memory_gate=memory_gate,
        memory_top_k=settings.long_term_memory_top_k,
        # 会话前情注入（批次 21）：默认 false，开启后提示词多一段"本次会话前情"
        session_summary_enabled=settings.session_summary_enabled,
    )
