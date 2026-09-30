"""RAG 在线问答主流程与旧接口兼容层。

离线知识库构建已经拆分到 ``app/rag_ocr_bge.py``：
PDF/OCR/表格解析 -> 清洗 -> 分块 -> BGE-M3 向量化 -> Milvus 入库。

本文件负责在线流程：
问题向量化 -> Dense/BM25 检索 -> RRF 融合 -> BGE 重排 -> DeepSeek 生成。

为了不影响现有 API、路由和测试，本文件仍导出原来的离线函数名；这些入口会把
任务转交给 ``rag_ocr_bge``，实际离线实现不再写在这里。
"""

import re
from functools import lru_cache
from pathlib import Path
from threading import RLock

import httpx
from sentence_transformers import CrossEncoder

from app import rag_ocr_bge as offline


# 两个模块共享同一个配置对象、模型锁和 Milvus 客户端，避免重复加载模型或连接。
settings = offline.settings
_model_lock = offline._model_lock
# 保存离线模块的真实实现。兼容代理会临时替换模块属性，因此不能在代理内部
# 再通过 offline.xxx 寻址，否则替换后可能重新调用代理自身并形成递归。
_offline_safe_vl = offline._safe_vl
_offline_table_text = offline._table_text
_offline_extract_pdf_text = offline.extract_pdf_text
# 兼容入口会短暂替换离线模块中的视觉函数；可重入锁确保并发调用不会相互覆盖。
_compatibility_lock = RLock()

# 医疗安全提示与急症短路关键词属于在线答案生成规则。
MEDICAL_NOTICE = "本回答仅用于健康知识教育，不能替代医生的诊断和治疗。"
EMERGENCY_TERMS = ("胸痛", "呼吸困难", "意识不清", "昏迷", "抽搐", "自杀", "服毒")
RRF_CONSTANT = 60


# =============================================================================
# 离线流程兼容入口
# 真正实现位于 app/rag_ocr_bge.py；保留这些名字可确保旧路由和导入代码继续运行。
# =============================================================================

TextChunk = offline.TextChunk
ocr_engine = offline.ocr_engine
clean_pdf_text = offline.clean_pdf_text
clean_text = offline.clean_text
remove_repeated_watermarks = offline.remove_repeated_watermarks
vl_components = offline.vl_components
_clean_vl_output = offline._clean_vl_output
_visual_task = offline._visual_task
_find_tables = offline._find_tables
_render_region = offline._render_region
_ocr_file = offline._ocr_file
_vl_image = offline._vl_image


def _safe_vl(path: Path, task: str) -> str:
    """兼容旧调用路径，并允许测试临时替换本模块的 ``_vl_image``。"""
    with _compatibility_lock:
        original = offline._vl_image
        offline._vl_image = _vl_image
        try:
            return _offline_safe_vl(path, task)
        finally:
            offline._vl_image = original


def _table_text(page, folder: Path, tables: list, use_visual: bool) -> str:
    """兼容旧调用路径，实际表格抽取由离线模块完成。"""
    with _compatibility_lock:
        original = offline._safe_vl
        offline._safe_vl = _safe_vl
        try:
            return _offline_table_text(page, folder, tables, use_visual)
        finally:
            offline._safe_vl = original


def extract_pdf_text(path: Path, use_visual: bool = True) -> str:
    """兼容旧调用路径，实际 PDF/OCR/VL 解析由离线模块完成。"""
    with _compatibility_lock:
        original_safe_vl = offline._safe_vl
        original_table_text = offline._table_text
        offline._safe_vl = _safe_vl
        offline._table_text = _table_text
        try:
            return _offline_extract_pdf_text(path, use_visual)
        finally:
            offline._safe_vl = original_safe_vl
            offline._table_text = original_table_text


def chunk_text(text: str, size: int = 500, overlap: int = 80) -> list[TextChunk]:
    """兼容旧调用路径，实际文本清洗和分块由离线模块完成。"""
    return offline.chunk_text(text, size, overlap)


embedding_model = offline.embedding_model
encode_documents = offline.encode_documents
milvus_client = offline.milvus_client
ensure_collection = offline.ensure_collection
delete_document = offline.delete_document
build_index = offline.build_index
classify_document = offline.classify_document
update_document_catalog = offline.update_document_catalog


# =============================================================================
# 第 4 步：用户问题向量化
# =============================================================================

def encode_query(query: str) -> list[float]:
    """清洗问题并使用与离线文档相同的 BGE-M3 生成归一化向量。"""
    clean_query = clean_text(query)
    if not clean_query:
        raise ValueError("用户问题不能为空")
    # 保留本模块调用点，兼容已有测试和外部代码对 encode_documents 的替换。
    return encode_documents([clean_query])[0]


# =============================================================================
# 第 5 步：Milvus Dense 语义检索 + BM25 关键词检索
# =============================================================================

def _filter(role_id: int, user_id: int) -> str:
    """限制角色范围，并只允许用户访问公开文档或自己上传的私有文档。"""
    return f"role_id == {int(role_id)} and (is_public == true or owner_id == {int(user_id)})"


def search_milvus(
    data: list, field: str, role_id: int, user_id: int, limit: int
) -> list[list[dict]]:
    """执行一路 Milvus 检索；field 决定使用 COSINE 还是 BM25。"""
    ensure_collection()
    metric = "BM25" if field == "sparse_vector" else "COSINE"
    # ef=64 是 HNSW 查询深度；BM25 不需要这个参数。
    params = {} if metric == "BM25" else {"ef": 64}
    return milvus_client().search(
        settings.milvus_collection,
        data=data,
        anns_field=field,
        filter=_filter(role_id, user_id),
        limit=limit,
        search_params={"metric_type": metric, "params": params},
        output_fields=["document_id", "text", "source"],
    )


def retrieve_candidates(
    query: str, query_vector: list[float], role_id: int, user_id: int
) -> tuple[list[list[dict]], list[list[dict]]]:
    """分别执行 Dense 语义召回和 BM25 关键词召回。"""
    dense_result = search_milvus(
        [query_vector], "dense_vector", role_id, user_id, settings.top_k_dense
    )
    sparse_result = search_milvus(
        [query], "sparse_vector", role_id, user_id, settings.top_k_sparse
    )
    return dense_result, sparse_result


# =============================================================================
# 第 6 步：恢复原文并通过 RRF 融合两路召回
# =============================================================================

def restore_original_text(result: list[list[dict]]) -> list[dict]:
    """从 Milvus entity 中取回文本块、文档 ID 和来源。"""
    return [
        {
            "chunk_id": str(hit["id"]),
            "document_id": int(hit["entity"]["document_id"]),
            "text": hit["entity"]["text"],
            "source": hit["entity"]["source"],
            "raw_score": float(hit.get("distance", 0.0)),
        }
        for hit in (result[0] if result else [])
    ]


def reciprocal_rank_fusion(
    result_lists: list[list[dict]], constant: int = RRF_CONSTANT
) -> list[dict]:
    """按 ``1 / (constant + rank)`` 融合不同量纲的召回排名。"""
    fused: dict[str, dict] = {}
    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            fused.setdefault(item["chunk_id"], {**item, "score": 0.0})
            # 同一个文本块被两路找到时分数累加，因此排名会自然提高。
            fused[item["chunk_id"]]["score"] += 1.0 / (constant + rank)
    return sorted(fused.values(), key=lambda item: item["score"], reverse=True)


def recall_original_text(
    dense_result: list[list[dict]], sparse_result: list[list[dict]], candidate_limit: int
) -> list[dict]:
    """恢复两路原文、RRF 去重融合并截取候选集。"""
    dense_hits = restore_original_text(dense_result)
    sparse_hits = restore_original_text(sparse_result)
    return reciprocal_rank_fusion([dense_hits, sparse_hits])[:candidate_limit]


# =============================================================================
# 第 7 步：BGE-reranker 精排
# =============================================================================

@lru_cache(maxsize=1)
def reranker_model() -> CrossEncoder:
    """延迟加载本地 BGE-reranker-base，之后复用同一实例。"""
    with _model_lock:
        return CrossEncoder(
            settings.rerank_model,
            device=settings.model_device,
            local_files_only=True,
            cache_folder=str(settings.huggingface_cache_dir),
        )


def rerank_candidates(query: str, candidates: list[dict], limit: int) -> list[dict]:
    """联合读取“问题 + 候选原文”并重新计算精确相关性。"""
    if settings.rerank_enabled and candidates:
        scores = reranker_model().predict([(query, item["text"]) for item in candidates])
        for item, score in zip(candidates, scores, strict=True):
            item["score"] = float(score)
    return sorted(candidates, key=lambda item: item["score"], reverse=True)[:limit]


def hybrid_search(query: str, role_id: int, user_id: int, top_k: int | None = None) -> list[dict]:
    """在线检索调度器：依次执行问题向量化、双路检索、融合和重排。"""
    limit = top_k or settings.top_k_final
    query_vector = encode_query(query)
    dense, sparse = retrieve_candidates(query, query_vector, role_id, user_id)
    # 至少给精排 12 条候选，避免召回阶段过早丢掉潜在正确答案。
    candidates = recall_original_text(dense, sparse, max(limit * 3, 12))
    return rerank_candidates(query, candidates, limit)


# =============================================================================
# 第 8 步：构建提示词并调用 DeepSeek
# =============================================================================

def postprocess(text: str) -> str:
    """删除模型推理标签，并压缩多余空行。"""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def build_generation_messages(
    question: str, role_prompt: str, history: list[dict], hits: list[dict]
) -> list[dict]:
    """把角色、最近 10 轮对话、知识库证据和问题组成 DeepSeek messages。"""
    context = "\n\n".join(
        f"[{index}] 来源：{hit['source']}\n{hit['text']}"
        for index, hit in enumerate(hits, 1)
    ) or "未检索到相关资料。"
    if settings.general_knowledge_fallback:
        answer_policy = (
            "优先依据给定知识库资料回答，并用[1]这样的编号引用对应资料。"
            "先判断资料是否与问题直接相关；资料不足或无关时，可以使用可靠的通用知识回答，"
            "但必须以“以下为大模型通用知识回答（本地知识库未提供直接依据）”开头，"
            "且不得给通用知识添加知识库引用。不得编造来源。"
        )
    else:
        answer_policy = "只能依据给定知识库资料回答；资料不足时明确说不知道，不得编造。"
    system_prompt = (
        f"{role_prompt}\n{answer_policy}"
        "不做诊断、不调整处方，遇到不确定或高风险问题建议咨询专业人员。"
        "末尾保留医疗免责声明。"
    )
    return [
        {"role": "system", "content": system_prompt},
        *history[-20:],
        {"role": "user", "content": f"知识库资料：\n{context}\n\n用户问题：{question}"},
    ]


def call_deepseek(messages: list[dict]) -> str:
    """调用 DeepSeek 的 OpenAI 兼容接口，低温度减少随机发挥。"""
    with httpx.Client(timeout=90) as client:
        response = client.post(
            f"{settings.deepseek_base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {settings.deepseek_api_key}"},
            json={
                "model": settings.deepseek_model,
                "messages": messages,
                "temperature": 0.2,
            },
        )
        response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def generate_answer(question: str, prompt: str, history: list[dict], hits: list[dict]) -> str:
    """安全检查后生成答案；未配置 API 时降级返回最相关知识块。"""
    if any(term in question for term in EMERGENCY_TERMS):
        return "你描述的情况可能需要紧急处理。请立即拨打 120 或前往最近的急诊。\n\n" + MEDICAL_NOTICE
    if not settings.llm_enabled or not settings.deepseek_api_key:
        evidence = hits[0]["text"] if hits else "知识库中没有找到足够信息。"
        return f"当前未配置 DeepSeek API，先返回最相关资料：\n\n{evidence}\n\n{MEDICAL_NOTICE}"
    messages = build_generation_messages(question, prompt, history, hits)
    answer = postprocess(call_deepseek(messages))
    return answer if MEDICAL_NOTICE in answer else f"{answer}\n\n{MEDICAL_NOTICE}"
