# 工单编号：人工智能NLP-RAG项目-LightRAG优化
"""LightRAG 引擎接线：大模型 / 嵌入模型 / neo4j 图存储 / 招股书定制的抽取 schema

四件事，都在这个文件里：

1. **大模型接线**：deepseek-flash。⚠️ 关键参数 `reasoning_effort=none` ——
   它是推理模型，默认每次调用要烧掉一万多 reasoning token（实测单次 33 秒）；
   关掉之后 1.9 秒，抽取质量对结构化任务没有肉眼可见的差别。建图要上千次调用，
   不关掉这一条整个工单跑不完（详见 优化/过程问题记录.md 问题 1）。

2. **嵌入接线**：bge-m3，与 RAG 那一路用同一个模型 —— 两边嵌入不同的话，
   检索结果的差异就分不清是「图结构带来的」还是「嵌入模型带来的」。

3. **图存储**：Docker 里的 neo4j。LightRAG 默认用 NetworkX 存本地文件，
   换成 neo4j 才能算「知识图谱」这个产出物。

4. **抽取 schema**：把默认的通用实体类型（Person/Organization/Concept…）
   换成招股说明书领域的类型。这是工单验收点 1 要求的那件事，
   见下面的 ENTITY_TYPES_GUIDANCE。
"""
import os
from pathlib import Path

from lightrag import LightRAG
from lightrag.kg.neo4j_impl import Neo4JStorage
from lightrag.prompt import PROMPTS
from lightrag.utils import EmbeddingFunc
from openai import AsyncOpenAI

from config import (LLM_API_BASE, LLM_API_KEY, LLM_MODEL, WORKING_DIR,
                    BGE_M3_PATH, LLM_MAX_TOKENS)

# ---------------- 招股说明书定制的实体类型（验收点 1） ----------------
# 默认那套是通用的（Person / Creature / Location / Event…），用在招股书上
# 会把「军工资质」「募投项目」「技术标准」全塞进 Other，图谱失去区分度。
# 这里换成招股书真正会出现的类型，并写明判据 —— 不写判据模型就会乱归类。
ENTITY_TYPES_GUIDANCE = """\
Classify each entity using one of the following types. If no type fits, use `Other`.

- Company: 发行主体及其子公司、参股公司、控股股东、实际控制人控制的其他企业、
  竞争对手、客户、供应商等一切企业法人
- Person: 董事、监事、高级管理人员、核心技术人员、自然人股东
- Institution: 政府机关、监管机构、行业协会、交易所、大学及科研院所
  （注意：本身是企业的不算，走 Company）
- Product: 具体产品或服务，如某型号芯片、某软件产品、某项技术服务
- Technology: 技术、专利、软件著作权、技术标准、工艺方法
- Qualification: 资质与认证，如武器装备科研生产许可证、高新技术企业证书、
  保密资格、质量体系认证
- Industry: 行业、细分市场、应用领域，如电子信息行业、军用领域、IC 市场
- Project: 募集资金投资项目、研发项目、工程项目
- FinancialMetric: 有明确数值的财务与经营指标，如营业收入、毛利率、
  注册资本、本次发行股数、募集资金总额
- Period: 报告期、会计年度、具体时点，如 2016 年度、报告期内、2018 年末
- Location: 地理位置的省、市、园区
- Chart: 文档中的图表，如组织结构图、市场结构与应用增长图、行业产业链图
  （招股书里大量关键信息只存在于图中，单列一类便于检索时定位）
"""

# 在默认抽取指令之上追加的招股书专用规则。
# 只追加、不替换 —— 默认 prompt 里那套输出格式与字段定义是 LightRAG
# 解析器认的，改了会解析失败。
_EXTRA_RULES = """

---Prospectus-Specific Rules---
7. **Resolve Self-References (招股说明书最关键的一条):**
  - Prospectus text refers to the issuer as `本公司`, `公司`, `发行人`, `股份公司`.
    **Never** emit those as entity names. Always resolve them to the issuer's full
    registered name given in the `---Section Context---` or, when that is not
    available, to `{issuer}`.
  - The same applies to `子公司`, `控股股东`, `实际控制人` when the text names them
    elsewhere — use the concrete name instead of the generic role.

8. **Preserve Chinese Proper Nouns Verbatim:**
  - Entity names must be copied exactly as they appear in the source, in Chinese,
    including the full legal suffix (e.g. `武汉兴图新科电子股份有限公司`).
    Do not translate, abbreviate, or re-order. An abbreviation may appear in the
    description, but the `name` field stays the full form.

9. **Chart-Derived Facts:**
  - Text blocks marked as coming from a chart (图表解析) are legitimate sources.
    Extract entities and relationships from them like any other text.

10. **Numbers Belong in Descriptions:**
  - Quantitative facts (amounts, ratios, dates, counts) must be written into the
    `description` field verbatim with their unit and period, because downstream
    question answering has to read the number straight out of the graph.
"""


def install_prospectus_prompts(issuer_hint: str = "本次发行的发行人"):
    """把招股书专用的实体类型与规则装进 LightRAG 的 prompt 表。

    必须在构造 LightRAG 之前调用 —— prompt 是在实例初始化时读进内存的。
    """
    PROMPTS["default_entity_types_guidance"] = ENTITY_TYPES_GUIDANCE
    for key in ("entity_extraction_system_prompt",
                "entity_extraction_json_system_prompt"):
        if key in PROMPTS and "Prospectus-Specific Rules" not in PROMPTS[key]:
            PROMPTS[key] = PROMPTS[key] + _EXTRA_RULES.replace(
                "{issuer}", issuer_hint)


# ---------------- 大模型 ----------------
_client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE)


async def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
    """LightRAG 调用的大模型入口。

    签名要与 LightRAG 期望的一致：返回纯文本，不要包成对象。

    reasoning_effort=none 走 extra_body 而不是具名参数：这个端点是
    OpenAI 兼容接口，具名参数会被 SDK 按 OpenAI 的类型校验拦下
    （官方取值为 low/medium/high，不含 none）。
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages or [])
    messages.append({"role": "user", "content": prompt})

    resp = await _client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=kwargs.get("max_tokens", LLM_MAX_TOKENS),
        temperature=kwargs.get("temperature", 0.0),
        extra_body={"reasoning_effort": "none"},
    )
    return resp.choices[0].message.content or ""


# ---------------- 嵌入 ----------------
_st_model = None


def _embedding_model():
    """bge-m3 全局只加载一次（1.2GB，重复加载会把显存吃光）。"""
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer(str(BGE_M3_PATH), device="cuda")
        _st_model.max_seq_length = 8192       # bge-m3 支持 8k，招股书块较长
    return _st_model


async def embed_func(texts):
    import numpy as np
    model = _embedding_model()
    vecs = model.encode(list(texts), batch_size=8, normalize_embeddings=True,
                        show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)


# bge-m3 稠密向量维度固定 1024，与 RAG 那一路一致
EMBEDDING = EmbeddingFunc(embedding_dim=1024, max_token_size=8192,
                          func=embed_func, model_name="bge-m3")


# ---------------- 引擎 ----------------
def build_engine(issuer_hint: str = "本次发行的发行人") -> LightRAG:
    """构造接好 neo4j 的 LightRAG 实例。

    neo4j 连接信息走环境变量（LightRAG 内部就是这么读的），
    这里做一次转写并给出默认值，省得每台机器都要先 export 一遍。
    """
    os.environ.setdefault("NEO4J_URI", os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    os.environ.setdefault("NEO4J_USERNAME", os.getenv("NEO4J_USERNAME", "neo4j"))
    os.environ.setdefault("NEO4J_PASSWORD", os.getenv("NEO4J_PASSWORD", "neo4j123"))
    os.environ.setdefault("NEO4J_DATABASE", os.getenv("NEO4J_DATABASE", "neo4j"))

    install_prospectus_prompts(issuer_hint)

    work = Path(WORKING_DIR)
    work.mkdir(parents=True, exist_ok=True)

    return LightRAG(
        working_dir=str(work),
        graph_storage="Neo4JStorage",
        llm_model_func=llm_func,
        llm_model_name=LLM_MODEL,
        llm_model_max_async=int(os.getenv("LIGHTRAG_LLM_ASYNC", "16")),
        embedding_func=EMBEDDING,
        embedding_func_max_async=8,
        # 关掉追问式二次抽取：一次调用变成两次，建图时间翻倍，
        # 而招股书是格式规范的文本，首轮抽取的召回已经够全。
        entity_extract_max_gleaning=0,
        # 关掉「同一实体描述合并时再叫一次大模型」的强制阈值：
        # 招股书里公司名反复出现，按默认阈值 8 会触发上千次摘要调用。
        force_llm_summary_on_merge=9999,
        max_parallel_insert=int(os.getenv("LIGHTRAG_INSERT_ASYNC", "4")),
        default_llm_timeout=300,
        log_level="INFO",
    )


__all__ = ["build_engine", "install_prospectus_prompts", "ENTITY_TYPES_GUIDANCE",
           "llm_func", "EMBEDDING"]
