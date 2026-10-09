# 工单编号：人工智能NLP-RAG项目-LightRAG优化
"""全局配置"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

# ---------- 文本分块 ----------
# 分块要够大：财务表格的标题往往在表格上一页或上一段，块太小会把标题和表格
# 拆散，表格块就只剩一堆数字，语义检索完全找不到。实测 500/80 时正确表格排在
# 第 209 名，800/200 才能进入 Top-10。
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

# ---------- 向量模型 ----------
# 具体用哪个模型见下面的 EMBEDDING_MODELS 注册表
USE_FP16 = True
RRF_K = 60                # 混合检索的 RRF 常数
DENSE_WEIGHT = 0.7        # 稠密(语义)权重
SPARSE_WEIGHT = 0.3       # 稀疏(关键词)权重

# ---------- 两阶段检索（工单2 优化点）----------
# 粗排召回 RECALL_K 个候选，重排序可选用交叉编码器精排，再取 TOP_K 个送进大模型。
RECALL_K = 20
TOP_K = 8
RERANKER_PATH = os.getenv("RERANKER_PATH", r"D:\Pycharm\yzq\models\bge-reranker-v2-m3")

# 送给重排序模型的正文上限。模型只看 512 token，喂更长也会被截断，
# 白白拉长前向时间（实测 20 候选喂全长要 1.2 秒，截断到 400 字只需 0.51 秒）。
RERANK_MAX_CHARS = 400


# ---------- 图像内容解析（工单4）----------
# 招股书里的组织结构图、市场结构图，数据不在文本层，必须让多模态模型读图后
# 转成文字再入库。工单备注要求「使用多模态模型（CLIP 或多模态大模型）实现」。
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "deepseek-flash")   # 支持图像输入的模型
IMAGE_DPI = 150          # 渲染清晰度：太低图表里的字看不清，太高请求体过大
MAX_CHART_PAGES = 40     # 单份文档最多解析多少页图表，控制耗时

# ---------- 大模型 ----------
LLM_API_BASE = os.getenv("DEEPSEEK_BASE_URL")
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY")
LLM_MODEL = "deepseek-flash"
LLM_TIMEOUT = 30
# deepseek-flash 是推理模型：reasoning token 先消耗额度，给小了正文会返回空字符串。
# 上下文较长时 reasoning 能吃掉上千 token，所以必须给足（实测 1024 会返回空）。
LLM_MAX_TOKENS = 4096

# 推理开关：none 表示关掉思维链。
#
# ⚠️ 这不只是性能优化，更是**对比公平性**的要求：本工单要比 RAG 与 LightRAG，
# 两边必须跑完全相同的大模型配置，否则指标差异里混进了「谁的模型配置更强」。
# 而 LightRAG 那边不关掉推理根本跑不完（建图上千次调用，单次 33 秒），
# 所以统一关掉，两边一致。
#
# 关掉后单次调用从 33 秒降到 1.9 秒，reasoning token 归零，正文照常产出。
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "none")



# ---------- 混合检索（工单6）----------
# 检索策略：vector 向量检索 / fulltext 全文检索 / hybrid 混合检索
RETRIEVAL_MODE = "hybrid"
# 混合检索里「向量那一路」的权重，全文那一路取 1-alpha
RERANK_ALPHA = 0.5
# 融合算法：weighted 加权平均 / rrf 名次融合 / vote 投票制
HYBRID_FUSION = "weighted"
# 重排算法：none / tfidf / feedback / llm / cross
RERANK_METHOD = "none"

# 可供选择的嵌入模型（工单要求「支持多种嵌入模型，如 bge、m3e 及其他」）。
# 每项写明**加载方式**：不同模型的接口不一样，不能一视同仁。
#   flagembedding         BGE-M3 走 FlagEmbedding，能同时给出稠密 + 稀疏向量
#   sentence_transformers m3e 这类普通句向量模型，只有稠密向量
EMBEDDING_MODELS = {
    "bge-m3": {"path": os.getenv("BGE_M3_PATH", "D:/Pycharm/yzq/models/bge-m3"),
               "backend": "flagembedding"},
    "m3e-base": {"path": os.getenv("M3E_PATH", "D:/Pycharm/yzq/models/m3e-base"),
                 "backend": "sentence_transformers"},
}
EMBEDDING_MODEL = "bge-m3"


# ---------- LightRAG（工单12 新增）----------
# 嵌入模型沿用 RAG 那一路的 bge-m3：两边嵌入不同的话，检索结果的差异就分不清
# 是「图结构带来的」还是「嵌入模型带来的」。
BGE_M3_PATH = Path(EMBEDDING_MODELS["bge-m3"]["path"])
# LightRAG 自己的运行目录（KV/向量库/doc-status 存本地，图存 neo4j）
WORKING_DIR = Path(os.getenv("LIGHTRAG_WORKING_DIR", str(BASE_DIR / "lightrag_cache")))

# 图存储：Docker Desktop 里的 neo4j。容器用 NEO4J_AUTH=neo4j/neo4j123 起，
# 端口 7474(HTTP)/7687(bolt) 映射到宿主机。
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j123")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# 查询模式：mix = 局部(实体/向量) + 全局(关系/主题) 两条路都走，
# 对应工单描述的「双层检索机制」。可选 local / global / hybrid / naive / mix。
LIGHTRAG_QUERY_MODE = os.getenv("LIGHTRAG_QUERY_MODE", "mix")


# ---------- 知识库缓存 / 反馈 ----------
# 路径可用环境变量覆盖：容器部署时知识库放在 Docker 卷上，
# 卷挂在 /kb，与代码目录分开，重建镜像不会把库弄丢。
CACHE_ROOT = Path(os.getenv("KB_CACHE_DIR", str(BASE_DIR / "kb_cache")))
FEEDBACK_LOG = Path(os.getenv("FEEDBACK_LOG", str(BASE_DIR / "feedback.jsonl")))


