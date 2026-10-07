# 工单编号：人工智能NLP-RAG-功能测试及评估
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


# ---------- 知识库缓存 / 反馈 ----------
CACHE_ROOT = BASE_DIR / "kb_cache"
FEEDBACK_LOG = BASE_DIR / "feedback.jsonl"

# ---------- 工单指定的测试问题 ----------
# 唯一来源是 ccf_testset.TEST_SET（工单7 的测试用例集），这里只做一次转写，
# 不再另抄一份 —— 抄一份迟早会和测试集对不上，而界面下拉框和评估脚本
# 必须问的是同一批题。
from ccf_testset import TEST_SET as _TEST_SET           # noqa: E402

TEST_QUESTIONS = [{"id": t["id"], "question": t["question"]} for t in _TEST_SET]

# ---------- 工单问题的答案页（用于界面显示「检索精确度」）----------
# 产出物要求「针对如下问题进行检索，显示检索到的答案及检索精确度」。
# 精确度 = 召回块里落在答案页上的比例，需要知道标准答案页才能算。
#
# 工单7 的知识库是 9 份年报合一，页码在不同文档间会重复（平安银行第 22 页
# 和邮储银行第 22 页是两回事），所以答案页必须带上是**哪份文档**的页码；
# 跨文档题的标准页分属多份文档，界面上的精确度对它没有意义，排除在外。
ANSWER_PAGES = {}
ANSWER_DOC = {}
for _t in _TEST_SET:
    if len(_t["gold_pages"]) == 1:
        _doc, _pages = next(iter(_t["gold_pages"].items()))
        ANSWER_DOC[_t["id"]] = _doc
        ANSWER_PAGES[_t["id"]] = _pages

# ---------- 界面文案（中英双语，验收标准要求支持中英文问答） ----------
I18N = {
    "upload":       ("上传 PDF（必填）", "Upload PDF (required)"),
    "init":         ("初始化知识库", "Build knowledge base"),
    "kb_select":    ("选择已建知识库", "Select an existing knowledge base"),
    "kb_load":      ("加载", "Load"),
    "kb_delete":    ("删除", "Delete"),
    "status":       ("状态", "Status"),
    "pick_q":       ("选择工单测试问题", "Pick a work-order test question"),
    "question":     ("请输入问题（中文或英文均可）", "Ask a question (Chinese or English)"),
    "mode":         ("回答模式", "Mode"),
    "mode_rag":     ("RAG（BGE-M3 检索）", "RAG (BGE-M3 retrieval)"),
    "mode_llm":     ("纯 LLM（不检索）", "LLM only (no retrieval)"),
    "audio":        ("语音输入（可选）", "Voice input (optional)"),
    "ask":          ("提问", "Ask"),
    "answer":       ("回答", "Answer"),
    "context":      ("检索到的上下文", "Retrieved context"),
    "meta":         ("元信息", "Metadata"),
    "good":         ("👍 回答正确", "👍 Good"),
    "bad":          ("👎 回答有误", "👎 Bad"),
    "feedback_ok":  ("感谢反馈，已记录。", "Thanks, your feedback was recorded."),
}
