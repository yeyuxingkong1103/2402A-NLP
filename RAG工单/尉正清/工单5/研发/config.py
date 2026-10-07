# 工单编号：人工智能NLP-RAG-Query 理解优化任务
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

# ---------- 向量模型：本地 BGE-M3 ----------
BGE_M3_PATH = os.getenv("BGE_M3_PATH", r"D:\Pycharm\yzq\models\bge-m3")
USE_FP16 = True
RRF_K = 60                # 混合检索的 RRF 常数
DENSE_WEIGHT = 0.7        # 稠密(语义)权重
SPARSE_WEIGHT = 0.3       # 稀疏(关键词)权重

# ---------- 两阶段检索（工单2 优化点）----------
# 粗排召回 RECALL_K 个候选，重排序可选用交叉编码器精排，再取 TOP_K 个送进大模型。
RECALL_K = 20
TOP_K = 8
RERANKER_PATH = os.getenv("RERANKER_PATH", r"D:\Pycharm\yzq\models\bge-reranker-v2-m3")

# ⚠️ 重排序默认关闭 —— 这是本工单实测后的结论，不是没做完。
# 在招股书这 10 个问题上，bge-reranker-v2-m3 会系统性地把「语义相近的散文」
# 排在「真正含答案的财务表格」之前（例：问军用领域收入，它把第 378 页
# 「直接军方+间接军方合计」提到第 343 页汇总表之前，两者口径不同、数值差 0.01）。
# 实测对比：
#     配置              准确率   检索精确率   平均响应   最差
#     关闭重排序        100%     22%         1.69s     2.22s
#     开启(取4块)        90%     45%         2.52s     3.36s
#     开启(取8块)        90%     19%         3.19s    11.51s
# 开启后准确率、精确率、响应时间三项全输，故默认关闭。交叉编码器对数字表格
# 不敏感是已知问题，若换用叙述型文档可把开关打开再评估。
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "false").lower() == "true"
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

# ---------- 知识库缓存 / 反馈 ----------
CACHE_ROOT = BASE_DIR / "kb_cache"
FEEDBACK_LOG = BASE_DIR / "feedback.jsonl"

# ---------- 工单指定的 16 个测试问题（6 题力源信息 + 10 题兴图新科）----------
TEST_QUESTIONS = [
    {"id": 5,   "question": "武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成，其中大客户销售部有几个销售处构成？"},
    {"id": 6,   "question": "武汉力源信息技术股份有限公司招股意向书中，从2008年中国IC市场应用结构与增长图中可以看出，增长率最快的是哪个行业？负增长的是哪个行业？"},
    {"id": 1,   "question": "武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？"},
    {"id": 2,   "question": "武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目？"},
    {"id": 3,   "question": "与武汉力源信息技术股份有限公司存在控制关系的关联方是谁，持股比例和本公司关系是什么？"},
    {"id": 4,   "question": "与武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些？"},
    {"id": 260, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"},
    {"id": 95,  "question": "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？"},
    {"id": 33,  "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？"},
    {"id": 34,  "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？"},
    {"id": 957, "question": "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？"},
    {"id": 793, "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？"},
    {"id": 795, "question": "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？"},
    {"id": 543, "question": "武汉兴图新科电子股份有限公司注册资本是多少？"},
    {"id": 531, "question": "武汉兴图新科电子股份有限公司法定代表人是谁？"},
    {"id": 207, "question": "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？"},
]


# ---------- 工单问题的答案页（用于界面显示「检索精确度」）----------
# 工单产出物要求「针对如下问题进行检索，显示检索到的答案及检索精确度」。
# 精确度 = 召回块里落在答案页上的比例，需要知道标准答案页才能算，故内置在此。
ANSWER_PAGES = {
    5: [39], 6: [72],
    1: [2, 22, 24], 2: [22, 24], 3: [157], 4: [157],
    260: [343], 95: [181], 33: [343], 34: [152], 957: [181],
    793: [152], 795: [181], 543: [22, 52], 531: [22, 52], 207: [30, 479, 490],
}

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
