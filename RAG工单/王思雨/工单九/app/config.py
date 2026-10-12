# 工单编号：人工智能NLP-RAG-Graph RAG优化任务
# 配置层：本单 .env 优先，未定义的项一律回落到 08 工单 config（复用其 Milvus/Redis/模型/凭据口径）
import os, sys, unicodedata; from pathlib import Path; from dotenv import load_dotenv  # 环境变量与搜索路径 / 文本归一化 / 路径对象（规避中文路径转义）/ 载入 .env 密钥
ROOT = Path(__file__).resolve().parent.parent                                 # 工单九根目录
DATA, EVAL, FRONT = ROOT / "data", ROOT / "eval", ROOT / "frontend"           # 数据目录 / 评估结果目录 / 前端目录
load_dotenv(ROOT / ".env", override=False)                                    # 先载入本单 .env；已存在的真实环境变量优先，不被覆盖
sys.path.insert(0, str(ROOT.parent))                                          # 把 RAG 目录加入搜索路径，使 `工单八` 成为可导入的命名空间包
import 工单八.app.config as C8                                                # 08 工单配置：Milvus 集合 gd8_chunks、Redis、模型路径与全部检索阈值
def _e(k, d=""): return os.getenv(k, d).strip()                               # 本单取参口：去首尾空白，防止 .env 行尾空格污染路径
def fix(s): return unicodedata.normalize("NFKC", s or "").translate({0x2EDB: "风", 0x2ED3: "长"})  # 文本归一化：本附件把"银行"输出成康熙部首"银⾏"（NFKC 实测修 23 种），CJK 部首补充区的 ⻛/⻓ NFKC 不管须手工补；不归一化会让 BM25 与实体链接整体失配
BGE_M3, BGE_RERANKER = _e("BGE_M3_PATH", C8.EMBED_MODEL_PATH), _e("BGE_RERANKER_PATH", C8.RERANK_MODEL_PATH)  # 向量/重排模型路径（本单变量名优先，缺省回落 08）
SAMPLE_PDF = Path(_e("SAMPLE_Q_PDF", str(DATA / "sample_questions.pdf")))      # 工单附件题集：环境变量 SAMPLE_Q_PDF 可覆盖
PARSED_JSON, GRAPH_NX, ROOT08_GRAPH = DATA / "sample_questions.parsed.json", DATA / "graph_nx.json", C8.DATA / "graph.json"  # 题集解析缓存 / 本单 NetworkX 图谱（与 08 分开，互不污染）/ 08 已建图谱（本单在其抽取结果上重表示为 NetworkX，不重复调大模型）
API_PORT = int(_e("API_PORT", "8012"))                                        # 端口 8012（8000 工单五、8010 工单七、8011 工单八已占）
HOPS, EXPAND_TERMS = int(_e("HOPS", "2")), int(_e("EXPAND_TERMS", "8"))       # 【优化·检索层】图扩展跳数 1→2、注入 BM25 的邻居词 6→8（治多期聚合类召回短板）
VEC_W, BM25_W = float(_e("VEC_W", "0.6")), float(_e("BM25_W", "0.4"))         # 【优化·检索层】融合权重：08 为 0.7/0.3，数值与表格型问题靠关键词更稳，故下调向量权重
FUSION_K, RERANK_N = int(_e("FUSION_K", "40")), int(_e("RERANK_N", "6"))      # 【优化·检索层】融合候选 30→40（多留候选给重排裁）、喂模型片段 8→6（直接提 context_precision）
COMM_MAX, COMM_MIN = int(_e("COMM_MAX", "12")), int(_e("COMM_MIN", "3"))      # 【优化·图谱层】最多生成摘要的社区数 / 社区最小成员数（过小社区摘要无信息量）
USE_COMM = _e("USE_COMM", "1") == "1"                                         # 【优化·检索层】社区摘要 global 通道开关（消融实验用）
DEADLINE_S, TARGET_MS = float(_e("DEADLINE_S", "3.0")), int(_e("TARGET_MS", "3000"))  # 端到端预算（秒，超预算走降级）/ 工单验收目标 3000ms
