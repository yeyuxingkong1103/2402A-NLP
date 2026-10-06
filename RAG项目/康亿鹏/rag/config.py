"""全局配置：常量直接写在本文件，仅 DeepSeek 密钥与 API 地址从环境变量读取。"""  # 模块说明：集中管理项目所有配置项
import os  # 仅用于读取 DeepSeek 相关环境变量
from pathlib import Path  # 用于处理文件路径

base_dir = Path(__file__).resolve().parent  # 项目根目录（本文件所在目录的绝对路径）


# ---------- DeepSeek（唯一从环境变量读取的部分） ----------
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY1", "").strip()  # DeepSeek API 密钥（必填，去首尾空白）
deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")  # DeepSeek API 服务地址
deepseek_model = "deepseek-chat"  # 使用的模型；也可改为 deepseek-reasoner
llm_temperature = 0.1  # 生成温度：越低回答越确定
llm_max_tokens = 2048  # 回答的最大 token 数

# ---------- Embedding / Reranker ----------
bge_m3_model = r"D:\models\bge-m3"  # BGE-M3 本地模型路径（也可填 HuggingFace ID）
bge_reranker_model = r"D:\models\bge-reranker-large"  # BGE 重排模型名
device = "cuda"  # 运行设备：cuda / cpu
use_fp16 = True  # 是否启用半精度；cpu 下会自动降级为 fp32

# ---------- Milvus ----------
milvus_uri = "http://localhost:19530"  # Milvus 服务地址

# ---------- 检索 / 切分 ----------
retrieve_k = 10  # 稠密向量召回条数（初筛候选数量）
sparse_k = 10  # BM25 稀疏召回条数（关键词/专名匹配候选数量）
rrf_k = 60  # RRF 融合常数：两路结果按 1/(rrf_k + 名次) 累加，越大名次差异越平滑
rerank_top_n = 3  # 重排后保留条数（最终送入 LLM 的数量）
chunk_size = 512  # 每个片段的最大字符数
chunk_overlap = 50  # 相邻片段间的重叠字符数（避免语义被切断）

# ---------- 文档目录 ----------
docs_dir = base_dir / "docs"  # 默认文档目录（项目根下的 docs/）

# ---------- MinerU 云端文档解析（parse_pdf.py 使用，在 mineru 环境运行） ----------
mineru_api_key = os.getenv("MINERU_API_KEY", "").strip()  # 云端 API 密钥（从环境变量读取，禁止写明文；申请：https://mineru.net/apiManage/token）
mineru_api_url = "https://mineru.net/api"  # 云端精准解析 API 地址
mineru_tier = "standard"  # 云端 v1 API 最高开放档位（小模型+VLM 混合，high）；advanced 仅本地/自建服务可用，云端会报 quality_tier_unavailable
mineru_include_images = False  # 解析结果是否包含图片（RAG 入库不需要）
mineru_pdf_dir = base_dir / "docs" / "医疗数据"  # 待解析 PDF 所在目录（parse_pdf.py 递归扫描其中全部 .pdf）
mineru_out_dir = base_dir / "docs_mineru"  # 解析出的 Markdown 保存目录（供 main.py ingest 入库）

# ---------- Redis（对话历史） ----------
redis_host = "localhost"  # Redis 服务地址
redis_port = 6379  # Redis 端口
redis_db = 0  # Redis 数据库编号
history_ttl = 60 * 60 * 24 * 7  # 对话历史保留时长（秒，默认 1 周），过期自动清理
history_turns = 10  # 每个领域保留最近 10 轮对话（每轮 = 1 问 1 答）

# ---------- 领域集合映射 ----------
domain_collections = {  # 领域名 -> Milvus 集合名
    "medical": "rag_medical",  # 医疗知识库集合
    "education": "rag_education",  # 教育知识库集合
}

# ---------- MySQL（用户认证持久层） ----------
mysql_host = "localhost"  # MySQL 服务地址
mysql_port = 3306  # MySQL 端口
mysql_user = "root"  # MySQL 用户名
mysql_password = "123456"  # MySQL 密码（改成你自己的实际密码）
mysql_db = "rag项目"  # 数据库名（需先在 MySQL 中创建：CREATE DATABASE rag CHARACTER SET utf8mb4）

# ---------- JWT（认证令牌） ----------
jwt_secret = "change-me-in-production"  # JWT 签名密钥（生产环境务必修改）
jwt_expire_hours = 24  # JWT 过期时间（小时）
