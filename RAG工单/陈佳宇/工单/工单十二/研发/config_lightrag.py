# config_lightrag.py 工单12 LightRAG配置
import os

# ========== 路径配置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR,"lightrag_cache")
PDF_DIR = os.path.join(BASE_DIR,"pdf")

# ========== Embedding：复用工单11微调后的BGE模型 ==========
EMBEDDING_MODEL_PATH = "/mnt/d/models/bge-base-zh-v1.5"

# ========== LLM配置（Ollama，WSL网关IP） ==========
LLM_MODEL = "qwen2:7b"
LLM_HOST = "http://10.255.255.254:11434/v1"

# ========== 【工单核心】中文金融实体关系抽取Prompt，替换LightRAG默认英文Prompt ==========
ENTITY_EXTRACT_PROMPT = """
你是金融招股说明书知识抽取专家。
从下面文本提取实体与实体之间的关系。
【允许实体类型只能选用下面列表内的】
["公司","部门","人物","募集投资项目","行业","产品","财务指标","关联方","销售处","业务板块"]

【允许关系类型只能选用下面列表内的】
["持股","隶属","投资","属于","参与制定标准","收入来源","业务覆盖","构成"]

输出严格为json数组，每一条字段："entity_name","entity_type","related_entity_name","relation_type"
不要额外解释，不要英文，严禁编造原文不存在的实体和关系。
文本内容：
{input_text}
"""

# 文本分块参数
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
