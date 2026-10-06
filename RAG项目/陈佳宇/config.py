import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = os.getenv("DEEPSEEK_URL", "https://api.deepseek.com/v1/chat/completions")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
MILVUS_URI = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")
EMB_MODEL_PATH = os.getenv("EMB_MODEL_PATH", "")
RERANK_MODEL_PATH = os.getenv("RERANK_MODEL_PATH", "")
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "rag_character")
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
HISTORY_MAX_TURNS = int(os.getenv("HISTORY_MAX_TURNS", 5))
JWT_SECRET = os.getenv("JWT_SECRET", "rag_character_default_secret")
JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", 24))
