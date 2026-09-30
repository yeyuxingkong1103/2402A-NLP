"""configs/settings.py —— 全局配置（pydantic-settings 读取 .env 与环境变量）。

在链路中的位置（被全项目引用）：
    src/offline/*、src/online/*、src/memory/*、src/api/*、src/models/database.py
    几乎每个模块都通过 get_settings() 取配置

这是本项目的"配置总台"：**能在不改代码的前提下把整套系统从"完整形态"
降到"能跑通的最小形态"的开关全部在这里**。
三个降级开关（README 里"资源不足时可使用降级配置"指的就是它们）：
    EMBEDDING_BACKEND=hash     免下载 BGE-M3 模型，用哈希假向量
    LLM_BACKEND=mock           免本地大模型，生成步骤直接返回固定文本
    MYSQL_URL=sqlite:///...    免 MySQL，改用单文件 SQLite
三个一起打开，就得到一套"零外部依赖也能端到端跑通"的配置。

读取优先级（pydantic-settings 的规则）：环境变量 > .env 文件 > 这里的默认值。

关于 alias 的用法：
    带 alias 的字段，环境变量名就是 alias 值（如 secret_key 读 JWT_SECRET_KEY）；
    不带 alias 的字段，环境变量名就是字段名的大写形式（如 port 读 PORT）。
    这套映射让 .env 里的键名可以按习惯来写。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent  # 本文件在 configs/ 下，上跳一级即项目根


class Settings(BaseSettings):
    """全部配置项。

    model_config 三个设置各自的作用：
        env_file          从项目根的 .env 读取
        extra="ignore"    .env 里有本类没定义的键时忽略而不报错 ——
                          这让 .env 可以留一些给其他组件（如 Docker Compose）用的变量
        case_sensitive=False  环境变量名大小写不敏感，MYSQL_URL 与 mysql_url 等价
    """

    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore", case_sensitive=False)

    # ---------------- 应用与服务
    app_name: str = "RAG Roleplay System"
    environment: Literal["dev", "test", "prod"] = "dev"
    # 注意 environment 会影响安全行为：src/api/deps.py 里的 X-User-Id 开发旁路
    # 在 environment == "prod" 时被强制关闭。所以生产部署必须把它设成 prod，
    # 否则任何人都能用一个请求头伪造身份。

    host: str = "127.0.0.1"
    port: int = 8902  # 本线（src 新架构）用 8902，backend 主线用 8901 —— 分开端口两条线才能同时运行。
                      # 原先是两边都用 8901，结果后启动的那个会 bind 失败（Errno 10048），
                      # 且 uvicorn 先打印 "Application startup complete" 再报错，看起来像启动成功了

    # ---------------- 认证与鉴权
    secret_key: str = Field(default="change-this-secret-with-at-least-32-bytes", alias="JWT_SECRET_KEY")
    # JWT 签名密钥。**生产环境必须替换**：这个默认值是公开的，
    # 沿用默认值等于任何人都能自签一个合法令牌。
    # 建议至少 32 字节（HS256 的安全强度上限就是密钥长度）。

    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 1440  # 令牌有效期，默认 24 小时。生产环境建议调短以缩小泄露窗口
    allow_dev_header_auth: bool = Field(default=False, alias="ALLOW_DEV_HEADER_AUTH")
    # 开发旁路开关：打开后可以用 X-User-Id 请求头直接指定身份，不必先登录取令牌。
    # 默认关闭；且即便打开，在 environment == "prod" 时也会失效（双重保险）。

    # ---------------- 存储
    mysql_url: str = Field(default="sqlite:///./data/rag_roleplay.sqlite3", alias="MYSQL_URL")
    # 关系库连接串。**默认就是降级形态**（SQLite 单文件），
    # 换成 mysql+pymysql://user:pass@host/db 即切到 MySQL，代码不用改。

    redis_url: str = Field(default="redis://127.0.0.1:6379/0", alias="REDIS_URL")
    # 短期记忆与检索缓存。连不上时由 src/memory/short_term.py 自动降级到进程内字典。

    milvus_uri: str = Field(default="http://127.0.0.1:19530", alias="MILVUS_URI")
    milvus_token: str = Field(default="", alias="MILVUS_TOKEN")  # 本地无鉴权部署留空
    milvus_collection: str = "kb_chunks"          # 知识库集合（新架构用）
    milvus_memory_collection: str = "role_memories"  # 长期记忆集合
    tenant_id: str = "default"                    # 默认租户，构建管线未显式指定时用它

    # ---------------- 向量化
    embedding_backend: Literal["flag", "api", "hash"] = "flag"
    # flag 本地 BGE-M3 / api 远程接口 / hash 纯哈希降级。
    # 资源不足时改成 hash 就能免去模型下载，流程照常跑通（检索质量下降）。

    embedding_model: str = "BAAI/bge-m3"          # 输出 1024 维，必须与 Milvus 集合的向量维度一致
    embedding_device: str = "cpu"                 # cpu / cuda:0。非 cpu 时 embedder 会自动开半精度
    embedding_api_url: str = "http://127.0.0.1:11434/api/embed"  # embedding_backend=api 时的远端地址

    # ---------------- 精排
    rerank_model: str = "BAAI/bge-reranker-v2-m3"  # cross-encoder 精排模型
    rerank_device: str = "cpu"

    # ---------------- 大模型
    llm_backend: Literal["local", "api", "ollama", "mock"] = "ollama"
    # 四个取值，但 src/online/llm.py 只实现了三条分支：
    #   ollama -> 走 Ollama 原生 /api/chat 协议
    #   mock   -> 不发请求，返回固定文本（免大模型跑通流程用）
    #   local / api（以及任何其他值）-> 都走 OpenAI 兼容的 /chat/completions 分支
    # 也就是说 local 与 api 当前是同一套实现，靠 llm_base_url 区分指向哪里。

    llm_base_url: str = "http://127.0.0.1:8000/v1"  # OpenAI 兼容服务的基址（会自动拼 /chat/completions）
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_model: str = "qwen2.5-7b-instruct"
    ollama_chat_url: str = "http://127.0.0.1:11434/api/chat"

    # ---------------- 记忆与对话
    short_memory_rounds: int = 10   # 短期记忆保留轮数。一轮 = 一问一答，故实际保留 20 条消息
    max_message_chars: int = 4000   # 单条消息长度上限，防止超长输入打爆上下文窗口

    # ---------------- 检索
    retrieval_top_k: int = 12                          # 双路召回条数
    rerank_top_k: int = 6                              # 精排后保留条数
    similarity_threshold: float = 0.25                 # 相似度门限，低于它视为不可用依据
    # 注意角色卡（src/schemas/roleplay.py 的 RoleCard）也带一份同名默认值 0.25，
    # 角色卡显式给出时会覆盖这里的全局值。

    # ---------------- 路径
    log_dir: Path = ROOT_DIR / "logs"                  # 应用日志目录（按天滚动，保留 14 天）
    data_dir: Path = ROOT_DIR / "data"                 # 数据根目录（上传文件与构建数据）
    role_dir: Path = ROOT_DIR / "configs" / "roles"    # 角色卡 JSON 目录，启动时由 sync_roles 同步入库
    # 用 Path 而不是 str：这些值会被用于路径拼接与目录创建，
    # 声明成 Path 类型让 pydantic 直接完成转换，调用方不必到处 Path()。


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """取全局配置（进程内只构造一次）。

    返回：
        Settings 实例。

    lru_cache(maxsize=1) 的作用：
        配置从 .env 读取并解析，重复构造既浪费又可能读到不一致的快照
        （万一 .env 中途被改）。缓存成单例保证全进程用同一份配置，
        也让"同一请求内多次取配置"是零成本的。

    副作用：会在首次调用时创建 data/ 与 logs/ 目录。
        这两个目录是所有模块的共同依赖（写数据、写日志），
        在取配置时顺手建好，比让每个使用者各自记得建目录更可靠。

    测试提示：因为结果被缓存，测试中切换环境变量后需要
        get_settings.cache_clear() 才能生效。
    """
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    return settings
