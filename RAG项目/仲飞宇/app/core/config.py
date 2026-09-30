"""全局配置。

从环境变量 / 项目根目录 .env 文件读取，所有第三方服务（LLM、Embedding、
Milvus、SQL、Redis）都通过这里的配置驱动，便于 MVP(轻量本地) -> 生产(服务化)平滑切换。
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录（app/core/config.py -> 上溯三层）
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 优先级（从高到低）：真实环境变量 > .env.{APP_ENV} > .env
#
# 真实环境变量最高，是 12-factor 的通行做法，也是云上唯一不用改文件就能传密钥的方式。
# 但 `.env.{APP_ENV}` 必须用 override=True 才能压过 `.env`（分层覆盖），而这会连真实
# 环境变量一起盖掉——于是 `APP_ENV=prod LLM_API_KEY=sk-real bash scripts/run.sh`
# 会被 `.env.prod` 里的占位符静默顶掉，应用拿着字面量 `<你的…key>` 去请求，只得到 401，
# 没有任何线索指向配置覆盖顺序（实测于 2026-09-21）。
# 故：先把真实环境快照下来，两个文件都加载完再写回。
_real_environ = dict(os.environ)

# 实际加载到的配置文件，供启动日志与 /health 之外的排查使用。
# 为什么必须留痕：`APP_ENV=production`（或大小写写错）时 load_dotenv 对不存在的文件
# **静默返回 False**，应用悄悄退回 .env —— 在 DrvFs/不区分大小写的文件系统上
# `APP_ENV=PROD` 还能读到 .env.prod，Linux 上同样的命令就读不到，同一份部署文档
# 在两处行为不同且无任何日志。这类"配置静默回落"正是本项目反复踩的坑。
_env_files: list[str] = []
_env = os.environ.get("APP_ENV", "").strip()

if (PROJECT_ROOT / ".env").is_file():
    load_dotenv(PROJECT_ROOT / ".env")  # 基础配置（本地开发默认）
    _env_files.append(".env")

# 环境切换：APP_ENV=dev/test/prod 时，再加载 .env.{env} 覆盖基础配置。
# 开发/测试/生产三套环境用同一份代码、不同配置驱动（见 .env.dev/.env.test/.env.prod）。
if _env:
    _env_file = PROJECT_ROOT / f".env.{_env}"
    if _env_file.is_file():
        load_dotenv(_env_file, override=True)
        _env_files.append(f".env.{_env}")
    else:
        # 配置加载发生在 setup_logging 之前，这里只能直接写 stderr（否则这条警告会丢）
        print(
            f"[config] 警告：APP_ENV={_env!r} 但 {_env_file.name} 不存在 —— 只加载了 "
            f"{_env_files or ['(无)']}，应用将退回 .env 甚至内置默认值"
            f"（sqlite + 本地 Milvus + ollama@localhost）。请检查环境名拼写（区分大小写）。",
            file=sys.stderr,
        )

# 写回真实环境变量，兑现上面声明的优先级
os.environ.update(_real_environ)


def _get(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)


def _int(key: str, default: int) -> int:
    v = _get(key)
    if v is None or str(v).strip() == "":
        return default
    try:
        return int(v)
    except ValueError:
        # 报错必须带键名：以前抛的是 `invalid literal for int()`，看不出是哪个配置项，
        # 而它在 import 期就会让 app / ingest / pytest 全部起不来。
        raise ValueError(f"配置项 {key}={v!r} 不是整数（来自环境变量或 .env 文件）") from None


def _float(key: str, default: float) -> float:
    v = _get(key)
    if v is None or str(v).strip() == "":
        return default
    try:
        return float(v)
    except ValueError:
        raise ValueError(f"配置项 {key}={v!r} 不是数字（来自环境变量或 .env 文件）") from None


def _bool(key: str, default: bool) -> bool:
    v = _get(key)
    if v is None or str(v).strip() == "":
        return default
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "on", "y", "t"}:
        return True
    if s in {"0", "false", "no", "off", "n", "f"}:
        return False
    # 认不出的值以前静默取 default（`OCR_ENABLED=enabled`、`SUMMARY_ENABLED=ture` 都当"关"）。
    # 开关类配置写错就等于关掉功能，且没有任何痕迹——喊一声。
    print(
        f"[config] 警告：{key}={v!r} 不是布尔值，按 {default!r} 处理"
        f"（认得的写法：true/false、yes/no、on/off、1/0）",
        file=sys.stderr,
    )
    return default


@dataclass
class Settings:
    # ===== LLM =====
    llm_provider: str = "ollama"            # ollama | openai_compat | dummy
    ollama_base_url: str = "http://localhost:11434/v1"
    llm_base_url: str = ""                  # 其它 OpenAI 兼容服务商
    llm_api_key: str = "ollama"
    llm_model: str = "qwen2.5:7b"
    llm_temperature: float = 0.7
    # 单次 LLM 请求的超时（秒）。别吃 SDK 默认的 600s：Ollama 挂住不返回时，
    # 每个 /chat 会占住一个线程池 worker，堆够了就连 /health（ping 也走网络）一起排队。
    # 本机实测真回答 69s（暖）/ >2 分钟（冷启动），300s 留足余量。
    llm_timeout: float = 300.0

    # ===== Embedding =====
    embed_provider: str = "ollama"          # ollama | openai_compat | dummy
    embed_base_url: str = "http://localhost:11434/v1"
    embed_api_key: str = "ollama"
    embed_model: str = "bge-m3"
    embed_dim: int = 1024
    # 单次 /embeddings 请求带多少条文本。**不是性能参数，是可用性参数**：本机实测一次发
    # 838 条时 Ollama 会 400 掉（它内部 tokenize 的子进程连接被拒），512 条正常。
    # 256 是留了一倍余量后的取值；在线 API 一般另有更小的上限，换 provider 时调这一项。
    embed_batch_size: int = 256

    # ===== 文档解析 / OCR =====
    ocr_enabled: bool = True  # 扫描件/图片兜底（RapidOCR，惰性加载，默认开）
    min_chunk_chars: int = 10  # 入库低质量过滤：低于此字符数的 chunk 丢弃
    summary_enabled: bool = False  # 入库时用 LLM 给每个 chunk 生成摘要（费 LLM，默认关）

    # ===== Milvus =====
    milvus_uri: str = "./data/milvus.db"    # 本地 lite；服务器填 http://host:19530
    milvus_collection: str = "role_knowledge"

    # ===== 关系库 =====
    sql_url: str = "sqlite:///./data/app.db"

    # ===== 短期记忆 =====
    memory_backend: str = "memory"          # memory | redis
    redis_url: str = "redis://localhost:6379/0"
    memory_max_turns: int = 10
    # 送进提示词的历史字符预算。轮数只是上限，真正决定装不装得下的是回答长度：
    # 单条 1670 字时，4096 上下文只装得下 3 轮。不裁的话 Ollama 会静默丢最老的几轮，
    # 应用看不见也控制不了。默认值按 16384 上下文留足余量（见 docs / 记忆）。
    history_max_chars: int = 12000
    # 整个提示词（系统消息 + 历史 + 本轮提问）的字符预算。
    #
    # 为什么还需要一个整体预算：history_max_chars 只管历史，而系统消息
    # （人设 + TOP_K 条资料）完全不进那个预算。资料大小随 chunk_size 走，
    # 系统消息一变大，历史就按原额度照留，prompt 整体超窗。
    #
    # 超窗的后果实测（qwen3:8b / n_ctx=16384 / 用 Ollama 自带 prompt_eval_count 读真值）：
    #   chunk_size=2000 -> prompt 15355 tokens -> 装得下，没丢轮次，但只剩约 1000 tokens
    #                      给生成，回答长一点照样会被 finish_reason=length 砍断
    #   chunk_size=3000 -> prompt 18850 tokens -> 服务端只评估了 14760，**静默丢最老轮次**
    #   chunk_size=6000 -> prompt 29286 tokens -> 只评估 8194（约 n_ctx 一半）
    # 丢轮次时 finish_reason 仍是 stop、回答照常生成，日志里只有一条与实际不符的
    # 「历史裁剪：保留 N 轮」。所以安全线大致就是 n_ctx 本身：别让 prompt 顶到 16k。
    # 整体预算让历史在系统消息变大时自动让位，这正是唯一能兜住这件事的地方。
    #
    # 14000 字符：实测自然中文约 0.70 tokens/字（2000/3000/4000 三档 0.694~0.695，
    # 真实语料另测约 0.73），对应约 10k tokens，给生成留约 6k。取值偏保守。
    # 别拿「同一个字重复 N 遍」的填充文本去量这个比值：重复汉字 BPE 合并不了，
    # 会把 token 量虚报约 35%，曾据此算出 1.04 tokens/字这种错数。
    prompt_max_chars: int = 14000

    # ===== 检索 =====
    top_k: int = 5
    score_threshold: float = 0.3
    query_rewrite_enabled: bool = False  # 对话时 LLM 改写/扩写 query 多查询召回（默认关）
    reranker: str = "score_fusion"          # score_fusion | bge
    # 精排前的粗排召回池宽度（仅 reranker=bge 时生效）。精排只能从池子里挑，
    # 池宽等于 top_k 时它就只能换换顺序、捞不出粗排漏掉的好候选，形同虚设。
    # 所以开精排时按 4x top_k（默认 20）宽召回，再精排截断回 top_k。
    # score_fusion 时不生效，召回行为与加这个参数前完全一致。
    rerank_pool: int = 20
    rerank_base_url: str = "http://127.0.0.1:8001/v1"   # bge 时使用的 rerank 服务
    rerank_api_key: str = ""                # 在线服务（硅基流动等）才需要；本地留空
    rerank_model: str = "BAAI/bge-reranker-v2-m3"

    # ===== 日志 =====
    log_level: str = "INFO"
    log_dir: str = "./logs"
    # 本进程的监听端口，**由启动脚本注入**（scripts/run_workers.sh 里多 worker 各带一个）。
    # 只用来给日志分文件：多个进程共写一个 app.log 时，TimedRotatingFileHandler 的
    # 「改名 + 重新打开」在多进程下会互相踩（实测跨轮转时输的那个进程日志直接丢，
    # 还抛 FileNotFoundError），按端口各写各的文件就彻底没有这个共享点。
    # 留空 = 单实例，继续写 logs/app.log（deploy 文档与自检脚本引用的就是这个路径）。
    worker_port: str = ""

    # ===== 配置来源（只读，由 load_settings 填）=====
    # 启动时打出来，免得"到底读了哪几层配置"要靠猜（APP_ENV 拼错会静默回落）
    app_env: str = ""
    env_files: tuple[str, ...] = ()

    @property
    def llm_effective_base_url(self) -> str:
        """LLM 实际 base_url：openai_compat 优先用显式 base_url，否则用 Ollama。"""
        return self.llm_base_url or self.ollama_base_url

    @property
    def embed_effective_base_url(self) -> str:
        return self.embed_base_url or self.ollama_base_url


def load_settings() -> Settings:
    return Settings(
        llm_provider=_get("LLM_PROVIDER", "ollama") or "ollama",
        ollama_base_url=_get("OLLAMA_BASE_URL", "http://localhost:11434/v1") or "http://localhost:11434/v1",
        llm_base_url=_get("LLM_BASE_URL", "") or "",
        llm_api_key=_get("LLM_API_KEY", "ollama") or "ollama",
        llm_model=_get("LLM_MODEL", "qwen2.5:7b") or "qwen2.5:7b",
        llm_temperature=_float("LLM_TEMPERATURE", 0.7),
        llm_timeout=_float("LLM_TIMEOUT", 300.0),
        embed_provider=_get("EMBED_PROVIDER", "ollama") or "ollama",
        embed_base_url=_get("EMBED_BASE_URL", "http://localhost:11434/v1") or "http://localhost:11434/v1",
        embed_api_key=_get("EMBED_API_KEY", "ollama") or "ollama",
        embed_model=_get("EMBED_MODEL", "bge-m3") or "bge-m3",
        embed_dim=_int("EMBED_DIM", 1024),
        embed_batch_size=_int("EMBED_BATCH_SIZE", 256),
        ocr_enabled=_bool("OCR_ENABLED", True),
        min_chunk_chars=_int("MIN_CHUNK_CHARS", 10),
        summary_enabled=_bool("SUMMARY_ENABLED", False),
        milvus_uri=_get("MILVUS_DB_URI", "./data/milvus.db") or "./data/milvus.db",
        milvus_collection=_get("MILVUS_COLLECTION", "role_knowledge") or "role_knowledge",
        sql_url=_get("SQL_URL", "sqlite:///./data/app.db") or "sqlite:///./data/app.db",
        memory_backend=_get("MEMORY_BACKEND", "memory") or "memory",
        redis_url=_get("REDIS_URL", "redis://localhost:6379/0") or "redis://localhost:6379/0",
        memory_max_turns=_int("MEMORY_MAX_TURNS", 10),
        history_max_chars=_int("HISTORY_MAX_CHARS", 12000),
        prompt_max_chars=_int("PROMPT_MAX_CHARS", 14000),
        top_k=_int("TOP_K", 5),
        score_threshold=_float("SCORE_THRESHOLD", 0.3),
        query_rewrite_enabled=_bool("QUERY_REWRITE_ENABLED", False),
        reranker=_get("RERANKER", "score_fusion") or "score_fusion",
        rerank_pool=int(_get("RERANK_POOL", "20") or "20"),
        rerank_base_url=_get("RERANK_BASE_URL", "http://127.0.0.1:8001/v1") or "http://127.0.0.1:8001/v1",
        rerank_api_key=_get("RERANK_API_KEY", "") or "",
        rerank_model=_get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3") or "BAAI/bge-reranker-v2-m3",
        log_level=_get("LOG_LEVEL", "INFO") or "INFO",
        log_dir=_get("LOG_DIR", "./logs") or "./logs",
        worker_port=_get("WORKER_PORT", ""),
        app_env=_env,
        env_files=tuple(_env_files),
    )


# 模块级单例：进程启动时加载一次
settings = load_settings()
