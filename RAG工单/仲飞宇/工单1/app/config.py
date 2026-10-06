# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 工单01 - 基于PDF文档的问答系统
"""
全局配置。

【为什么 Ollama 地址要动态发现】
Ollama 跑在 Windows 宿主上，WSL2 默认是 NAT 网络模式，WSL 通过「默认网关」
访问宿主。这个地址（实测 172.27.224.1）**每次 WSL 重启都会变**，写死必然失效。
所以这里自动从 `ip route` 取网关，只在取不到时才回退到 .env 里的值。
"""

from __future__ import annotations

import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（本文件在 app/ 下）
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def detect_host_ip() -> str | None:
    """取 WSL 默认网关 == Windows 宿主 IP。取不到返回 None。"""
    try:
        out = subprocess.run(
            ["ip", "route", "show", "default"],
            capture_output=True, text=True, timeout=3, check=False,
        ).stdout
        m = re.search(r"default via (\d+\.\d+\.\d+\.\d+)", out)
        return m.group(1) if m else None
    except Exception:
        return None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------- Ollama ----------
    # 留空则启动时自动探测宿主 IP
    ollama_host: str = ""
    ollama_port: int = 11434

    llm_model: str = "qwen3:8b"
    embed_model: str = "bge-m3"
    embed_dim: int = 1024

    # 关键：关闭思考链 —— 开着的 qwen3 要十几秒，3 秒指标必挂
    llm_think: bool = False
    # 关键：模型常驻，否则每次冷启动 7.2s
    llm_keep_alive: str = "30m"
    # 上下文窗口固定，别放大（KV cache 吃显存）
    llm_num_ctx: int = 4096
    # 限制生成长度，压 TTFT
    llm_max_tokens: int = 200
    llm_temperature: float = 0.1

    # ---------- Milvus ----------
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "rag_chunks"

    # ---------- 显存预算 ----------
    # 查询时把 bge-m3 放到 CPU 上跑。
    # 【为什么】实测 qwen3:8b 在 num_ctx=4096 下占 6.29GB 显存，而系统空闲
    # 只有 6.23GB。bge-m3 一旦上 GPU 就会把 qwen3 挤出去，下次问答要付
    # 6 秒冷启动 —— 实测 TTFT 从 35ms 劣化到 6117ms。查询只嵌入一句短文本，
    # CPU 上开销可忽略，换来 qwen3 常驻。入库是离线批量，仍走 GPU（快得多）。
    embed_query_on_cpu: bool = True

    # ---------- 检索 ----------
    retrieve_top_k: int = 3
    # 每个片段送入 LLM 的最大字数。
    # 【为什么是 600 而不是 300】原方案为了压 prefill 定的 300，实测会**切掉答案**：
    # id=793 的召回片段长 356 字，含答案的「下游行业为各类终端用户……主要包括
    # 军队、政府机关、能源」在 336 字处，300 字的窗口装不下，模型只看到前文图注
    # 里的「军队企事业单位」，答案就缺了两项。
    # 我们的 chunk 本身就是 300–600 字（中位 521），取 600 相当于基本不截断。
    # 放宽后实测 TTFT P95 仍 <2s，远低于 3000ms 验收线，代价可以接受。
    context_chunk_chars: int = 600

    # ---------- 分块 ----------
    chunk_min_chars: int = 300
    chunk_max_chars: int = 600
    chunk_overlap_sentences: int = 1

    # ---------- Query 理解 ----------
    # 「实体专名」清单：查询时会被抽象掉。
    # 【为什么】工单01 的 10 道题全部写成「武汉兴图新科电子股份有限公司…」，
    # 而招股书正文从不自称全称，一律用「公司/发行人/兴图新科」。
    # 这 16 个字会主导查询向量，实测去掉后 recall@5 从 7/10 升到 9/10，
    # 配合"原问句 + 抽象版"双路召回可达 10/10。
    query_entity_names: list[str] = [
        "武汉兴图新科电子股份有限公司",
        "武汉兴图新科电子",
        "兴图新科",
    ]
    # 双路召回时每路的候选池大小
    retrieve_pool_size: int = 30

    # ---------- PDF 页码映射 ----------
    # 页码偏移 = position_index − 页脚印刷页码。
    # 【实测】招股说明书1.pdf 每页页脚 y≈779/842 处印着 `1-1-N`，且
    #         index N → 页脚 1-1-N，即**偏移为 0**（此前侦察得到的 −1 是错的）。
    # 该值决定引用页码的正误，属「静默答错」类风险，由 preflight.py 每次校验。
    page_label_offset: int = 0

    # ---------- 去重 ----------
    # SimHash 海明距离阈值，<= 该值视为重复
    simhash_max_distance: int = 3

    # ---------- 数据目录 ----------
    # 默认放 WSL 原生盘（/mnt/c 是 9p 协议，慢 1-2 个数量级）
    data_dir: str = str(Path.home() / "rag-data")

    # ---------- 其他 ----------
    # 允许同时进行的生成请求数。
    # 【为什么是 1】8GB 显存装不下两份 KV cache，并发只能排队（见技术文档 §5）。
    max_concurrency: int = 1

    # ------------------------------------------------------------------
    @property
    def ollama_base_url(self) -> str:
        host = self.ollama_host.strip() or detect_host_ip() or "127.0.0.1"
        return f"http://{host}:{self.ollama_port}"

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    def ensure_dirs(self) -> None:
        for sub in ("raw", "parsed", "eval", "feedback"):
            (self.data_path / sub).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s


settings = get_settings()

# 工单编号常量 —— 每个源文件都要带上（工单硬约束）
WORK_ORDER_ID = "人工智能NLP-RAG-基于PDF文档的问答系统"
