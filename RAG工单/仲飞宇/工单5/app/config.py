# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
# 工单编号：人工智能NLP-RAG-Query 理解优化任务
# 工单01 - 基于PDF文档的问答系统
# 工单02 - 基于PDF文档的问答系统的优化
# 工单03 - PDF文档的表格解析及检索优化
# 工单04 - 图像内容解析及检索优化
# 工单05 - Query 理解优化（多轮对话 + 指代消解/省略补全）
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
    # 嵌入模型也要钉住。默认（不传 keep_alive）是 5 分钟就卸载，
    # 实测重载一次要 2355ms（热态 35ms），隔几分钟提问就会撞上，直接吃 TTFT。
    embed_keep_alive: str = "30m"
    # 上下文窗口固定，别放大（KV cache 吃显存）
    llm_num_ctx: int = 4096
    # 限制生成长度，压 TTFT
    llm_max_tokens: int = 200
    llm_temperature: float = 0.1
    # 请求超时（原先硬编码在 OllamaClient.__init__ 里的 120.0）
    ollama_timeout: float = 120.0

    # ---------- 工单04：图像语义解析 ----------
    # 是否在入库时对图表/结构图做多模态转写。关掉 = 完整降级回工单03 的行为
    # （14/14），便于 A/B 对照与出问题时的回滚。
    image_enable: bool = True
    # 【为什么是 3B 而不是 7B】实测 8GB 卡上 qwen3:8b 常驻 5.58GB，7B 级 VLM
    # （约 6GB）根本挤不进来；3B（约 3.2GB）在入库期与 qwen3 互斥、与 bge-m3
    # 天然错峰，够用。实测它读页 71 的柱状图**数值全对**，只是把最上面那条
    # （IC卡 -2.0%）的归属记错了 —— 这类错误由 reviewed_text 人工覆写兜底，
    # 见 app/core/image_cache.py。
    vlm_model: str = "qwen2.5vl:3b"
    # 转写批次内常驻，批次结束显式 unload（见 ollama_client.unload）
    vlm_keep_alive: str = "5m"
    vlm_num_ctx: int = 4096
    # 不能沿用 llm_max_tokens=200：一张图表有七八个类别，200 token 会被截断
    vlm_max_tokens: int = 800
    # 转写要的是"照抄"不是"发挥"，温度必须为 0
    vlm_temperature: float = 0.0
    vlm_timeout: float = 300.0
    # 【为什么必须调它】实测书1 页 1-1-223（openVone 视音频中间件示意图）会让
    # qwen2.5vl 陷入重复生成，Ollama 直接中断并返回 **HTTP 500**：
    #     {"error":"prediction aborted, token repeat limit reached"}
    # 温度 0（贪心）恰恰最容易死循环，所以提高重复惩罚来破环。
    # 调高它是安全的：转写要的是"照抄"，压重复不会改变正确内容。
    vlm_repeat_penalty: float = 1.3
    # 参与重复判定的窗口（默认 64 偏短，图表条目多、模式相似，放宽一点）
    vlm_repeat_last_n: int = 256

    # 图区渲染 dpi：150 实测把页 71 的图区渲成 874x395 px，VLM 读得清
    image_render_dpi: int = 150
    # ---- 位图噪声过滤（实测书2 的 get_image_info 有 5920 个实例，98.2% 是噪声）----
    # R1 最短边 < 该值 → 丢（1x394 / 1x914 那类 1 像素细条）
    image_min_bitmap_px: int = 32
    # R2 同一 xref 在本页出现 ≥ 该次数 → 丢（页眉水印 143x127 每页重复 15 次）
    image_max_repeat_per_page: int = 3
    # R3 同一 xref 在 ≥ 该页数出现 → 丢
    image_max_repeat_pages: int = 3
    # R4 版面面积占比 < 该值 → 丢（小 logo）
    image_min_area_ratio: float = 0.024
    # R5 版面面积占比 > 该值 → 丢。**整页都是图的是"扫描页"不是"插图"**：
    # 实测书1 的封面（102%）与 13 页签字页（75–87%）、书2 的 7 页签字页
    # （91–92%）都会被 R5 挡掉，而真正的示意图占比只有 3–43%。
    # 这些扫描页的文字层本来就有（签字页的正文是"签名/日期"），无须 VLM 转写。
    image_max_area_ratio: float = 0.60
    # 题注锚定：图区上方多少 pt 内找题注
    image_caption_gap: float = 40.0
    # 每页最多几个图区（页 115/117 有几十个截图拼版）
    image_max_figures_per_page: int = 4
    # 是否解析矢量图（组织结构图）。判据见 figure_locator：本页无 ≥8 行的表
    # 且块级竖排标签 ≥5 个 —— 实测 898 页零误判。
    enable_vector_figures: bool = True

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

    # ---------- 检索剖面（工单02） ----------
    # 用哪套检索策略。可选值见 app/core/profiles.py 的 PROFILES：
    #   baseline（朴素地板）/ delivered（工单01 交付版）/ optimized（工单02 优化版）
    # 这里只是**默认值**；剖面本身是不可变对象，随调用链显式传递，
    # 不在请求里改写全局 settings（见 profiles.py 顶部注释的竞态说明）。
    retrieval_profile: str = "delivered"

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

    # ---------- 工单05：Query 理解（多轮对话） ----------
    # 会话记忆：**进程内**、带 TTL 与轮数上限。刻意不引入 Redis —— 本工单的主题是
    # 「指代消解 / 省略补全」，与会话存储介质无关；进程重启即失效是已知且可接受的限度
    # （写进文档，不藏）。
    session_ttl_seconds: int = 1800          # 30 分钟不活动即过期
    session_max_turns: int = 8               # 单会话保留的轮数（有界 deque）
    session_history_turns: int = 3           # 送进 prompt 的最近轮数
    session_max_sessions: int = 64           # 会话数上限（超了 LRU 逐出）
    # 【为什么要有这道硬闸门】history 与本次 context **共享** num_ctx 预算。
    # 实测 16 题 context 最大 2470 字（id=2 表格题）；num_ctx=4096、max_tokens=200
    # → 提示词可用约 3896 token。取 2600 给 history 留闸门，超了就从最旧开始丢并如实
    # 标注 history_dropped。**不能只靠 maxlen**：不设闸门时 Ollama 会**静默截断**
    # （切掉 system 或 context，接口照常 200 而答案凭空消失）。
    session_prompt_budget_chars: int = 2600
    # 默认只把「改写后的问题」放进 history、**不放答案**：实测 5 轮剧本没有任何一轮需要
    # 读到上一轮的答案文本（Q2/Q3 靠代词消解、Q4 靠问点继承），放进去只是白占 prefill
    # （每 100 字 ≈25ms）。需要时再打开。
    session_history_with_answers: bool = False

    # 改写：**规则优先**，规则没解出「疑似指代/省略」时才调模型兜底。
    query_rewrite_llm_fallback: bool = True
    query_rewrite_timeout: float = 20.0
    query_rewrite_max_tokens: int = 200
    query_intent_lookback: int = 2           # 问点模板向前回溯的轮数

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
# 各工单的编号**互不相同**（各自 PDF 的「备注」栏写死的），被改到的文件都要写上：
# 前序工单的产物仍在服役，后序工单在其上叠加能力。
WORK_ORDER_ID = "人工智能NLP-RAG-基于PDF文档的问答系统"
WORK_ORDER_ID_02 = "人工智能NLP-RAG-基于PDF文档的问答系统优化"
WORK_ORDER_ID_03 = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"
WORK_ORDER_ID_04 = "人工智能NLP-RAG-图像内容解析及检索优化"
WORK_ORDER_ID_05 = "人工智能NLP-RAG-Query 理解优化任务"
