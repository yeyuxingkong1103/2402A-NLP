# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
# 工单04 - 图像内容解析及检索优化
"""
图像语义转写：把图区渲染成 PNG，交给多模态模型读成文字。

【为什么是"入库期转写"而不是"查询期读图"】
验收要求"提问到回答 ≤3 秒"。3B 视觉模型读一张图要十几秒 —— 查询期读图
这个指标必挂。所以图在**入库时**就被转写成文字进了知识库，问答期走的还是
原来的纯文本检索链路（bge-m3 + qwen3），一行都没改。

【为什么用多模态大模型而不是 CLIP】工单备注允许"CLIP 或多模态大模型"。
CLIP 给的是图文相似度，不是文字 —— 要让它进知识库得给 Milvus 加一路 CLIP
向量（schema 建表即定死，必须重建）并在查询侧再挂一个 CLIP 文本编码器，
而且它**回答不了**"销售部有几个部门"这种计数题。多模态大模型把图读成文字，
直接复用现有链路，代价只在入库那一次。

【显存时序】8GB 卡实测装不下 qwen3:8b(5.58GB) + 3B VLM(约3.2GB)，所以
入库时按 `unload(聊天模型) → 逐图转写 → unload(VLM)` 排程，见 pipeline。
转写与嵌入（bge-m3）也天然错峰：转写在 embed 阶段之前全部结束。
"""

from __future__ import annotations

import asyncio
import base64
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from app.config import settings
from app.core import image_cache
from app.core.ollama_client import OllamaClient, OllamaError, get_client

#: 渲染用锁：PyMuPDF 的 Document 不是线程安全的，而 `asyncio.to_thread`
#: 会把它丢到线程池里跑。加锁 = 串行渲染，打开一次文档反复用，
#: 比"每张图各开一次文档"（14MB PDF × 49 次）快得多。
_render_lock = threading.Lock()


@dataclass
class FigureStats:
    """转写阶段统计，供入库脚本打印与验收留痕。"""

    n_figures: int = 0
    n_vlm_calls: int = 0
    n_cache_hits: int = 0
    n_reviewed: int = 0
    n_failures: int = 0
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return {
            "figures": self.n_figures, "vlm_calls": self.n_vlm_calls,
            "cache_hits": self.n_cache_hits, "reviewed": self.n_reviewed,
            "failures": self.n_failures, "seconds": round(self.seconds, 1),
        }


# ----------------------------------------------------------------------
# 提示词
# ----------------------------------------------------------------------
_PROMPT_CHART = """你是招股说明书图表转录器。下面是一张图表，图题是「{caption}」。
只依据图中可见内容，逐项输出图中每一个类别的名称与数值（带单位），一行一条，格式：类别：数值。

要求：
- 增长率/百分比/负值必须标出正负号（例如「计算机：-4.2%」）。
- **逐条核对标签与数值的对应关系**：柱状图的数值标签可能在柱子上方或下方，
  不要按顺序臆断归属；拿不准的那一条写「类别不可辨：数值」，不要猜。
- 图中若有多个子图（如饼图 + 柱状图），分别标注来源子图。
- 看不清的地方写「不可辨」，绝对不要用常识或行业经验补全。
- 只输出条目，不要写解释、不要写总结。"""

_PROMPT_DIAGRAM = """你是招股说明书图示转录器。下面是一张组织结构图/流程图，图题是「{caption}」。
请输出其层级结构：从最高层开始，逐层列出每个方框里的文字，用缩进表示从属关系。

要求：
- 只写图中出现的文字，不要补充任何常识或猜测。
- 同一层的方框按从左到右的顺序列出。
- 若图中标注了数量，照抄。
- 只输出结构，不要写解释。{labels_hint}"""


def prompt_for(kind: str, caption: str, labels: str = "") -> str:
    """按图的类型选提示词。

    【diagram 为什么要把文本层标签塞进去】组织结构图的方框文字**本来就在
    PDF 的文本层里**（页 38 实测 17 个标签一字不差），把它们一并给模型当锚，
    可以避免它把"大客户销售部"读成别的字 —— id=5 要数的正是这些。
    """
    cap = caption or "（图题未识别）"
    if kind == "diagram":
        hint = ""
        if labels:
            hint = (f"\n\n图中方框的文字识别结果参考（可能顺序有出入，请据此还原层级）：\n{labels}")
        return _PROMPT_DIAGRAM.format(caption=cap, labels_hint=hint)
    return _PROMPT_CHART.format(caption=cap)


# ----------------------------------------------------------------------
# 渲染
# ----------------------------------------------------------------------
def render_clip(doc, page_no: int, bbox, dpi: int) -> bytes:
    """把图区渲染成 PNG 字节（加锁串行，见 _render_lock 的注释）。"""
    with _render_lock:
        page = doc[page_no]
        clip = pymupdf.Rect(*bbox)
        # 不能超出页面边界，否则 PyMuPDF 会报警告并裁切
        clip = clip & page.rect
        pix = page.get_pixmap(clip=clip, dpi=dpi)
        return pix.tobytes("png")


# ----------------------------------------------------------------------
# 转写器
# ----------------------------------------------------------------------
class FigureTranscriber:
    """把 pages 里每个 Figure 的 text 字段填上（走缓存 + VLM）。"""

    def __init__(self, pdf_path: str | Path,
                 client: OllamaClient | None = None,
                 progress=None, force: bool = False) -> None:
        self.pdf_path = Path(pdf_path)
        self.client = client or get_client()
        self.progress = progress
        self.force = force          # True = 忽略缓存重跑 VLM

    async def run(self, pages) -> FigureStats:
        st = FigureStats()
        figures = [f for pc in pages for f in pc.figures]
        st.n_figures = len(figures)
        if not figures or not settings.image_enable:
            return st

        t0 = time.perf_counter()
        doc = pymupdf.open(self.pdf_path)
        try:
            for i, fig in enumerate(figures, 1):
                png = await asyncio.to_thread(
                    render_clip, doc, fig.page_no, fig.bbox, settings.image_render_dpi)
                key = image_cache.cache_key(png, settings.vlm_model)
                rec = None if self.force else image_cache.load(key)
                # 缓存里存的如果是**失败记录**（没转写出文本、也没人工覆写），
                # 那它不算命中 —— 否则一次瞬时 500 会让这张图永久缺失。
                if rec is not None and not rec.effective_text:
                    rec = None
                if rec is None:
                    rec = image_cache.make_transcript(
                        key=key, png=png, doc_name=self.pdf_path.name,
                        page_no=fig.page_no, page_label=fig.page_label,
                        kind=fig.kind, bbox=fig.bbox,
                        dpi=settings.image_render_dpi,
                        model=settings.vlm_model,
                        caption=fig.title, labels=fig.labels)
                    try:
                        rec.vlm_text = await self._transcribe(png, fig)
                        st.n_vlm_calls += 1
                    except OllamaError as e:
                        # 单张图失败**不中断整库**：题注与文本层标签仍然进库，
                        # 但失败数必须如实记下来（否则就是静默答错）
                        rec.note = f"VLM 失败：{str(e)[:160]}"
                        st.n_failures += 1
                    image_cache.save(rec)
                else:
                    st.n_cache_hits += 1
                    # 缓存里的题注/标签可能比本次解析更新（人工修正过），以缓存为准
                    if rec.caption:
                        fig.title = rec.caption
                    if rec.labels:
                        fig.labels = rec.labels
                fig.text = rec.effective_text
                fig.reviewed = bool(rec.reviewed)
                if self.progress is not None:
                    self.progress(i, len(figures), fig)
        finally:
            doc.close()
        st.n_reviewed = sum(1 for f in figures if f.reviewed)
        st.seconds = time.perf_counter() - t0
        return st

    async def _transcribe(self, png: bytes, fig) -> str:
        """调一次 VLM；对**瞬时故障**重试。

        【为什么必须重试】实测书1 的 16 张图里有 10 张返回 `500 Internal Server
        Error`，而同一张图单独重调就是 200 —— 那是模型重载/显存腾挪期的瞬时失败，
        不是这张图有问题。没有重试就会**静默丢掉整张图的语义**（只留下题注），
        而丢掉的恰好可能是某道题的答案。
        """
        prompt = prompt_for(fig.kind, fig.title, fig.labels)
        payload = [{"role": "user", "content": prompt}]
        images = [base64.b64encode(png).decode("ascii")]
        last: Exception | None = None
        for attempt in range(3):
            try:
                text = await self.client.chat(
                    payload,
                    images=images,
                    model=settings.vlm_model,
                    temperature=settings.vlm_temperature,
                    max_tokens=settings.vlm_max_tokens,
                    num_ctx=settings.vlm_num_ctx,
                    keep_alive=settings.vlm_keep_alive,
                    timeout=settings.vlm_timeout,
                    # 压重复：不压的话某些图会死循环，Ollama 直接 500
                    # （实测页 1-1-223：「prediction aborted, token repeat limit reached」）
                    extra_options={
                        "repeat_penalty": settings.vlm_repeat_penalty,
                        "repeat_last_n": settings.vlm_repeat_last_n,
                    },
                )
                if text.strip():
                    return text
                last = OllamaError("模型返回空文本")
            except OllamaError as e:
                last = e
            if attempt < 2:
                await asyncio.sleep(2.0 * (attempt + 1))
        raise last or OllamaError("转写失败")


# ----------------------------------------------------------------------
# 入库期的显存排程
# ----------------------------------------------------------------------
async def unload_chat_model(client: OllamaClient | None = None) -> bool:
    """转写前把聊天模型赶下显存（见 ollama_client.unload 的注释）。"""
    return await (client or get_client()).unload(settings.llm_model)


async def unload_vlm(client: OllamaClient | None = None) -> bool:
    """转写后把 VLM 赶下显存，让紧随其后的嵌入阶段拿回 GPU。"""
    return await (client or get_client()).unload(settings.vlm_model)
