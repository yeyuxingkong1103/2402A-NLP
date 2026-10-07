# 工单编号：人工智能NLP-RAG-功能测试及评估
"""图像内容解析：把 PDF 里的图表交给多模态模型，转成可检索的文字

工单4 的核心。招股书里的组织结构图、市场结构图等，**数据不在文本层**：
  - 组织结构图（招股书2 第 39 页）文字虽然在文本层，但被平铺成一串，
    看不出「销售部下面挂着哪几个部门、大客户销售部下面挂着哪几个销售处」；
  - 市场应用结构图（第 72 页）是位图，柱子的数值压根不在文本层。

这类问题靠文本检索永远答不出来，必须先让多模态模型"看懂"图，
把图转成文字描述再入库。工单备注即要求「使用多模态模型（CLIP 或多模态大模型）实现」。
"""
import base64
import json
import re
import time
from pathlib import Path

import pymupdf

from config import (
    IMAGE_DPI, IMAGE_MODEL, LLM_API_BASE, LLM_API_KEY, LLM_TIMEOUT, MAX_CHART_PAGES,
)
from rag_engine import LLMError

# 含大图的页基本就是图表页；另外结构图/示意图可能是矢量绘制（没有位图），
# 靠关键词补充。
_CHART_WORDS = re.compile(r"组织结构图|结构图|示意图|趋势图|如下图所示|如下图")


def chart_pages(pdf_path, min_side=250):
    """挑出含图表的页。

    三类都要收：
      1. 有较大位图的页（大多数统计图）
      2. 出现「结构图 / 如下图」等字样的页 —— **图表可能画成矢量，没有位图**
      3. 上一条所述页面的**下一页**：招股书里常见「……如下图所示：」写在页尾，
         真正的图在下一页开头（实测组织结构图就是第 38 页写引言、第 39 页才是图，
         只按 1、2 两条会整整漏掉这张图）。
    """
    doc = pymupdf.open(pdf_path)
    try:
        n = doc.page_count
        picked = set()
        for i, page in enumerate(doc, start=1):
            if any(x[2] > min_side and x[3] > min_side for x in page.get_images()):
                picked.add(i)
            if _CHART_WORDS.search(page.get_text()):
                picked.add(i)
                if i < n:
                    picked.add(i + 1)
    finally:
        doc.close()
    return sorted(picked)


def _render(pdf_path, page_no, dpi=IMAGE_DPI):
    doc = pymupdf.open(pdf_path)
    try:
        pix = doc[page_no - 1].get_pixmap(dpi=dpi)
        return pix.tobytes("png")
    finally:
        doc.close()


PROMPT = """你是文档图像解析助手。这是招股意向书中的一页，请把页面上的图表内容
转写成文字，供后续检索使用。

要求：
1. 若是组织结构图/流程图：**逐层**列出层级关系，明确写出「谁下面包含谁」，
   例如「销售部下属：电话及网络销售部、渠道销售部、大客户销售部、国际贸易部」，
   并写出每个部门下属的机构。层级关系是最重要的信息。
2. 若是柱状图/折线图/饼图：列出每个类别及其数值，并明确指出**最大值、最小值、
   增长最快的项、出现负增长的项**。
3. 图表的标题、单位、资料来源一并写出。
4. 只写图上真实可见的内容，看不清的写"不可辨认"，不要推测或补全。

直接输出转写结果，不要加多余说明。"""


def describe_page(pdf_path, page_no, retry=2):
    """把一页渲染后交给多模态模型描述。

    走 OpenAI 兼容的 /chat/completions，图像用 base64 data URL 传入。
    """
    import requests

    if not LLM_API_BASE or not LLM_API_KEY:
        raise LLMError("未配置 DEEPSEEK_BASE_URL / DEEPSEEK_API_KEY 环境变量")

    b64 = base64.b64encode(_render(pdf_path, page_no)).decode()
    payload = {
        "model": IMAGE_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
        "temperature": 0.0,
        "max_tokens": 2048,          # 推理模型，给小了正文会返回空
    }
    last = None
    for _ in range(retry + 1):
        try:
            resp = requests.post(
                f"{LLM_API_BASE.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {LLM_API_KEY}",
                         "Content-Type": "application/json"},
                json=payload, timeout=LLM_TIMEOUT * 3,
            )
            resp.raise_for_status()
            content = (resp.json()["choices"][0]["message"].get("content") or "").strip()
            if content:
                return content
            last = "模型返回空内容"
        except Exception as exc:                               # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(1)
    raise LLMError(f"第 {page_no} 页图像解析失败：{last}")


def parse_pdf_images(pdf_path, limit=MAX_CHART_PAGES, progress=None, cache_dir=None):
    """带缓存的入口：逐页调多模态模型（实测约 18 秒/页），结果落盘复用。

    缓存按**页**增量维护：选择规则变宽（比如后来补上「图注的下一页」）或某页
    因网络抖动失败时，重跑只补解析缺的那几页，已解析的不再重复调用。
    """
    target = chart_pages(pdf_path)[:limit]
    cache = None
    done = {}
    if cache_dir:
        cache = Path(cache_dir) / f"{Path(pdf_path).stem}_images.json"
        if cache.exists():
            try:
                done = {c["page"]: c for c in json.loads(cache.read_text(encoding="utf-8"))}
                print(f"  [图像] 缓存已有 {len(done)} 页")
            except (OSError, ValueError):
                print("  [图像] 缓存损坏，全部重解析")
                done = {}

    missing = [p for p in target if p not in done]
    if missing:
        print(f"  [图像] 待解析 {len(missing)} 页：{missing[:12]}", flush=True)
        for n, page_no in enumerate(missing, start=1):
            if progress:
                progress(f"[图像解析] {n}/{len(missing)} 第 {page_no} 页 ...")
            try:
                text = describe_page(pdf_path, page_no)
            except LLMError as exc:
                print(f"  [跳过] {exc}")
                continue
            # 带上页码，与大模型提示词里的「第 N 页」格式一致，便于引用出处
            done[page_no] = {"page": page_no, "type": "image",
                             "text": f"【第{page_no}页 图表】{text}"}
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps([done[p] for p in sorted(done)],
                                        ensure_ascii=False), encoding="utf-8")
    return [done[p] for p in target if p in done]
