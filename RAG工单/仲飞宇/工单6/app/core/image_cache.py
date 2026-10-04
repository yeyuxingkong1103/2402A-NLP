# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
# 工单04 - 图像内容解析及检索优化
"""
图像转写缓存：按**内容寻址**存 VLM 的转写结果，并且支持人工覆写。

【为什么必须缓存】两件事：

1. **VLM 会读错，而错的那一格可能正好是答案。** 实测页 71 的柱状图，
   qwen2.5vl:3b 把七个类别的数值全读对了，却把最上面那条 `-2.0%` 的归属
   记成了"计算机"（实际是 IC 卡）。而 id=6 问的恰恰是"负增长的是哪个行业"。
   → 没有人工修正的口子，这道题就永远是错的。
2. **重灌整库不该重跑 VLM。** 一张图一次调用要十几秒，49 个图区就是十几分钟；
   而"改提示词 / 换模型 / 调 dpi"在开发期很常见。

【按内容寻址的好处】key = sha256(**渲染出的 PNG 字节** + 模型名 + 提示词版本)。
于是：
  · 页 71 与页 309 共用同一张位图（同 xref、字节完全相同）→ **同一个 key**
    → 只调一次 VLM。实测书2 里这类重复不少。
  · 改 dpi / 改 clip / 换模型 / 改提示词，任何一项变了 key 就变，自动重跑；
    没变的继续命中缓存。
  · 缓存里存一份 PNG，是为了能**亲眼核对** —— 本机没装 poppler，
    这是唯一能把 PDF 里的图拿出来看的途径（见 scripts/transcribe_images.py --dump-png）。

【人工覆写】**不要直接改 vlm_text**，改 `reviewed_text`：
`effective_text()` 取 `reviewed_text or vlm_text`，并把 `reviewed` 标成 True。
这样"模型原本读成什么"这件事被保留下来 —— 验收时要能说清哪些是模型读的、
哪些是人工校正的，否则等于把人工结果冒充成模型能力。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from app.config import settings

#: 提示词版本 —— 改提示词就 +1，让旧缓存自动失效（PNG 仍复用，只重跑 VLM）
PROMPT_VERSION = "v1"


def cache_dir() -> Path:
    d = settings.data_path / "image_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_key(png: bytes, model: str, prompt_version: str = PROMPT_VERSION) -> str:
    """内容寻址的缓存键。"""
    h = hashlib.sha256()
    h.update(png)
    h.update(b"\x00")
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(prompt_version.encode("utf-8"))
    return h.hexdigest()[:16]


@dataclass
class Transcript:
    """一张图的转写记录（落成 {key}.json，人类可读）。"""

    key: str
    doc_name: str
    page_no: int
    page_label: str
    kind: str                     # chart | diagram
    bbox: list[float]
    dpi: int
    model: str
    prompt_version: str
    caption: str = ""             # 题注
    labels: str = ""              # 图内文本层文字（仅 diagram）
    vlm_text: str = ""            # 多模态模型原始输出
    seconds: float = 0.0
    created_at: str = ""
    # ---------- 人工核对 ----------
    reviewed: bool = False
    reviewed_text: str | None = None   # 人工修正后的转写（优先于 vlm_text）
    note: str = ""

    @property
    def effective_text(self) -> str:
        """最终用于入库的文本：人工覆写优先。"""
        return (self.reviewed_text or "").strip() or self.vlm_text


def transcript_path(key: str) -> Path:
    return cache_dir() / f"{key}.json"


def png_path(key: str) -> Path:
    return cache_dir() / f"{key}.png"


def load(key: str) -> Transcript | None:
    p = transcript_path(key)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    known = {f for f in Transcript.__dataclass_fields__}
    return Transcript(**{k: v for k, v in data.items() if k in known})


def save(rec: Transcript) -> Path:
    p = transcript_path(rec.key)
    p.write_text(json.dumps(asdict(rec), ensure_ascii=False, indent=2),
                 encoding="utf-8")
    return p


def save_png(key: str, png: bytes) -> Path:
    p = png_path(key)
    p.write_bytes(png)
    return p


def make_transcript(*, key: str, png: bytes, doc_name: str, page_no: int,
                    page_label: str, kind: str, bbox, dpi: int, model: str,
                    caption: str = "", labels: str = "") -> Transcript:
    """新建一条记录（同时把 PNG 落盘，供人工核对）。"""
    save_png(key, png)
    return Transcript(
        key=key, doc_name=doc_name, page_no=page_no, page_label=page_label,
        kind=kind, bbox=[float(v) for v in bbox], dpi=dpi, model=model,
        prompt_version=PROMPT_VERSION, caption=caption, labels=labels,
        created_at=datetime.now().isoformat(timespec="seconds"),
    )


def all_transcripts() -> list[Transcript]:
    out = []
    for p in sorted(cache_dir().glob("*.json")):
        rec = load(p.stem)
        if rec is not None:
            out.append(rec)
    return out
