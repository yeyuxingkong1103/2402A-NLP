# -*- coding: utf-8 -*-
"""
图像语义解析模块（多模态）
工单编号：人工智能NLP-RAG-图像内容解析及检索优化

工单备注要求：「PDF 中的图像语义解析使用多模态模型（CLIP 或多模态大模型）实现」。
本模块提供两种实现，可单独使用也可级联：

  1. CLIPImageParser  —— 基于 CLIP 的零样本图像分类 + 图文匹配
                         用途：快速判定图像类别（组织结构图/柱状图/…），
                         并把图像向量与文本描述对齐，支持以文搜图。
  2. VLMImageParser   —— 基于多模态大模型（Qwen-VL / GPT-4V 类接口）
                         用途：把图表「读」成结构化文本，这是回答
                         「组织结构图里销售部有几个下属部门」这类问题的关键。

为什么必须做图像解析：
  招股说明书里的组织结构图、市场结构图绝大多数是**矢量绘制**，
  PyMuPDF 的文本层取不到，向量检索完全无法命中；
  只有把图渲染成位图后再做多模态理解，才能把图中信息变成可检索文本。
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path

from . import config, llm
from .pdf_parse import PageBlock

# ---------------------------------------------------------------------------
# 图表类型标签集（CLIP 零样本分类用）
# ---------------------------------------------------------------------------
CHART_LABELS = {
    "组织结构图": "公司组织结构图，展示部门、子公司、销售处的层级隶属关系",
    "股权结构图": "股权结构图，展示股东持股比例与控股关系",
    "柱状图": "柱状图，用柱子的高矮对比不同类别的数值大小",
    "折线图": "折线图，展示指标随时间的变化趋势",
    "饼图": "饼图，展示各部分占总体的比例构成",
    "流程图": "流程图，展示业务流程或生产工艺的先后顺序",
    "产品图": "产品实物照片或产品示意图",
    "表格截图": "以图片形式嵌入的表格",
    "地图": "地理分布地图",
    "其他示意图": "其他类型的示意图或插图",
}

_TEMPLATE = "这是一张{}"


# ---------------------------------------------------------------------------
# 1. CLIP 零样本分类
# ---------------------------------------------------------------------------
class CLIPImageParser:
    """
    使用 CLIP 做图像分类与图文匹配。

    模型选择：优先加载本地缓存的 `OFA-Sys/chinese-clip-vit-base-patch16`
    （中文图文匹配效果显著优于原版 CLIP），缺失时回退 `openai/clip-vit-base-patch32`。
    """

    name = "clip"

    def __init__(self, model_name: str = "OFA-Sys/chinese-clip-vit-base-patch16"):
        self.model_name = model_name
        self._model = None
        self._labels: list[str] = []
        self._label_vecs = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from transformers import CLIPModel, CLIPProcessor
            self._model = (CLIPModel.from_pretrained(self.model_name),
                           CLIPProcessor.from_pretrained(self.model_name))
        except Exception as e:
            print(f"  [warn] 中文 CLIP 加载失败（{e}），回退英文 CLIP")
            from transformers import CLIPModel, CLIPProcessor
            self.model_name = "openai/clip-vit-base-patch32"
            self._model = (CLIPModel.from_pretrained(self.model_name),
                           CLIPProcessor.from_pretrained(self.model_name))

    def classify(self, image_path: str | Path, top_n: int = 3) -> list[dict]:
        """零样本分类：返回图像最可能属于哪类图表。"""
        import torch
        from PIL import Image

        self._load()
        model, processor = self._model

        labels = list(CHART_LABELS.keys())
        texts = [CHART_LABELS[l] for l in labels]
        img = Image.open(image_path).convert("RGB")

        inputs = processor(text=texts, images=img, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = model(**inputs)
        probs = out.logits_per_image.softmax(dim=1)[0]

        ranked = sorted(zip(labels, probs.tolist()), key=lambda x: -x[1])[:top_n]
        return [{"label": l, "score": round(s, 4)} for l, s in ranked]

    def embed_image(self, image_path: str | Path):
        """把图像编码为向量，用于「以文搜图」。"""
        import torch
        from PIL import Image

        self._load()
        model, processor = self._model
        img = Image.open(image_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            feat = model.get_image_features(**inputs)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat[0].cpu().numpy()


# ---------------------------------------------------------------------------
# 2. 多模态大模型解析（主力方案）
# ---------------------------------------------------------------------------
_CHART_PROMPT = """请仔细阅读这张来自招股说明书的图片，输出结构化的中文描述。

要求：
1. **图表类型**：判断这是组织结构图 / 股权结构图 / 柱状图 / 折线图 / 饼图 / 流程图 / 其他。
2. **标题**：如果图中有标题，原样抄录。
3. **数据内容**（最重要）：
   - 如果是组织结构图/股权结构图：逐条列出「父节点 → 子节点」的隶属关系，
     并注明每个节点的名称。层级关系必须准确，不要遗漏任何分支。
   - 如果是柱状图/折线图/饼图：把图中的**每一个数据标签都读出来**，
     以「类别：数值」的形式逐条列出，不要估算、不要遗漏；
     如果有多个年份/系列，按系列分组列出。
   - 如果是流程图：按先后顺序列出各个步骤。
4. **关键结论**：如果有明显的最值（最大/最小/增长最快/负增长）请指出。
5. 如果图片模糊或信息不全，如实说明「图像不清晰，无法识别 X」，不要编造。

直接输出 Markdown 格式的描述文本，不要输出 JSON，不要有任何额外说明。"""

_ORG_PROMPT = """这是一张组织结构图。请把图中的层级结构完整地转成 Markdown 文本，
格式如下：

图标题：<如果有>

层级结构：
- 顶层节点名
  - 二级节点名
    - 三级节点名
      …

要求：
1. 严格按照图中的隶属关系缩进，不要臆造层级
2. 每个节点名称逐字抄录，不要改写、不要合并
3. 如果某个节点下有多个并列子节点，全部列出，一个都不能少
4. 如果有文字说明（如部门职责），一并抄录
5. 不确定的节点标注「(不清晰)」

只输出上述 Markdown 内容。"""


class VLMImageParser:
    """
    多模态大模型图像解析器。

    后端优先级：
      1. `openai` 兼容接口的多模态模型（环境变量 RAG_VLM_MODEL 指定）
      2. 本地 Ollama 的视觉模型（如 qwen2.5-vl:7b）
    """

    name = "vlm"

    def __init__(self, backend: str = "auto", model: str | None = None):
        self.backend = backend
        self.model = model

    # -- 后端选择 -----------------------------------------------------------
    def _resolve_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        import os
        if os.environ.get("RAG_VLM_MODEL"):
            return "openai"
        try:
            import urllib.request
            with urllib.request.urlopen(
                f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3
            ) as r:
                tags = json.loads(r.read())
            names = [m["name"] for m in tags.get("models", [])]
            if any("vl" in n.lower() or "vision" in n.lower() or "llava" in n.lower()
                   for n in names):
                return "ollama"
        except Exception:
            pass
        return "openai"

    # -- 主入口 -------------------------------------------------------------
    def describe(self, image_path: str | Path, kind: str = "auto") -> str:
        """
        生成图像的语义描述文本。

        Args:
            kind: auto=通用图表解析；org=组织结构图专用解析
        """
        prompt = _ORG_PROMPT if kind == "org" else _CHART_PROMPT
        backend = self._resolve_backend()
        try:
            if backend == "ollama":
                return self._call_ollama(image_path, prompt)
            return self._call_openai(image_path, prompt)
        except Exception as e:
            print(f"  [warn] 多模态解析失败（{image_path}）：{e}")
            return ""

    def _call_openai(self, image_path: str | Path, prompt: str) -> str:
        """走 OpenAI 兼容的多模态接口。"""
        import os
        from openai import OpenAI

        model = self.model or os.environ.get("RAG_VLM_MODEL", "qwen-vl-max")
        client = OpenAI(
            api_key=os.environ.get("RAG_VLM_API_KEY", config.DEEPSEEK_API_KEY),
            base_url=os.environ.get("RAG_VLM_BASE_URL", config.DEEPSEEK_BASE_URL),
        )
        b64 = _to_base64(image_path)
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
            temperature=0.0, max_tokens=2000,
        )
        return (resp.choices[0].message.content or "").strip()

    def _call_ollama(self, image_path: str | Path, prompt: str) -> str:
        """走本地 Ollama 视觉模型。"""
        import urllib.request

        model = self.model or _pick_ollama_vlm()
        payload = json.dumps({
            "model": model, "prompt": prompt, "stream": False,
            "images": [_to_base64(image_path)],
            "options": {"temperature": 0.0},
        }).encode()
        req = urllib.request.Request(
            f"{config.OLLAMA_BASE_URL}/api/generate", data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read()).get("response", "").strip()

    # -- 图表转结构化数据 ---------------------------------------------------
    def extract_chart_data(self, image_path: str | Path) -> dict:
        """
        把图表读成结构化数据（用于精确回答数值类问题）。
        返回 {"type":..., "title":..., "series":[{"name":..., "data":{...}}]}
        """
        prompt = """读取这张图表，把其中的数据完整地转成 JSON。

输出格式：
{
  "type": "柱状图|折线图|饼图|组织结构图|其他",
  "title": "<图标题，无则空字符串>",
  "x_axis": "<横轴含义>",
  "y_axis": "<纵轴含义，含单位>",
  "series": [
    {"name": "<系列名>", "data": {"<类别或年份>": <数值>, ...}}
  ],
  "notes": "<图中的补充说明、图例、脚注>"
}

要求：
1. 数值必须是图中实际标注的数字，不要估算、不要推断
2. 读不出来的数值填 null，不要编造
3. 所有类别/年份都要列出，一个都不能漏
4. 只输出 JSON"""

        backend = self._resolve_backend()
        try:
            raw = (self._call_ollama(image_path, prompt) if backend == "ollama"
                   else self._call_openai(image_path, prompt))
            m = re.search(r"\{.*\}", raw, re.S)
            return json.loads(m.group(0)) if m else {"_raw": raw}
        except Exception as e:
            return {"_error": str(e)}


def _to_base64(path: str | Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def _pick_ollama_vlm() -> str:
    import urllib.request
    try:
        with urllib.request.urlopen(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3) as r:
            for m in json.loads(r.read()).get("models", []):
                n = m["name"]
                if "vl" in n.lower() or "vision" in n.lower() or "llava" in n.lower():
                    return n
    except Exception:
        pass
    return "qwen2.5-vl:7b"


# ---------------------------------------------------------------------------
# 3. 批量解析入口
# ---------------------------------------------------------------------------
def enrich_image_blocks(
    blocks: list[PageBlock],
    kind_hint: dict[int, str] | None = None,
    use_vlm: bool = True,
    use_clip: bool = True,
    skip_low_value: bool = True,
) -> list[PageBlock]:
    """
    给图像类 PageBlock 回填语义描述（content 字段）。

    流程：CLIP 分类 → 判定是否值得深解析 → VLM 生成描述 → 写入 content。

    Args:
        kind_hint: {页码: "org"} 指定某些页按组织结构图模板解析
        skip_low_value: 跳过产品照片等无检索价值的图，节省算力
    """
    kind_hint = kind_hint or {}
    vlm = VLMImageParser() if use_vlm else None
    clip = CLIPImageParser() if use_clip else None

    for b in blocks:
        if b.type != "image":
            continue
        img_path = b.extra.get("image_path")
        if not img_path or not Path(img_path).exists():
            continue

        # --- CLIP 分类 ---
        if clip:
            try:
                preds = clip.classify(img_path, top_n=2)
                b.extra["clip_labels"] = preds
                top = preds[0]["label"] if preds else ""
                if skip_low_value and top in ("产品图", "地图") and preds[0]["score"] > 0.5:
                    b.content = f"[图像] 第{b.page}页包含一张{top}，无检索价值，已跳过详解析。"
                    continue
            except Exception as e:
                print(f"  [warn] CLIP 分类失败：{e}")
                top = ""
        else:
            top = ""

        # --- VLM 描述 ---
        if vlm:
            kind = kind_hint.get(b.page, "org" if top == "组织结构图" else "auto")
            desc = vlm.describe(img_path, kind=kind)
            if desc:
                header = f"[图表] 《{b.doc}》第{b.page}页"
                if top:
                    header += f"｜类型：{top}"
                b.content = f"{header}\n{desc}"
                b.extra["vlm_parsed"] = True
                continue

        # --- 兜底：仅有 CLIP 标签 ---
        if top:
            b.content = (f"[图表] 《{b.doc}》第{b.page}页包含一张{top}。"
                         f"（未启用多模态解析，图中细节不可检索）")

    return blocks


def enrich_with_clip_only(blocks: list[PageBlock]) -> list[PageBlock]:
    """仅用 CLIP 做粗粒度标注（无多模态模型时的降级方案）。"""
    return enrich_image_blocks(blocks, use_vlm=False, use_clip=True)
