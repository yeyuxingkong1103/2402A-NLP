# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""★多模态语义解析：VLM 结构化描述 + CLIP 图像向量。

双通道分工（用户已确认方案）：
  - CLIP          → 负责「找到图」（跨模态检索向量）
  - 多模态大模型   → 负责「读懂图」（生成可供 LLM 作答的结构化描述）

实测硬约束：
  1. max_tokens 必须 ≥ 2000，否则配额被挤占、答案截断为空
  2. 必须显式 thinking=disabled：本机模型默认思维链达 4200~8600 tokens，
     max_tokens=8000 仍会在答案中途截断（详见 vlm_describe 注释）
  3. 提示词必须强制「完整枚举 + 保留层级」，否则 6 个销售处会漏答
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path

from rag04.config import Settings
from rag04.schema import FigureBlock

logger = logging.getLogger("rag04.vlparser")


def _preload_fragile_deps() -> None:
    """Windows 本机环境修复（必修，否则 CLIP 编码段错误）。

    实测：本机 sklearn 1.9 → pandas 3.0 → pyarrow 24 这条链的**首次导入**若发生在
    进程后段（torch/PIL 已加载后，例如 get_clip 内惰性 import），pyarrow 初始化会
    破坏 Windows 堆（0xC0000374），进程随即 access violation 段错误 —— 无预热时
    完整 CLIP 流程 20+ 次全部崩溃，预热后 9/9 成功。把该链提前到本模块导入期
    （此时 torch/PIL 尚未加载）即可稳定规避。非 Windows 平台无此问题，不预热。
    """
    try:
        import sklearn  # noqa: F401
    except Exception:
        pass


if sys.platform == "win32":
    _preload_fragile_deps()

_FIGURE_PROMPT = """请解析这张来自上市公司招股说明书的图表，用于构建可检索的知识库。

图表题注：{caption}

请严格按以下要求输出：
1. 先用一句话概括这张图在表达什么。
2. 然后**逐条完整列出**图中出现的**每一个**文字标签、方框、部门、数值、类别。
   - 必须保留**层级关系**：明确写出「A 下属 B、C、D」这样的上下级结构。
   - 若图中有多层结构，**每一层的每一个节点都必须列出，不得省略、不得合并**。
   - 若是统计图，逐一列出每个类别的名称与数值（含百分比）。
3. 使用中文回答。

重要约束：
- 只描述图中**实际可见**的内容。不要推测、不要补充常识、不要编造。
- 如果某部分字迹模糊无法识别，明确写「无法识别」，不要猜测。
- 宁多勿少：宁可列出多余的项目，也不要遗漏任何一个节点。
"""


def build_figure_prompt(caption: str = "") -> str:
    """构造图区解析提示词。强制完整枚举 + 层级保留 + 禁止推测。"""
    return _FIGURE_PROMPT.format(caption=caption or "（无题注）")


def _cache_key(image_path: str, caption: str) -> str:
    h = hashlib.md5(Path(image_path).read_bytes()).hexdigest()[:16]
    return f"{h}_{hashlib.md5(caption.encode('utf-8')).hexdigest()[:8]}"


def _read_cache(s: Settings, key: str) -> str | None:
    p = Path(s.vlm_cache_dir) / f"{key}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))["description"]
        except Exception:
            return None
    return None


def _write_cache(s: Settings, key: str, desc: str) -> None:
    Path(s.vlm_cache_dir).mkdir(parents=True, exist_ok=True)
    p = Path(s.vlm_cache_dir) / f"{key}.json"
    p.write_text(json.dumps({"description": desc}, ensure_ascii=False), encoding="utf-8")


def _default_client(s: Settings):
    """构造 OpenAI 兼容客户端。key 依次尝试 API / QWEN_API / OPENAI_API_KEY。"""
    import os
    from openai import OpenAI

    key = os.environ.get("API") or os.environ.get("QWEN_API") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("未找到视觉模型 API key（API / QWEN_API / OPENAI_API_KEY 均未设置）")
    base = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")
    return OpenAI(api_key=key, base_url=base)


def ocr_fallback(image_path: str) -> str:
    """OCR 兜底：VLM 完全不可用时至少榨出图中文字。"""
    try:
        from rapidocr_onnxruntime import RapidOCR
        engine = RapidOCR()
        result, _ = engine(image_path)
        if not result:
            return ""
        return "\n".join(line[1] for line in result if len(line) > 1)
    except Exception as e:
        logger.warning("OCR 兜底亦失败：%s", e)
        return ""


def _ocr_with_notice(image_path: str,
                     on_fallback: Callable[[str], None] | None) -> str:
    """OCR 兜底并向外通报「该结果来自 OCR」，避免与真实 VLM 描述混淆。"""
    text = ocr_fallback(image_path)
    if on_fallback is not None:
        try:
            on_fallback(text)
        except Exception as cb_err:  # pragma: no cover - 回调异常不得破坏降级链
            logger.warning("OCR 兜底回调执行失败：%s", cb_err)
    return text


def _is_unsupported_param_error(e: Exception) -> bool:
    """判定异常是否属于「服务端不认识 thinking 参数」这一类。

    不能只测错误文本里有无 thinking：多数 OpenAI 兼容服务端拒绝未知参数的
    报文是 `Unrecognized request argument` / `extra fields not permitted` /
    直接点名 extra_body，都不会出现 thinking 字样。故以错误类别为主触发：
    请求带了 thinking 时，任何 400/422（含 openai.BadRequestError）都视为
    「本后端不支持该参数」；文本含 thinking 仍作为补充触发。
    """
    if "thinking" in str(e).lower():
        return True
    try:
        from openai import BadRequestError
        if isinstance(e, BadRequestError):
            return True
    except Exception:
        pass
    status = getattr(e, "status_code", None)
    if status is None:
        status = getattr(getattr(e, "response", None), "status_code", None)
    return isinstance(status, int) and status in (400, 422)


def vlm_describe(image_path: str, caption: str, settings: Settings,
                 client=None, retries: int = 3,
                 on_fallback: Callable[[str], None] | None = None) -> str:
    """调用多模态大模型生成结构化描述。

    失败链：重试 retries 次（指数退避）→ OCR 兜底。
    `on_fallback(ocr_text)`：结果来自 OCR 兜底时回调（VLM 成功不回调），
    供调用方区分「真实 VLM 描述」与「OCR 文本」；返回值类型不变。
    """
    key = _cache_key(image_path, caption)
    cached = _read_cache(settings, key)
    if cached:
        return cached

    try:
        cli = client or _default_client(settings)
    except Exception as e:
        logger.warning("视觉客户端初始化失败，转 OCR 兜底：%s", e)
        return _ocr_with_notice(image_path, on_fallback)

    b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    prompt = build_figure_prompt(caption)

    # 实测（本机 deepseek-v4-flash-vision-exp）：该模型默认先输出超长思维链，
    # 组织图单张图 reasoning 达 4200~8600 tokens 才开始产出答案 —— 即使
    # max_tokens=8000 也会在答案进行到一半时 finish_reason=length 截断。
    # 关闭 thinking 后同一张图 438 tokens 内 finish_reason=stop、答案完整。
    # 故显式请求关闭思维链；若服务端不认识该参数（换 endpoint/模型的场景），
    # 剥掉后立即重试，不影响降级链。
    thinking_disabled = True
    last_err: Exception | None = None
    attempt = 0
    while attempt < retries:
        attempt += 1
        try:
            kwargs: dict = dict(
                model=settings.vlm_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }],
                # 硬约束：低于 2000 时思维链会挤满配额导致答案为空
                max_tokens=max(settings.vlm_max_tokens, 2000),
            )
            if thinking_disabled:
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            resp = cli.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            finish = getattr(choice, "finish_reason", "stop")
            content = (choice.message.content or "").strip()

            if finish == "length":
                logger.warning("第 %d 次解析被截断(finish_reason=length)，重试", attempt)
                last_err = RuntimeError("输出被截断")
                time.sleep(2 ** attempt)
                continue
            if not content:
                logger.warning("第 %d 次解析返回空内容，重试", attempt)
                last_err = RuntimeError("空内容")
                time.sleep(2 ** attempt)
                continue

            _write_cache(settings, key, content)
            return content

        except Exception as e:
            last_err = e
            if thinking_disabled and _is_unsupported_param_error(e):
                # 服务端/模型不支持 thinking 开关：剥掉该参数立即重试。
                # 该重试不占 retries 配额（attempt 回退），否则 retries=1 或
                # 拒绝发生在最后一次时，不带 thinking 的请求永远不会发出。
                thinking_disabled = False
                attempt -= 1
                logger.warning("服务端可能不识别 thinking 参数（%s），剥离后重试",
                               type(e).__name__)
                continue
            logger.warning("第 %d/%d 次视觉解析失败：%s: %s", attempt, retries, type(e).__name__, e)
            if attempt < retries:
                time.sleep(2 ** attempt)

    logger.error("视觉解析彻底失败，转 OCR 兜底：%s", last_err)
    return _ocr_with_notice(image_path, on_fallback)


# ---------------- CLIP ----------------

_CLIP_CACHE: dict = {}


def get_clip(settings: Settings):
    """惰性加载并缓存 CLIP。返回 (model, processor, device)。"""
    if "clip" in _CLIP_CACHE:
        return _CLIP_CACHE["clip"]

    # 注：Windows 下 sklearn/pandas/pyarrow 依赖链已在模块导入期预热
    # （见 _preload_fragile_deps），此处直接导入 torch/transformers 即可。
    import torch
    from transformers import CLIPModel, CLIPProcessor

    path = Path(settings.models_dir) / "clip-vit-base-patch32"
    if not path.exists():
        raise FileNotFoundError(
            f"CLIP 未就位：{path}。请先运行 python scripts/download_models.py"
        )
    model = CLIPModel.from_pretrained(str(path), local_files_only=True)
    proc = CLIPProcessor.from_pretrained(str(path), local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    _CLIP_CACHE["clip"] = (model, proc, device)
    return _CLIP_CACHE["clip"]


def _to_list(t) -> list[float]:
    """特征 → 512 维 L2 归一化 list。

    兼容 transformers 4.x（get_*_features 直接返回 Tensor）与 5.x（返回
    BaseModelOutputWithPooling，投影后的向量在 pooler_output；实测 5.15）。
    """
    import torch

    if not torch.is_tensor(t):
        for attr in ("pooler_output", "image_embeds", "text_embeds"):
            v = getattr(t, attr, None)
            if torch.is_tensor(v):
                t = v
                break
        else:  # pragma: no cover - 版本不兼容时给出可诊断的错误而非静默错值
            raise TypeError(f"无法从 {type(t).__name__} 取出特征张量")
    return torch.nn.functional.normalize(t, dim=-1).squeeze(0).cpu().tolist()


def clip_encode_image(image_path: str, settings: Settings) -> list[float]:
    """图像 → 512 维归一化向量。"""
    import torch
    from PIL import Image

    model, proc, device = get_clip(settings)
    img = Image.open(image_path).convert("RGB")
    inputs = proc(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
    return _to_list(feats)


def clip_encode_text(text: str, settings: Settings) -> list[float]:
    """文本 → 512 维归一化向量（用于「以文搜图」）。"""
    import torch

    model, proc, device = get_clip(settings)
    inputs = proc(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
    return _to_list(feats)


def parse_figures(figs: list[FigureBlock], settings: Settings,
                  logger=None) -> list[FigureBlock]:
    """为每个图区回填 description 与 clip_vector。单项失败不影响其余。"""
    log = logger or logging.getLogger("rag04.vlparser")
    for i, f in enumerate(figs, 1):
        used_ocr = False

        def _mark_ocr_fallback(_ocr_text: str = "") -> None:
            nonlocal used_ocr
            used_ocr = True

        try:
            f.description = vlm_describe(f.image_path, f.caption, settings,
                                         on_fallback=_mark_ocr_fallback)
            if used_ocr:
                # VLM 通道失败、结果是 OCR 文本：必须留痕，否则下游无法把
                # 结构化描述与 OCR 行噪声区分开（等同于静默降级）。
                f.parse_warning = ("VLM失败，已OCR兜底" if f.description
                                   else "VLM失败，OCR兜底亦无输出")
                log.warning("图 %s VLM 失败，已OCR兜底", f.figure_id)
            elif not f.description:
                # vlm_describe 内部走完 VLM→OCR 降级仍无内容：同样必须留痕
                f.parse_warning = "VLM与OCR均无输出，描述为空"
                log.warning("图 %s VLM 与 OCR 均无输出", f.figure_id)
        except Exception as e:
            f.description = ocr_fallback(f.image_path)
            f.parse_warning = f"VLM失败({type(e).__name__})，已OCR兜底"
            log.warning("图 %s 描述失败，已兜底：%s", f.figure_id, e)

        if settings.use_clip_retrieval:
            try:
                f.clip_vector = clip_encode_image(f.image_path, settings)
            except Exception as e:
                f.clip_vector = []
                f.parse_warning = (f.parse_warning + "; " if f.parse_warning else "") + \
                    f"CLIP失败({type(e).__name__})"
                log.warning("图 %s CLIP 编码失败：%s", f.figure_id, e)

        log.info("图解析进度 %d/%d：%s", i, len(figs), f.figure_id)
    return figs
