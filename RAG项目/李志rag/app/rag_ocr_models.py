"""离线 RAG 使用的 PP-OCRv6 与 PaddleOCR-VL 模型封装。"""

import logging
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from threading import Lock

from app.config import get_settings
from app.rag_text_processing import clean_pdf_text

settings = get_settings()
logger = logging.getLogger(__name__)
_vl_lock = Lock()
VL_PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "chart": "Chart Recognition:",
}


@lru_cache(maxsize=1)
def ocr_engine():
    """创建并缓存基础 PaddleOCR 引擎。

    返回对象内部会加载 PP-OCRv6_medium_det（检测文字位置）和
    PP-OCRv6_medium_rec（识别文字内容）。本函数第一次调用较慢，之后复用。
    """
    # 告诉 PaddleX 去 D 盘模型目录查找权重，避免重新下载到 C 盘。
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(settings.paddlex_cache_dir)
    # 延迟导入：没有 OCR 任务时，不加载 PaddleOCR 及其依赖。
    from paddleocr import PaddleOCR

    # 构造中文 OCR 流水线。
    return PaddleOCR(
        # 中文模型同时也能识别常见数字和英文字母。
        lang="ch",
        # 当前 PDF 页面方向通常正常，关闭方向分类以减少 CPU 耗时。
        use_doc_orientation_classify=False,
        # 不执行透视拉伸和文档展平，避免额外模型开销。
        use_doc_unwarping=False,
        # 不判断每一行文字是否旋转，继续降低推理开销。
        use_textline_orientation=False,
    )


def clean_pdf_text(text: str) -> str:
    """清理 PDF 单页抽取结果：删除空字符、空行和行首尾空格。"""
    # PDF 中可能含 \x00；先删除，再逐行 strip，最后只保留非空行。
    return "\n".join(line.strip() for line in text.replace("\x00", "").splitlines() if line.strip())


def clean_text(text: str) -> str:
    """统一清洗全文：处理全角空格、连续空白和过多换行。"""
    # 删除空字符，并把中文全角空格替换成普通半角空格。
    text = text.replace("\x00", "").replace("\u3000", " ")
    # 同一行内多个空格或 Tab 合并为一个空格，减少无意义 token。
    text = re.sub(r"[ \t]+", " ", text)
    # 三个及以上换行压缩成两个，保留段落边界但避免大片空白。
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _watermark_key(line: str) -> str:
    """归一化一行文字，用于识别字体或空格不同但内容相同的水印。"""
    # 去掉所有空白并转为小写，使“内部 资料”和“内部资料”得到相同 key。
    return re.sub(r"\s+", "", line).strip().casefold()


def remove_repeated_watermarks(page_texts: list[str]) -> tuple[list[str], list[str]]:
    """删除多数页面重复出现的短文本水印，并返回清理后的页面和水印列表。

    这是面向 RAG 的文本去水印：同时覆盖 PDF 文字层和 OCR 结果，不改动原始
    PDF。表格行含 ``|``，为避免误删跨页表头，不参与水印判断。
    """
    # 配置关闭时原样返回；第二个返回值是“发现的水印列表”。
    if not settings.watermark_removal_enabled:
        return page_texts, []
    # 至少要求 2 页；默认配置要求水印至少出现在 3 页。
    minimum_pages = max(2, settings.watermark_min_pages)
    # 页数太少时无法可靠判断“跨页重复”，因此不做删除。
    if len(page_texts) < minimum_pages:
        return page_texts, []
    # page_occurrences 记录某一规范化文字出现在多少个不同页面。
    page_occurrences: dict[str, int] = {}
    # original_lines 保存原始写法，便于日志展示删除了什么。
    original_lines: dict[str, str] = {}
    # 逐页统计；同一句在同一页出现多次仍只算 1 页。
    for text in page_texts:
        page_keys: set[str] = set()
        for line in text.splitlines():
            # 去除行首尾空白后再判断。
            clean_line = line.strip()
            key = _watermark_key(clean_line)
            # 空行忽略；含 | 的表格行受保护；过长内容通常是正文，不当作水印。
            if not key or "|" in clean_line or len(key) > settings.watermark_max_chars:
                continue
            # set 保证同一页只计数一次。
            page_keys.add(key)
            # 只保存第一次看到的原始文字。
            original_lines.setdefault(key, clean_line)
        # 页面扫描结束后，再累计“出现页数”。
        for key in page_keys:
            page_occurrences[key] = page_occurrences.get(key, 0) + 1
    # required_pages 同时满足最少页数和页面覆盖比例，默认取 3 页与 60% 中较大者。
    required_pages = max(
        minimum_pages,
        # +0.9999 再取 int，相当于对正数向上取整。
        int(len(page_texts) * settings.watermark_page_ratio + 0.9999),
    )
    # 达到阈值的短文本被视为重复页眉、页脚或文字水印。
    watermark_keys = {
        key for key, count in page_occurrences.items() if count >= required_pages
    }
    # 没发现水印时不复制或改写页面文本。
    if not watermark_keys:
        return page_texts, []
    # 对每一页删除命中的水印行。
    cleaned_pages = []
    for text in page_texts:
        kept_lines = [
            line for line in text.splitlines()
            if _watermark_key(line.strip()) not in watermark_keys
        ]
        # 删除后再次清理可能留下的空行。
        cleaned_pages.append(clean_pdf_text("\n".join(kept_lines)))
    # 把规范化 key 转回人能读懂的原始文字。
    removed = [original_lines[key] for key in sorted(watermark_keys)]
    # 日志可以帮助管理员判断是否误删。
    logger.info("自动去除 %s 条跨页重复文字水印：%s", len(removed), removed)
    return cleaned_pages, removed


def _ocr_file(path: Path) -> str:
    """对一张图片执行 PP-OCRv6，并把识别结果按行拼成纯文本。"""
    # 用列表收集文字，避免循环内反复拼接字符串。
    texts: list[str] = []
    # predict 可能返回多个页面/区域结果，因此需要遍历。
    for item in ocr_engine().predict(str(path)):
        # PaddleOCR 3.x 把识别文字放在 json.res.rec_texts 中。
        texts.extend(
            text.strip()
            for text in item.json.get("res", {}).get("rec_texts", [])
            if text.strip()
        )
    # 一条识别结果占一行，便于后续分块。
    return "\n".join(texts)


def _default_rope_parameters(config, device=None, **_kwargs):
    """兼容 PaddleOCR-VL 官方代码与 Transformers 5.0 的默认 RoPE。"""
    # torch 在函数内部导入，只有真正加载视觉模型时才占用初始化时间。
    import torch

    # 某些配置直接提供 head_dim；没有时用隐藏维度除以注意力头数计算。
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    # RoPE 只对偶数维位置计算频率，因此步长为 2。
    positions = torch.arange(0, head_dim, 2, device=device, dtype=torch.float)
    # 返回逆频率和 attention scaling；1.0 表示不额外缩放。
    return 1.0 / (config.rope_theta ** (positions / head_dim)), 1.0


@lru_cache(maxsize=1)
def vl_components():
    """延迟加载 PaddleOCR-VL 模型与预处理器。

    这一段较复杂的原因不是 RAG 业务，而是 PaddleOCR-VL 官方远程代码与当前
    Transformers 5.0 的参数名存在差异，需要在加载前做小范围兼容处理。
    """
    # 视觉模型推理依赖 PyTorch。
    import torch
    # AutoProcessor 负责图片缩放、提示词模板和 token 化。
    from transformers import AutoProcessor
    # dynamic_module_utils 用于从本地模型目录载入厂商自定义 Python 类。
    from transformers import dynamic_module_utils
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    # 注册上面的 RoPE 兼容计算函数。
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    # 模型目录来自 .env 的 PADDLEOCR_VL_MODEL。
    model_path = settings.paddleocr_vl_model
    # 本项目要求模型提前下载到本地；路径错误时给出明确提示。
    if not model_path.exists():
        raise FileNotFoundError(f"PaddleOCR-VL 模型目录不存在：{model_path}")
    # 创建项目内动态模块缓存目录，避免污染用户主目录。
    settings.hf_modules_cache.mkdir(parents=True, exist_ok=True)
    # Transformers 会把 trust_remote_code 加载的 Python 文件缓存到这里。
    os.environ["HF_MODULES_CACHE"] = str(settings.hf_modules_cache.resolve())
    # 强制离线，避免运行时再次连接 Hugging Face。
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    # 同步修改 Transformers 模块内部读取的缓存变量。
    dynamic_module_utils.HF_MODULES_CACHE = os.environ["HF_MODULES_CACHE"]
    # 用兼容实现覆盖默认 RoPE 初始化入口。
    ROPE_INIT_FUNCTIONS["default"] = _default_rope_parameters
    # 从模型目录的 modeling_paddleocr_vl.py 动态取得模型类。
    model_class = get_class_from_dynamic_module(
        "modeling_paddleocr_vl.PaddleOCRVLForConditionalGeneration", str(model_path)
    )
    # 找到刚才动态导入的 Python 模块，准备修正 causal mask 参数名。
    model_module = sys.modules[model_class.__module__]
    # 保存原函数，兼容包装器最终仍调用官方实现。
    original_causal_mask = model_module.create_causal_mask

    def compatible_causal_mask(*args, **kwargs):
        """把新版 Transformers 参数名转换成模型代码期望的旧参数名。"""
        # 新版传 inputs_embeds，当前 PaddleOCR-VL 代码读取 input_embeds。
        if "inputs_embeds" in kwargs:
            kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
        return original_causal_mask(*args, **kwargs)

    # 将兼容包装器放回模型模块。
    model_module.create_causal_mask = compatible_causal_mask
    # 不同版本模型可能使用下面两个旋转位置编码类名之一。
    for class_name in ("RotaryEmbedding", "Ernie4_5RotaryEmbedding"):
        rotary_class = getattr(model_module, class_name)
        # 两个类统一使用兼容的 RoPE 参数函数。
        rotary_class.compute_default_rope_parameters = staticmethod(_default_rope_parameters)
    # CPU 不适合 bfloat16，因此用 float32；非 CPU 设备使用 bfloat16 节省显存。
    model_dtype = torch.float32 if settings.model_device == "cpu" else torch.bfloat16
    # 只从本地加载权重，移动到配置设备，并切换到 eval 推理模式。
    model = model_class.from_pretrained(
        str(model_path), local_files_only=True, dtype=model_dtype
    ).to(settings.model_device).eval()
    # 加载与权重配套的图片处理器和 tokenizer。
    processor = AutoProcessor.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True, use_fast=False
    )
    # 缓存装饰器会记住这个二元组，后续不再重新加载模型。
    return model, processor


def _vl_image(path: Path, task: str) -> str:
    """使用 PaddleOCR-VL 执行 OCR、表格或图表理解任务。"""
    # torch 控制无梯度推理；PIL 负责读取各种图片格式。
    import torch
    from PIL import Image

    # 取得缓存的模型和预处理器。
    model, processor = vl_components()
    # with 保证图片文件及时关闭；统一转换 RGB 以匹配模型输入。
    with Image.open(path) as source:
        image = source.convert("RGB")
    # 构造多模态对话：一张图片 + 一条任务指令。
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": VL_PROMPTS[task]},
    ]}]
    # 套用模型自带聊天模板，并一次性生成 PyTorch Tensor。
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors="pt", images_kwargs={"size": {
            # 短边至少达到模型要求，避免图片过小。
            "shortest_edge": processor.image_processor.min_pixels,
            # 长边像素受配置限制，防止大图在 CPU 上耗时过长或内存溢出。
            "longest_edge": settings.paddleocr_vl_max_pixels,
        }},
    # 输入 Tensor 必须与模型位于同一设备。
    ).to(model.device)
    # 锁保证一次只有一个视觉生成；inference_mode 禁用梯度和训练缓存。
    with _vl_lock, torch.inference_mode():
        # max_new_tokens 控制识别结果最大长度，避免模型无限生成。
        output = model.generate(**inputs, max_new_tokens=settings.paddleocr_vl_max_new_tokens)
    # output 前半部分包含输入 prompt，只截取模型新生成的 token。
    generated = output[0][inputs["input_ids"].shape[-1]:]
    # token 解码成文字，再清理厂商定义的表格与坐标标记。
    return _clean_vl_output(processor.decode(generated, skip_special_tokens=True))


def _clean_vl_output(text: str) -> str:
    """把 PaddleOCR-VL 的结构化控制标记转换为适合检索的纯文本。"""
    # <nl> 表示换行。
    text = text.replace("<nl>", "\n")
    # 单元格开始/分隔标记统一转成竖线，形成“列1 | 列2”的可读格式。
    text = re.sub(r"<(?:f|u|l|x)cel>", " | ", text)
    # <ecel> 表示一行单元格结束。
    text = text.replace("<ecel>", "\n")
    # 坐标定位 token 对 RAG 没有可读含义，直接删除。
    text = re.sub(r"<\|LOC_(?:BEGIN|END|SEP|\d+)\|>", "", text)
    # 使用普通 PDF 文本规则再次删除空行和多余空白。
    text = clean_pdf_text(text)
    # 至少包含两个可读字符才认为识别有效，否则返回空串触发基础结果降级。
    return text if re.search(r"[A-Za-z0-9\u4e00-\u9fff]{2}", text) else ""


def _safe_vl(path: Path, task: str) -> str:
    """视觉识别安全包装：失败只记录日志，不让整份 PDF 入库失败。"""
    # 配置关闭 PaddleOCR-VL 时直接跳过。
    if not settings.paddleocr_vl_enabled:
        return ""
    try:
        # 正常执行指定视觉任务。
        return _vl_image(path, task)
    except Exception as error:
        # 视觉增强属于附加能力，失败时保留 PyMuPDF/普通 OCR 的基础结果。
        logger.warning("PaddleOCR-VL %s 识别失败，使用基础解析结果：%s", task, error)
        return ""

