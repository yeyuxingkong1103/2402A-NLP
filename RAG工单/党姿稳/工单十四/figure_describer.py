# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-14-修复低质量工业PDF的解析与信息丢失
# 模块职责：对图纸页图像生成文字描述（跨模态信息的文本化）。优先调用本地视觉语言模型，
#           模型不可用时退化为"图内标注 + 邻页文字"的规则化描述。结果缓存到 data/ 复用。

import os                                                        # 路径与文件操作
import re                                                        # 噪声过滤正则
import json                                                      # 描述缓存
import config                                                    # 配置
from logger import get_logger                                    # 日志

logger = get_logger("rag14.figure")

PROMPT = (
    "这是一份中文专利说明书附图中的技术图纸，请用中文客观描述，不要臆造：\n"
    "1) 图号以及图纸表现的是什么装置；\n"
    "2) 图中出现的所有数字与字母标注（如 4、5、6、10、X1、h2）分别指向哪个部件；\n"
    "3) 各标注部件之间的相对位置关系（左右、上下、内外）；\n"
    "4) 图中箭头标示的气流或运动方向，按顺序它会先经过哪个部件、再经过哪个部件。\n"
    "请分点作答，只描述图中确实能看到的内容。"
)

_MODEL_CACHE = {"model": None, "processor": None}                # VLM 与处理器单例


def vlm_available():
    """判断本地 VLM 模型文件是否就绪。"""
    p = config.VLM_MODEL_PATH                                    # 模型目录
    if not os.path.isdir(p):                                     # 目录不存在
        return False
    has_cfg = os.path.exists(os.path.join(p, "config.json"))     # 配置
    has_w = any(f.endswith(".safetensors") for f in os.listdir(p))  # 权重
    return has_cfg and has_w


def load_vlm():
    """惰性加载 Qwen2-VL，CPU + bfloat16，限制上下文长度以控制内存。"""
    if _MODEL_CACHE["model"] is not None:                        # 已加载直接复用
        return _MODEL_CACHE["model"], _MODEL_CACHE["processor"]
    import torch                                                 # 延迟导入
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    logger.info("加载视觉语言模型 %s ...", os.path.basename(config.VLM_MODEL_PATH))
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        config.VLM_MODEL_PATH, torch_dtype=torch.bfloat16,     # bf16 在 CPU 上比 fp32 省一半内存
        low_cpu_mem_usage=True, device_map="cpu")               # 仅用 CPU
    model.eval()                                                 # 推理模式
    processor = AutoProcessor.from_pretrained(
        config.VLM_MODEL_PATH, min_pixels=256 * 28 * 28, max_pixels=1024 * 28 * 28)
    _MODEL_CACHE["model"], _MODEL_CACHE["processor"] = model, processor
    logger.info("视觉语言模型加载完成")
    return model, processor


def describe_with_vlm(image_path, max_new_tokens=320):
    """调用 VLM 生成图纸描述；失败返回 None。"""
    try:
        import torch                                             # 延迟导入
        from PIL import Image                                    # 读图
        model, processor = load_vlm()                            # 单例模型
        image = Image.open(image_path).convert("RGB")            # 打开图纸
        messages = [{"role": "user", "content": [               # 构造对话
            {"type": "image"}, {"type": "text", "text": PROMPT}]}]
        chat = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[chat], images=[image],          # 预处理
                           return_tensors="pt", padding=True)
        with torch.inference_mode():                             # 关闭梯度，省内存
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False)                # 贪心解码保证可复现
        text = processor.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                      skip_special_tokens=True)[0]
        return text.strip()
    except Exception as exc:                                     # 任何异常都退化为规则描述
        logger.warning("VLM 描述失败（%s），改用规则化描述", exc)
        return None


_NOISE_RE = re.compile(r"说明书附图|^\d{5,}|^第\d+/\d+页$|^图\s*\d*$")  # 页眉/页脚/图号等非标注噪声


def rule_description(page):
    """规则化兜底描述：用图内 OCR 标注 + 图号拼一段可用文本。

    OCR 会把附图页的页眉、专利号、页码一并识别出来，这些不是图中的部件标注，
    需要先过滤，只保留数字/字母类的标注（如 4、5、6、X1、h2）。
    """
    labels = "、".join(page.get("labels") or ["图1"])             # 图号
    tokens = []                                                  # 有效标注
    for box in page.get("ocr_boxes", []):                        # 逐条 OCR 结果
        t = box["text"].strip()                                  # 去掉空白
        if not t or _NOISE_RE.search(t):                         # 过滤页眉页脚
            continue
        if re.fullmatch(r"[A-Za-z]?[0-9]{1,3}['\"]?|[A-Za-z]{1,2}[0-9]{0,2}", t):
            tokens.append(t)                                     # 只保留"部件编号/尺寸代号"形态
    uniq = list(dict.fromkeys(tokens))                           # 去重保序
    body = "、".join(uniq) if uniq else "（未识别出标注）"          # 标注列表
    return f"第{page['page_no']}页附图为{labels}，图中识别到的部件/尺寸标注包括：{body}。"


def build_descriptions(pages, figure_files, cache_file, doc_name, use_vlm=True):
    """为一组图纸页生成描述，按页码缓存到 JSON，重复运行直接复用缓存。

    figure_files —— {page_no: 图片路径}；doc_name —— 来源文档名
    返回 {f"{doc_name}#p{page_no}": 描述文本}，键格式与 chunker 保持一致。
    """
    cache = {}                                                   # 旧缓存
    if os.path.exists(cache_file):                               # 读已有缓存，避免重复推理
        with open(cache_file, "r", encoding="utf-8") as fh:
            cache = json.load(fh)

    use_vlm = use_vlm and vlm_available()                        # 模型不存在则强制走规则化
    if not use_vlm:
        logger.warning("未检测到本地 VLM（%s），图纸描述将使用规则化兜底方案",
                       config.VLM_MODEL_PATH)

    result = {}                                                  # 本次结果
    for page in pages:                                           # 遍历页面
        if not page.get("figure"):                               # 非图纸页跳过
            continue
        key = f"{doc_name}#p{page['page_no']}"                   # 缓存键
        if key in cache and cache[key].strip():                  # 命中缓存直接复用
            result[key] = cache[key]
            logger.info("  图纸 p%d 复用缓存描述（%d 字）", page["page_no"], len(cache[key]))
            continue
        text = None                                              # 描述文本
        if use_vlm:                                              # 优先 VLM
            img = figure_files.get(page["page_no"])              # 对应图纸图片
            if img and os.path.exists(img):                      # 图片存在
                text = describe_with_vlm(img)                    # 推理
        if not text:                                             # 退化
            text = rule_description(page)
        result[key] = text                                       # 记录
        logger.info("  图纸 p%d 生成描述（%d 字）", page["page_no"], len(text))
    with open(cache_file, "w", encoding="utf-8") as fh:          # 写回缓存
        json.dump(result, fh, ensure_ascii=False, indent=2)
    return result
