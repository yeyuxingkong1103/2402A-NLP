# ============================================================
# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
# 项目名称：PDF文档的图像内容解析及检索优化
# 文件：parse_pdf_images.py
# 说明：从 PDF 提取图像，用 Qwen2.5-VL 生成语义描述，
#       输出到文本供向量化检索
# ============================================================
import os
import fitz  # pymupdf
import torch
from pathlib import Path
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


BASE = "/root/autodl-tmp/projects/RAG"
PDF_LIST = [
    ("招股说明书1", f"{BASE}/0_raw_data/招股说明书1.pdf"),
    ("招股说明书2", f"{BASE}/0_raw_data/招股说明书2.pdf"),
]
IMAGE_DIR = f"{BASE}/1_trans_data/images"
VLM_PATH = "/root/autodl-tmp/models/models/Qwen--Qwen2.5-VL-7B-Instruct/snapshots/master"
MIN_IMAGE_SIZE = 150  # 小于 150x150 的图像忽略（多为图标）

os.makedirs(IMAGE_DIR, exist_ok=True)


def extract_images(pdf_path, prefix):
    """从 PDF 提取所有图像，返回 [(page_num, img_index, img_path), ...]"""
    doc = fitz.open(pdf_path)
    results = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images(full=True)
        for idx, img_info in enumerate(images):
            xref = img_info[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.width < MIN_IMAGE_SIZE or pix.height < MIN_IMAGE_SIZE:
                    pix = None
                    continue
                # 转为 RGB（部分图像是 CMYK）
                if pix.n - pix.alpha >= 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img_name = f"{prefix}_第{page_num+1}页_图{idx+1}.png"
                img_path = os.path.join(IMAGE_DIR, img_name)
                pix.save(img_path)
                pix = None
                results.append((page_num + 1, idx + 1, img_path))
            except Exception as e:
                print(f"  跳过图像 xref={xref}：{e}")
    doc.close()
    return results


def describe_image(model, processor, img_path):
    """用 Qwen2.5-VL 生成图像语义描述"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{img_path}"},
                {
                    "type": "text",
                    "text": (
                        "这是一份招股说明书中的图像。请完成两件事：\n"
                        "1. 用一段话完整描述图像的内容（包含图中所有文字、标题、结构关系、数值等）。\n"
                        "2. 如果图像是组织结构图，请列出完整的层级结构；"
                        "如果是柱状图/折线图/饼图等图表，请列出图例和对应数值。\n\n"
                        "请直接输出描述，不要添加额外说明。"
                    ),
                },
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].strip()


def main():
    print("正在加载 Qwen2.5-VL-7B-Instruct ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(VLM_PATH)
    print("VLM 加载完成")

    all_descriptions = []

    for prefix, pdf_path in PDF_LIST:
        print(f"\n{'=' * 60}")
        print(f"处理：{prefix}")
        images = extract_images(pdf_path, prefix)
        print(f"提取到 {len(images)} 张图像（≥{MIN_IMAGE_SIZE}x{MIN_IMAGE_SIZE}）")

        for i, (page_num, img_idx, img_path) in enumerate(images, 1):
            print(f"  [{i}/{len(images)}] 解析第 {page_num} 页 图 {img_idx} ...")
            try:
                desc = describe_image(model, processor, img_path)
                all_descriptions.append({
                    "source": prefix,
                    "page": page_num,
                    "img_idx": img_idx,
                    "img_path": img_path,
                    "description": desc,
                })
            except Exception as e:
                print(f"    VLM 解析失败：{e}")

    # 输出到文本
    out_path = f"{BASE}/1_trans_data/图像描述.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in all_descriptions:
            f.write(f"\n===== {item['source']} 第 {item['page']} 页 图 {item['img_idx']} =====\n")
            f.write(f"【图像路径】{item['img_path']}\n")
            f.write(f"【图像描述】\n{item['description']}\n")
            f.write("### 图像描述结束 ###\n")

    print(f"\n完成！共解析 {len(all_descriptions)} 张图像，输出到 {out_path}")


if __name__ == "__main__":
    main()