# 工单编号：人工智能 NLP-RAG-混合检索任务
import json
import base64
from zhipuai import ZhipuAI
from config_v6 import CHUNK_FILE, ZHIPU_API_KEY, ZHIPU_MODEL

client = ZhipuAI(api_key=ZHIPU_API_KEY)

PROMPT = """请详细描述这张招股说明书中的图片内容，包括：
1. 图片类型（组织结构图 / 柱状图 / 折线图 / 流程图 / 表格截图等）
2. 图中的所有文字（标题、标签、数值、部门名等）
3. 如果是组织结构图：列出所有层级和部门名称，尤其注意销售部、大客户销售部的下级机构数量
4. 如果是柱状图/折线图：列出每个类别的数值，指出最高、最低、增长率最快、负增长的类别
5. 保留原文用词，不要总结太简略

图片内容："""


def image_caption(path):
    with open(path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    resp = client.chat.completions.create(
        model=ZHIPU_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img_b64}},
                {"type": "text", "text": PROMPT}
            ]
        }]
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    with open(CHUNK_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = data["chunks"]
    images = data["images"]
    print("待解析图像:", len(images))

    image_chunks = []
    for i, img in enumerate(images):
        try:
            cap = image_caption(img["path"])
            image_chunks.append({
                "page": img["page"],
                "text": f"[图像描述] {cap}",
                "type": "image",
                "source": img["source"],
                "path": img["path"]
            })
            print(f"{i+1}/{len(images)} ok, page={img['page']}")
        except Exception as e:
            print(f"{i+1}/{len(images)} fail: {e}")

    chunks.extend(image_chunks)
    data["chunks"] = chunks
    with open(CHUNK_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("总片段数:", len(chunks))