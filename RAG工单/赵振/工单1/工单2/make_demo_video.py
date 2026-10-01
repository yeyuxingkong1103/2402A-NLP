"""将应用截图和工单2评测结果串联成演示视频。"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).parent
WIDTH, HEIGHT = 1440, 960
FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"
TITLE = ImageFont.truetype(FONT_PATH, 50)
BODY = ImageFont.truetype(FONT_PATH, 28)


def card(title, subtitle):
    image = Image.new("RGB", (WIDTH, HEIGHT), "#f3f6fb")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((85, 125, 1355, 835), radius=30, fill="white", outline="#dce4ef", width=2)
    draw.text((145, 230), title, font=TITLE, fill="#153153")
    draw.text((145, 355), subtitle, font=BODY, fill="#475569")
    draw.text((145, 745), "人工智能NLP-RAG-基于PDF文档的问答系统优化", font=BODY, fill="#64748b")
    return image


def screenshot(path):
    image = Image.open(path).convert("RGB")
    image.thumbnail((WIDTH - 60, HEIGHT - 60))
    frame = Image.new("RGB", (WIDTH, HEIGHT), "white")
    frame.paste(image, ((WIDTH - image.width) // 2, (HEIGHT - image.height) // 2))
    return frame


shots = [
    card("PDF 文档问答系统优化", "从知识库选择、证据检索到十题优化前后评估"),
    screenshot(ROOT / "知识库管理" / "01-知识库与界面.png"),
    screenshot(ROOT / "PDF 解析模块" / "09-PDF上传.png"),
    screenshot(ROOT / "问答界面" / "03-军用收入问答.png"),
    screenshot(ROOT / "问答界面" / "05-英文问答.png"),
    screenshot(ROOT / "问答引擎" / "工单2评测截图" / "十题对照总览.png"),
    screenshot(ROOT / "问答引擎" / "工单2评测截图" / "对照题-95.png"),
    screenshot(ROOT / "问答引擎" / "工单2评测截图" / "对照题-260.png"),
    card("结果与边界", "十题优化答案覆盖 10/10；热请求平均 0.024 秒。冷启动和并发压力测试不计入本次结果。"),
]

output = ROOT / "演示视频.mp4"
writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), 2, (WIDTH, HEIGHT))
if not writer.isOpened():
    raise RuntimeError("无法创建演示视频")
for image in shots:
    frame = np.asarray(image)[:, :, ::-1]
    for _ in range(4):
        writer.write(frame)
writer.release()
print(output)
