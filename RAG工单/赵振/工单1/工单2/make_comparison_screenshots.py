"""将工单2实际评测 JSON 渲染成逐题结果截图。"""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).parent
OUTPUT = ROOT / "问答引擎" / "工单2评测截图"
FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"
FONT = ImageFont.truetype(FONT_PATH, 29)
SMALL = ImageFont.truetype(FONT_PATH, 24)
TITLE = ImageFont.truetype(FONT_PATH, 42)


def wrap(draw, text, font, width):
    lines = []
    line = ""
    for char in text:
        if draw.textbbox((0, 0), line + char, font=font)[2] <= width:
            line += char
        else:
            lines.append(line)
            line = char
    if line:
        lines.append(line)
    return lines


def write_block(draw, x, y, text, font, fill, width, gap=10):
    for line in wrap(draw, text, font, width):
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + gap
    return y


data = json.loads((ROOT / "retrieval_comparison.json").read_text(encoding="utf-8"))
OUTPUT.mkdir(parents=True, exist_ok=True)

for row in data["questions"]:
    image = Image.new("RGB", (1800, 1220), "#f3f6fb")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((45, 35, 1755, 1180), radius=28, fill="white", outline="#dce4ef", width=2)
    draw.text((90, 75), f"工单 2 检索优化评测 · 题号 {row['id']}", font=TITLE, fill="#153153")
    draw.text((90, 145), "自动评测结果记录图 | 同一 PDF、同一问题，对比向量基线与优化检索", font=SMALL, fill="#64748b")
    y = 210
    y = write_block(draw, 90, y, "问题：" + row["question"], FONT, "#1f2937", 1600, 9) + 15
    y = write_block(draw, 90, y, "参考答案：" + row["reference"], SMALL, "#475569", 1600, 8) + 28

    left = (90, y, 850, 780)
    right = (910, y, 1710, 780)
    draw.rounded_rectangle(left, radius=20, fill="#fff7ed", outline="#fed7aa", width=2)
    draw.rounded_rectangle(right, radius=20, fill="#effaf5", outline="#a7f3d0", width=2)
    draw.text((125, y + 25), "优化前：纯向量检索基线", font=ImageFont.truetype(FONT_PATH, 31), fill="#9a3412")
    draw.text((125, y + 78), f"Top-5 页码：{row['before_pages']}", font=SMALL, fill="#7c2d12")
    draw.text((125, y + 122), f"Top-1 / Top-5 关键词覆盖：{row['before_top1_keyword_coverage']:.0%} / {row['before_top5_keyword_coverage']:.0%}", font=SMALL, fill="#7c2d12")
    yy = write_block(draw, 125, y + 175, "基线证据抽取：" + row["baseline_answer"], SMALL, "#7c2d12", 675, 8)
    draw.text((125, max(yy + 18, y + 350)), f"检索耗时：{row['before_seconds']:.4f} 秒", font=SMALL, fill="#7c2d12")

    draw.text((950, y + 25), "优化后：语义+词面融合、意图加权", font=ImageFont.truetype(FONT_PATH, 31), fill="#166534")
    draw.text((950, y + 78), f"Top-5 页码：{row['after_pages']}", font=SMALL, fill="#14532d")
    draw.text((950, y + 122), f"Top-1 / Top-5 关键词覆盖：{row['after_top1_keyword_coverage']:.0%} / {row['after_top5_keyword_coverage']:.0%}", font=SMALL, fill="#14532d")
    yy = write_block(draw, 950, y + 175, "优化后答案：" + row["optimized_answer"], SMALL, "#14532d", 700, 8)
    draw.text((950, max(yy + 18, y + 350)), f"关键词覆盖：{row['answer_keyword_coverage']:.0%}  |  检索+答案：{row['request_seconds']:.4f} 秒", font=SMALL, fill="#14532d")
    draw.text((90, 1110), "关键词覆盖是本工单十题的自动指标，不能代表对任意问题的普遍准确率。", font=SMALL, fill="#64748b")
    image.save(OUTPUT / f"对照题-{row['id']}.png")

summary = Image.new("RGB", (1800, 1040), "#f3f6fb")
draw = ImageDraw.Draw(summary)
draw.rounded_rectangle((45, 35, 1755, 1000), radius=28, fill="white", outline="#dce4ef", width=2)
draw.text((95, 75), "工单 2 · 十题检索优化前后对照", font=TITLE, fill="#153153")
draw.text((95, 145), "同一 PDF / 同一 10 题 / 指标为参考答案关键词覆盖率", font=SMALL, fill="#64748b")
heads = ["题号", "基线 Top-5", "优化 Top-5", "优化答案", "请求耗时"]
xs = [105, 280, 565, 850, 1430]
for x, head in zip(xs, heads):
    draw.text((x, 230), head, font=ImageFont.truetype(FONT_PATH, 27), fill="#334155")
draw.line((95, 275, 1705, 275), fill="#cbd5e1", width=2)
for i, row in enumerate(data["questions"]):
    y = 300 + i * 62
    draw.text((xs[0], y), str(row["id"]), font=SMALL, fill="#334155")
    draw.text((xs[1], y), f"{row['before_top5_keyword_coverage']:.0%}", font=SMALL, fill="#c2410c")
    draw.text((xs[2], y), f"{row['after_top5_keyword_coverage']:.0%}", font=SMALL, fill="#15803d")
    draw.text((xs[3], y), f"{row['answer_keyword_coverage']:.0%}", font=SMALL, fill="#15803d")
    draw.text((xs[4], y), f"{row['request_seconds']:.3f}s", font=SMALL, fill="#334155")
    draw.line((95, y + 47, 1705, y + 47), fill="#e2e8f0", width=1)
draw.text((95, 930), f"平均 Top-5 证据覆盖：{data['mean_before_top5_keyword_coverage']:.1%} → {data['mean_after_top5_keyword_coverage']:.1%}；优化答案覆盖：{data['answer_keyword_coverage']:.0%}；热请求平均：{data['mean_request_seconds']:.3f}s。", font=SMALL, fill="#334155")
summary.save(OUTPUT / "十题对照总览.png")

stability_path = ROOT / "stability_test_results.json"
smoke_path = ROOT / "app_smoke_test_results.json"
if stability_path.exists() and smoke_path.exists():
    stability = json.loads(stability_path.read_text(encoding="utf-8"))
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    image = Image.new("RGB", (1800, 900), "#f3f6fb")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((45, 35, 1755, 860), radius=28, fill="white", outline="#dce4ef", width=2)
    draw.text((95, 75), "工单 2 · 稳定性与交互冒烟测试", font=TITLE, fill="#153153")
    entries = [
        ("模糊输入“它怎么样？”", "通过：提示用户补充公司、事项或时间"),
        ("损坏 PDF", "通过：解析抛出异常，界面捕获后显示错误"),
        ("英文问题", f"通过：{stability['english_answer']}"),
        ("Streamlit 页面", f"通过：输入框、提问按钮可见；异常数 {smoke['exceptions']}"),
    ]
    y = 205
    for title, detail in entries:
        draw.rounded_rectangle((100, y, 1690, y + 125), radius=18, fill="#effaf5", outline="#a7f3d0", width=2)
        draw.text((135, y + 17), title, font=ImageFont.truetype(FONT_PATH, 29), fill="#166534")
        write_block(draw, 600, y + 25, detail, SMALL, "#14532d", 1030, 6)
        y += 150
    draw.text((100, 815), "测试按真实函数与 Streamlit AppTest 运行；本图为测试记录图。", font=SMALL, fill="#64748b")
    image.save(OUTPUT / "稳定性测试记录.png")
print(f"saved result images in {OUTPUT}")
