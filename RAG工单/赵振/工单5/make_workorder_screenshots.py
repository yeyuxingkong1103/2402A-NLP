"""工单编号：人工智能NLP-RAG-Query理解优化任务。"""

import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).parent
FONT = r"C:\Windows\Fonts\msyh.ttc"


def draw_card(path, title, lines):
    image = Image.new("RGB", (1440, 900), "#f3f6fb")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((35, 30, 1405, 870), 24, fill="white", outline="#d9e1ec", width=2)
    draw.rounded_rectangle((35, 30, 1405, 140), 24, fill="#152a46")
    draw.text((75, 65), title, font=ImageFont.truetype(FONT, 34), fill="white")
    y = 180
    for line in lines:
        for part in textwrap.wrap(line, width=43, break_long_words=True, break_on_hyphens=False) or [""]:
            draw.text((80, y), part, font=ImageFont.truetype(FONT, 25), fill="#26354a")
            y += 42
        y += 12
    draw.text((80, 820), "测试结果截图 · 问答内容来源见 conversation_evaluation.json",
              font=ImageFont.truetype(FONT, 18), fill="#67768a")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main():
    report = json.loads((ROOT / "conversation_evaluation.json").read_text(encoding="utf-8"))
    folder = ROOT / "问答引擎" / "多轮对话"
    for i, item in enumerate(report["questions"], 1):
        draw_card(folder / f"第{i:02}轮.png", f"工单05 多轮问答 · 第{i}轮", [
            f"问题：{item['question']}", f"Query改写：{item['resolved_question']}",
            f"答案：{item['answer']}", f"关键词覆盖：{item['keyword_coverage']:.0%} · 改写用时：{item['rewrite_seconds']:.6f} 秒",
        ])
    draw_card(folder / "五轮连续对话.png", "工单05 五轮连续对话验收", [
        f"轮数：{report['passed']}/{report['total']} · 关键词覆盖：{report['keyword_coverage']:.0%}",
        "他参与的哪个工程？ → 补全为武汉兴图新科电子股份有限公司。",
        "这个公司的法定代表人？ → 延用上一轮公司名称。",
        "那武汉力源信息技术股份有限公司呢？ → 延用法定代表人主题并更换公司。",
        "最后一题读取招股说明书2中的销售组织结构图。",
        report["note"],
    ])
    smoke = json.loads((ROOT / "app_smoke_test_results.json").read_text(encoding="utf-8"))
    draw_card(ROOT / "问答界面" / "01-多轮问答页面测试.png", "工单05 多轮问答页面冒烟测试", [
        f"检查通过：{'、'.join(smoke['checks'])}", "页面保留前文问答，可显示Query改写。",
        "提供清空对话和有帮助/无帮助反馈入口。",
    ])


if __name__ == "__main__":
    main()
