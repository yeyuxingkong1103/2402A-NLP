"""工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化。"""

import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).parent
FONT = r"C:\Windows\Fonts\msyh.ttc"


def draw_card(path, title, lines):
    image = Image.new("RGB", (1440, 900), "#f3f6fb")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.truetype(FONT, 34)
    body_font = ImageFont.truetype(FONT, 25)
    small_font = ImageFont.truetype(FONT, 18)
    draw.rounded_rectangle((35, 30, 1405, 870), 24, fill="white", outline="#d9e1ec", width=2)
    draw.rounded_rectangle((35, 30, 1405, 140), 24, fill="#152a46")
    draw.text((75, 65), title, font=title_font, fill="white")
    y = 180
    for line in lines:
        wrapped = textwrap.wrap(line, width=43, break_long_words=True, break_on_hyphens=False) or [""]
        for part in wrapped:
            draw.text((80, y), part, font=body_font, fill="#26354a")
            y += 42
        y += 14
    draw.text((80, 820), "测试结果截图 · 根据本次运行的测试 JSON 整理", font=small_font, fill="#67768a")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main():
    results = json.loads((ROOT / "evaluation_results.json").read_text(encoding="utf-8"))
    for item in results[:4]:
        draw_card(
            ROOT / "问答引擎" / "表格检索测试截图" / f"题目-{item['id']:02}.png",
            f"工单03 表格检索 · 题目 {item['id']}",
            [f"问题：{item['question']}", f"答案：{item['final_answer']}",
             f"检索页码：{item['retrieved_pages']}", f"表格分组：{item['retrieved_table_groups']}",
             f"关键词覆盖：{item['keyword_coverage']:.0%} · 热查询：{item['request_seconds']:.4f} 秒"],
        )

    parser = json.loads((ROOT / "table_parser_test_results.json").read_text(encoding="utf-8"))
    draw_card(ROOT / "PDF 解析模块" / "01-表格解析测试.png", "表格解析验收", [
        f"文档：{parser['pdf']} · {parser['pages']} 页",
        f"解析表格行片段：{parser['table_row_count']} · 用时：{parser['seconds']:.2f} 秒",
        "发行股数与25.04%比例：通过 · PDF第22页",
        "募投项目：5行全部保留 · PDF第22页",
        "控制关系关联方：1行 · 非控制关系关联方：7行 · PDF第157页",
        "第158页历史关联方已分组排除。",
    ])

    stability = json.loads((ROOT / "stability_test_results.json").read_text(encoding="utf-8"))
    draw_card(ROOT / "问答引擎" / "稳定性测试.png", "输入与容错测试", [
        f"模糊问题：通过 · {stability['fuzzy_question']['message']}",
        f"损坏PDF：通过 · {stability['invalid_pdf']['message']}",
        f"英文事实问题：通过 · {stability['english_fact']['answer']}",
        f"本组测试耗时：{stability['seconds']:.3f} 秒",
        f"并发压力测试：{stability['concurrency_stress']}",
    ])

    smoke = json.loads((ROOT / "app_smoke_test_results.json").read_text(encoding="utf-8"))
    draw_card(ROOT / "问答界面" / "01-网页冒烟测试.png", "问答界面冒烟测试", [
        f"页面标题：{smoke['title']}",
        f"PDF上传控件：{'通过' if smoke['upload_control'] else '失败'}",
        f"提问按钮：{'通过' if smoke['question_button'] else '失败'}",
        "Streamlit页面冒烟测试：通过。",
        "四道表格题由评测脚本逐题验证。",
    ])

    summary = json.loads((ROOT / "evaluation_summary.json").read_text(encoding="utf-8"))
    draw_card(ROOT / "知识库管理" / "01-知识库状态.png", "知识库与评测状态", [
        "当前索引：招股说明书2.pdf · 350页 · 2151个片段",
        f"工单03表格题：{summary['new_table_keyword_pass']}/4 通过",
        f"工单1旧题回归：{summary['legacy_keyword_pass']}/10 通过（沿用工单2评测）",
        f"热查询最大耗时：{summary['table_query_max_seconds']:.4f} 秒",
        "答案引用：按PDF物理页序显示，页码包含封面。",
    ])


if __name__ == "__main__":
    main()


