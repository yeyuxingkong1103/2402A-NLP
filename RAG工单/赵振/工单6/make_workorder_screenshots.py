"""工单编号：人工智能NLP-RAG-混合检索任务。"""

import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).parent
FONT = r"C:\Windows\Fonts\msyh.ttc"


def draw_card(path, title, lines, footer="测试结果截图 · 固定小语料算法检查，不代表全库准确率"):
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
    draw.text((80, 820), footer, font=ImageFont.truetype(FONT, 18), fill="#67768a")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main():
    report = json.loads((ROOT / "hybrid_test_results.json").read_text(encoding="utf-8"))
    folder = ROOT / "问答引擎" / "混合检索测试截图"
    for result in report["strategy_results"]:
        draw_card(folder / f"{result['strategy']}.png", f"工单06 {result['strategy']}", [
            "测试问题：法定代表人 赵马克", f"首位片段页码：{result['top_page']} · 分数：{result['score']}",
            f"向量召回：{result['vector_candidates']} 条 · 全文召回：{result['fulltext_candidates']} 条",
            f"检索函数用时：{result['seconds']:.6f} 秒",
        ])
    draw_card(folder / "全文查询功能.png", "工单06 全文查询功能", [
        "布尔查询：法定代表人 AND 赵马克", "短语查询：\"法定代表人是赵马克\"",
        "模糊查询：Shenzhan → Shenzhen", "字段过滤：title / summary / body",
        "全部功能在小型固定语料上通过。",
    ])
    draw_card(folder / "三类重排.png", "工单06 三类重排", [
        "TF-IDF：使用字符 n-gram 文本相似度调整候选顺序。",
        "LLM：OpenAI兼容接口返回片段相关度 JSON；测试用模拟客户端。",
        "用户反馈：按用户有帮助/无帮助记录轻量调分。",
        "真实LLM端点未在本次测试中调用。",
    ])
    draw_card(folder / "权重与融合配置.png", "工单06 权重与融合配置", [
        "检索方式：向量 / 全文 / 混合。", "向量权重：0.0 到 1.0，可调。",
        "融合：加权平均 / RRF 投票 / 排序投票。", "嵌入模型：m3e-small / bge-small-zh-v1.5。",
        "更换嵌入模型后需重新建立索引。",
    ])
    evaluation = json.loads((ROOT / "hybrid_evaluation_results.json").read_text(encoding="utf-8"))
    lines = [f"评测文档：{evaluation['source']} · 固定问题6道"]
    for name, result in evaluation["modes"].items():
        lines.append(f"{name}：答案关键词覆盖 {result['answer_accuracy_by_keywords']:.0%}；"
                     f"证据关键词召回 {result['evidence_recall_by_keywords']:.0%}；"
                     f"最大检索时间 {result['max_seconds']:.3f} 秒")
    lines.append("关键词自动评估，不等同人工语义准确率。")
    draw_card(folder / "全PDF六题评测.png", "工单06 招股说明书2 六题检索评测", lines,
              "固定六题关键词评估；不等同人工语义准确率")
    smoke = json.loads((ROOT / "app_smoke_test_results.json").read_text(encoding="utf-8"))
    draw_card(ROOT / "问答界面" / "01-混合检索配置页面测试.png", "工单06 检索配置页面冒烟测试", [
        f"检查通过：{'、'.join(smoke['checks'])}", "模型、方式、权重、融合和重排可在侧栏配置。",
    ])


if __name__ == "__main__":
    main()
