"""
add_bitmaps_to_pdfs.py — 给三个领域的测试 PDF 插入位图图表

不修改项目代码，直接用 PIL 生成位图（真实数据）、用 PyMuPDF 插入 PDF。
数据来自 pdf_content_*.py 中已有的法规/指南/语法原文，不编造。

运行：
    python scripts/add_bitmaps_to_pdfs.py
"""
from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
PDF_DIR = HERE.parent / "data" / "generated_pdfs"

FONT_PATHS = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
ZH_FONT = next((p for p in FONT_PATHS if Path(p).exists()), None)
if not ZH_FONT:
    raise SystemExit("未找到中文字体，请检查 FONT_PATHS")


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(ZH_FONT, size)


def draw_table_image(rows: list[list[str]], title: str,
                     img_w: int = 1400, row_h: int = 56) -> Path:
    n_rows, n_cols = len(rows), len(rows[0])
    col_w = img_w // n_cols
    img_h = row_h * n_rows + 80
    img = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(img)
    draw.text((img_w // 2, 30), title, fill="#1a3d6d", font=_font(28), anchor="mm")
    y0 = 70
    f_head, f_cell = _font(24), _font(22)
    for r, row in enumerate(rows):
        is_head = (r == 0)
        draw.rectangle([0, y0 + r * row_h, img_w, y0 + (r + 1) * row_h],
                       fill="#eaf1fa" if is_head else "white", outline="#aaaaaa")
        for c, cell in enumerate(row):
            x = c * col_w
            draw.line([x, y0 + r * row_h, x, y0 + (r + 1) * row_h], fill="#aaaaaa")
            draw.text((x + col_w // 2, y0 + r * row_h + row_h // 2), str(cell),
                      fill="#1a3d6d" if is_head else "#333333",
                      font=f_head if is_head else f_cell, anchor="mm")
    tmp = HERE / f"_tmp_table_{abs(hash(title)) % 100000}.png"
    img.save(tmp, "PNG")
    return tmp


def draw_bar_image(labels: list[str], values: list[float], title: str,
                   img_w: int = 1400, img_h: int = 600) -> Path:
    img = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(img)
    draw.text((img_w // 2, 35), title, fill="#1a3d6d", font=_font(28), anchor="mm")
    margin_l, margin_r, margin_t, margin_b = 90, 40, 80, 110
    plot_w = img_w - margin_l - margin_r
    plot_h = img_h - margin_t - margin_b
    vmax = max(values) * 1.15 if values else 1
    bar_w = plot_w / len(labels) * 0.6
    gap = plot_w / len(labels)
    draw.line([margin_l, margin_t + plot_h, margin_l + plot_w, margin_t + plot_h],
              fill="#999999", width=2)
    f_lab, f_val = _font(22), _font(20)
    for i, (lab, val) in enumerate(zip(labels, values)):
        x = margin_l + i * gap + (gap - bar_w) / 2
        h = (val / vmax) * plot_h if vmax else 0
        y = margin_t + plot_h - h
        draw.rectangle([x, y, x + bar_w, margin_t + plot_h], fill="#4a90d9")
        draw.text((x + bar_w / 2, y - 10),
                  str(int(val)) if val == int(val) else f"{val:g}",
                  fill="#333333", font=f_val, anchor="mb")
        draw.text((x + bar_w / 2, margin_t + plot_h + 12), lab,
                  fill="#333333", font=f_lab, anchor="ma")
    tmp = HERE / f"_tmp_bar_{abs(hash(title)) % 100000}.png"
    img.save(tmp, "PNG")
    return tmp


def insert_image(pdf_path: Path, page_idx: int, img_path: Path,
                 x: float = 60, y: float = 620, w: float = 470) -> None:
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    rect = fitz.Rect(x, y, x + w, y + w * 0.45)
    page.insert_image(rect, filename=str(img_path))
    doc.saveIncr()
    doc.close()


# 数据均来自 pdf_content_*.py 中标注的真实法规/指南/语法来源
LEGAL_BITMAPS = [
    (2, "bar", ["不满3月", "3月-1年", "1-3年", "3年以上", "无固定期"],
     [0, 1, 2, 6, 6], "试用期上限（月）— 劳动合同法第19条"),
    (13, "bar", ["不满6月", "6月-1年", "每满1年", "超社平3倍"],
     [0.5, 1, 1, 12], "经济补偿标准（月工资）— 劳动合同法第47条"),
    (21, "bar", ["满1年不满10年", "满10年不满20年", "满20年以上"],
     [5, 10, 15], "带薪年休假天数 — 职工带薪年休假条例第3条"),
    (29, "table", [["伤残等级", "一次性补助金", "月津贴比例"],
                   ["一级", "27个月本人工资", "90%"],
                   ["二级", "25个月本人工资", "85%"],
                   ["三级", "23个月本人工资", "80%"],
                   ["四级", "21个月本人工资", "75%"]],
     "1-4级伤残工伤保险待遇 — 工伤保险条例第35条"),
]

ENGLISH_BITMAPS = [
    (0, "table", [["时间\\体", "一般", "进行", "完成", "完成进行"],
                  ["现在", "I work", "I am working", "I have worked", "I have been working"],
                  ["过去", "I worked", "I was working", "I had worked", "I had been working"],
                  ["将来", "I will work", "I will be working", "I will have worked", "I will have been working"]],
     "英语12时态结构表 — British Council Grammar Reference"),
    (17, "table", [["时间", "if 从句", "主句"],
                   ["与现在相反", "did / were", "would do"],
                   ["与过去相反", "had done", "would have done"],
                   ["与将来相反", "were to do / should do", "would do"]],
     "虚拟语气时态搭配 — Cambridge Grammar"),
    (33, "table", [["原级", "比较级", "最高级"],
                   ["tall", "taller", "tallest"],
                   ["beautiful", "more beautiful", "most beautiful"],
                   ["good", "better", "best"],
                   ["bad", "worse", "worst"]],
     "形容词比较级与最高级 — Cambridge Grammar"),
]

MEDICAL_BITMAPS = [
    (1, "bar", ["正常", "正常高值", "1级", "2级", "3级"],
     [110, 130, 150, 170, 190], "高血压分级（收缩压 mmHg）— 中国高血压防治指南2024"),
    (5, "table", [["检测指标", "糖尿病诊断标准"],
                   ["空腹血糖", "≥7.0 mmol/L"],
                   ["OGTT 2小时血糖", "≥11.1 mmol/L"],
                   ["随机血糖+症状", "≥11.1 mmol/L"],
                   ["HbA1c", "≥6.5%"]],
     "2型糖尿病诊断标准 — 中国2型糖尿病防治指南2020"),
    (7, "bar", ["一般成人", "年轻/病程短", "老年/并发症", "妊娠期"],
     [7.0, 6.5, 8.0, 6.5], "HbA1c 控制目标（%）— 中国2型糖尿病防治指南2020"),
    (9, "table", [["风险分层", "LDL-C 目标"],
                   ["低危", "<3.4 mmol/L"],
                   ["中危/高危", "<2.6 mmol/L"],
                   ["极高危", "<1.8 mmol/L 且降幅>50%"],
                   ["超高危", "<1.4 mmol/L 且降幅>50%"]],
     "LDL-C 目标值分层 — 中国血脂管理指南2023"),
]


def process(pdf_name: str, specs: list) -> None:
    pdf_path = PDF_DIR / pdf_name
    tmp_files: list[Path] = []
    for page_idx, kind, *args in specs:
        if kind == "bar":
            labels, values, title = args
            img = draw_bar_image(labels, values, title)
        else:
            rows, title = args
            img = draw_table_image(rows, title)
        tmp_files.append(img)
        insert_image(pdf_path, page_idx, img)
    for f in tmp_files:
        f.unlink(missing_ok=True)
    doc = fitz.open(pdf_path)
    n = sum(len(p.get_images()) for p in doc)
    doc.close()
    print(f"{pdf_name}: 插入 {len(specs)} 张位图，当前共 {n} 张")


def main() -> None:
    process("劳动法常见问题.pdf", LEGAL_BITMAPS)
    process("英语语法精讲.pdf", ENGLISH_BITMAPS)
    process("常见疾病诊疗指南.pdf", MEDICAL_BITMAPS)
    print("完成")


if __name__ == "__main__":
    main()
