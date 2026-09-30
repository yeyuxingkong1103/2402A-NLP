# -*- coding: utf-8 -*-
"""校验 data/raw 下已下载的 PDF，并生成清单 CSV + 首页渲染图。

用法:
  python verify_pdfs.py [目录] [--manifest _下载清单.csv] [--png logs/_verify_png]

校验口径（重要）：
  这批标准 PDF 大量使用 CID/字形编码字体（如方正兰亭），pypdf extract_text() 会抽出
  "G21G22..." 这类字形码而非可读文字，因此「文本命中标准号」不能作为校验依据。
  可靠口径 = ① 文件头 %PDF- ② pypdf 能解析且页数 > 0 ③ pymupdf 渲染首页做视觉核对。

【模块说明（补充）】
  输入：目录（默认 data/raw）下所有 .pdf 文件。
  输出：
    - {目录}/{--manifest}：下载清单 CSV（utf-8-sig，Excel 可直接打开），
      列为 文件/标准号/名称/大小KB/页数/PDF头/可读文本/首页图
    - {--png}/：每份 PDF 的首页渲染图（90dpi），已存在的不重复生成
  ⚠️ 阅读提示：本文件部分代码与注释存在不一致（如 NAME_RE 在 split_name 中被引用、
    main() 内部分变量名），以实际代码为准；单独运行可能抛 NameError。
"""
import argparse
import csv
import os
import re

import fitz  # pymupdf
from pypdf import PdfReader

# 文件名形如 "GB_Z 26573-2011 菠菜生产技术规范.pdf" / "T_CCPIA 262-2025 小麦....pdf"
# 文件系统里 "/" 被 sanitize 成 "_"，所以先还原前缀里的下划线再解析
NAME_RE = re.compile(r"^(GB/T|GB/Z|GB|T/CCPIA|T/C|NY/T|DB\d*)\s+(\S+)\s*(.*)$")
# 上面的正则分 3 组捕获：1=标准类型前缀(含可选的 /T /Z)  2=编号  3=标准名称


def split_name(stem):
    """(标准号, 名称)；识别不出标准号时标准号返回 ''，名称用整个文件名。

    参数 stem：不带扩展名的文件名。
    返回：("标准类型 编号", "标准名称") 二元组。
    """
    stem = re.sub(r"^([A-Z]{1,3})_", r"\1/", stem)   # 还原被文件系统替换掉的下划线：GB_Z -> GB/Z
    m = NAME_RE.match(stem)
    return (f"{m.group(1)} {m.group(2)}", m.group(3).strip()) if m else ("", stem)


def norm(s):
    """归一化字符串：只保留字母、数字和汉字，其余字符（空格/标点/连字符等）全部剔除。

    用于标准号与 PDF 文本的宽松比对（忽略空格与全半角差异）。返回处理后的字符串。
    """
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", s)


def main():
    """主流程：遍历目录 -> 逐份校验 -> 写清单 CSV -> 打印汇总。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default="data/raw")                  # 待校验目录，默认 data/raw
    ap.add_argument("--manifest", default="_下载清单.csv")                 # 输出的清单 CSV 文件名
    ap.add_argument("--png", default=os.path.join("logs", "_verify_png"))  # 首页渲染图输出目录
    args = ap.parse_args()
    os.makedirs(args.png, exist_ok=True)

    rows = []
    for fn in sorted(os.listdir(args.dir)):
        if not fn.lower().endswith(".pdf"):
            continue
        path = os.path.join(args.dir, fn)
        stdno, name = split_name(fn[:-4])            # 从文件名（去 .pdf 后缀）解析标准号与名称

        # ① 文件头校验：前 5 字节必须是 "%PDF-"
        with open(path, "rb") as f:
            header_ok = f.read(5) == b"%PDF-"
        # ② pypdf 解析校验：能读出页数，且前 3 页文本里能宽松命中标准号才算「可读文本」
        pages, readable = -1, False
        try:
            r = PdfReader(path)
            pages = len(r.pages)
            text = "".join((r.pages[i].extract_text() or "") for i in range(min(3, pages)))  # 只抽前 3 页（含封面/扉页的标准号）
            readable = len(norm(text)) > 50 and norm(stdno) in norm(text)  # 有效字符 >50 且含标准号才算真文本（非字形码）
        except Exception as e:
            print(f"  解析失败 {fn}: {e}")

        # ③ 首页渲染图：供人工视觉核对；文件名把非法字符换成下划线、截 60 字
        png = os.path.join(args.png, re.sub(r"[^\w.-]", "_", fn[:-4])[:60] + ".png")
        try:
            doc = fitz.open(path)
            if doc.page_count > 0 and not os.path.exists(png):   # 已有图不重复渲染
                doc[0].get_pixmap(dpi=90).save(png)              # 首页按 90dpi 渲染保存
            doc.close()
        except Exception as e:
            print(f"  渲染失败 {fn}: {e}")

        rows.append({"文件": fn, "标准号": stdno, "名称": name,
                     "大小KB": os.path.getsize(path) // 1024, "页数": pages,
                     "PDF头": header_ok, "可读文本": readable, "首页图": png})

    if not rows:
        print(f"{args.dir} 下没有 PDF")
        return
    # 写清单 CSV：utf-8-sig 带 BOM，Excel 直接打开不乱码
    man = os.path.join(args.dir, args.manifest)
    with open(man, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 汇总统计与逐行报告
    print(f"共 {len(rows)} 个 PDF | PDF头正常 {sum(r['PDF头'] for r in rows)} | "
          f"页数>0 {sum(r['页数'] > 0 for r in rows)} | 文本可直接检索 {sum(r['可读文本'] for r in rows)}")
    print(f"清单: {man}\n首页图: {args.png}")
    for r in rows:
        print(f"  {r['标准号'] or '(未识别)':<18} {r['页数']:>3}页 {r['大小KB']:>5}KB  "
              f"{'OK ' if r['页数'] > 0 else 'BAD'} {'文本' if r['可读文本'] else '字形码'}  {r['名称'][:34]}")
        # 「文本/字形码」列：可读文本=True 标「文本」，否则多为字形码字体 PDF，需靠渲染图核对


if __name__ == "__main__":
    main()
