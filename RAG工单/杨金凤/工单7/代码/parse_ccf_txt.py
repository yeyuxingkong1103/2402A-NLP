# ============================================================
# 工单编号：人工智能NLP-RAG-功能测试及评估
# 项目名称：PDF文档的功能测试及评估
# 文件：parse_ccf_txt.py
# 说明：把 ccf_competition/txt/ 里的 JSONL 文件转为纯文本
# ============================================================
import os
import json

TXT_DIR = "/root/autodl-tmp/projects/RAG/0_raw_data/ccf_competition/txt"
OUT_DIR = "/root/autodl-tmp/projects/RAG/1_trans_data/ccf"
os.makedirs(OUT_DIR, exist_ok=True)


def parse_jsonl(file_path):
    """解析 JSONL 文件，按页组织文本"""
    pages = {}  # {page_num: [texts]}
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            page = obj.get("page", 0)
            inside = obj.get("inside", "")
            if not inside:
                continue
            pages.setdefault(page, []).append(inside)
    return pages


def main():
    files = sorted(os.listdir(TXT_DIR))
    print(f"共 {len(files)} 个 txt 文件")

    for i, fname in enumerate(files, 1):
        file_path = os.path.join(TXT_DIR, fname)
        pages = parse_jsonl(file_path)

        # 生成纯文本
        full_text = []
        for page_num in sorted(pages.keys()):
            full_text.append(f"\n===== 第 {page_num} 页 =====")
            full_text.extend(pages[page_num])

        # 输出文件名：用索引 + 关键词（避免乱码）
        out_name = f"ccf_{i:02d}.txt"
        out_path = os.path.join(OUT_DIR, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(full_text))

        print(f"  [{i}/{len(files)}] {fname[:40]}... → {out_name}（{len(full_text)} 行）")

    print(f"\n完成，纯文本已保存到 {OUT_DIR}")


if __name__ == "__main__":
    main()