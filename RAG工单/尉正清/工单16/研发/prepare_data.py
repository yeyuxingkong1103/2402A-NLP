# 工单16：人工智能NLP-RAG-微调专用视觉语言模型
"""把 IMDR 的 QA 对 + RAGFlow 解析出的图文块，转成微调用的 JSONL

输出字段（工单要求）：`image` / `question` / `answer`

## 两条数据来源

1. **IMDR 的 QA 对**（`original_problems/questions.jsonl`）
   题面里常写「在文件中第7页的图片中，部件4相对于部件5…」，所以**图像就是
   那一页渲染出来的 PNG**。题目是四选一，训练目标写成
   `"B. 位于编号12的部件之内"`（字母 + 选项原文），让模型学会既选对又说出内容。

2. **RAGFlow 解析出的图文块**
   工单14/15 建的知识库里，图纸块带 VLM 生成的描述。这里把它们拉出来，
   配成「给我描述这张图纸 → 那段描述」的样本。

## 为什么要按文档切分训练/评估集

同一份专利会出 6 道题，如果按题随机切，同一份文档的题会同时出现在训练集和
评估集里 —— 模型见过那页图，评估就是虚高的。所以**按文档切**。

用法：
    python prepare_data.py --out data --per-doc 6 --eval-docs 60
"""
import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC = Path(r"D:\BW\RAG 工单\14-17附件\original_problems")
PDF_DIR = SRC / "documents"
QJSONL = SRC / "questions.jsonl"

PAGE_RE = re.compile(r"第\s*([0-9]+)\s*页")


def load_questions():
    """按文档分组。"""
    by_doc = defaultdict(list)
    for line in QJSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        by_doc[q["document"]].append(q)
    return by_doc


def answer_text(q):
    """把选项字母还原成「字母. 选项原文」。"""
    letter = (q.get("answer") or "").strip()[:1]
    for opt in q.get("options") or []:
        if opt.strip().startswith(letter + "."):
            return opt.strip()
    return letter


def prompt_of(q):
    """题面 + 选项 —— 与评估时用的问法保持一致。"""
    lines = [q["question"].rstrip("：:") + "："]
    lines += q.get("options") or []
    return "\n".join(lines) + "\n\n只回答正确选项的完整内容。"


def render_pages(pdf_path, pages, out_dir, dpi=140):
    """把指定页渲染成 PNG，返回 {页码: 路径}。"""
    import fitz
    out_dir.mkdir(parents=True, exist_ok=True)
    got = {}
    with fitz.open(pdf_path) as doc:
        for pno in pages:
            idx = pno - 1
            if idx < 0 or idx >= doc.page_count:
                continue
            out = out_dir / f"{pdf_path.stem}_p{pno}.png"
            if not out.exists():
                doc[idx].get_pixmap(dpi=dpi).save(out)
            got[pno] = out
    return got


def shuffle_options(q, rng):
    """把选项顺序打乱，并改写答案字母 —— 只对训练集做。

    为什么要做：IMDR 的选项位置是固定的，B 占了 43%。第一版微调（不打乱）
    训出来的模型学到了「答案多半在 B」这个**位置先验**，B 的准确率从 69%
    涨到 85%，而 D 的准确率从 28% **掉到 0%** —— 它根本没在读图。
    打乱之后位置不再有预测力，模型只能靠内容作答。
    """
    opts = list(q.get("options") or [])
    if len(opts) < 2:
        return q
    letter = (q.get("answer") or "").strip()[:1]
    correct = next((o for o in opts if o.strip().startswith(letter + ".")), None)
    if correct is None:
        return q
    rng.shuffle(opts)
    new_letter = "ABCD"[opts.index(correct)]
    q = dict(q)
    q["options"] = opts
    q["answer"] = new_letter
    return q


def build_imdr(by_doc, out_dir, eval_docs, max_train, seed=0, shuffle_train=False):
    """IMDR：带页码引用的题 → (当页渲染图, 题面, 答案)。

    **先定文档再渲染**：一份 PDF 可能上百页，把 1500 份全渲染一遍既慢又占地方。
    所以先按文档切分、按需取够条数，只渲染用得上的那些页。
    """
    docs = sorted(d for d, qs in by_doc.items()
                  if any(PAGE_RE.search(q["question"]) for q in qs)
                  and (PDF_DIR / d).exists())
    random.Random(seed).shuffle(docs)
    eval_set, train_set = set(docs[:eval_docs]), []

    n = 0
    for d in docs[eval_docs:]:
        if n >= max_train:
            break
        train_set.append(d)
        n += sum(1 for q in by_doc[d] if PAGE_RE.search(q["question"]))

    rows = {"train": [], "eval": []}
    rng = random.Random(seed + 1)
    for split, group in (("eval", eval_set), ("train", set(train_set))):
        for doc_name in sorted(group):
            use = [q for q in by_doc[doc_name] if PAGE_RE.search(q["question"])]
            pages = sorted({int(m.group(1)) for q in use for m in PAGE_RE.finditer(q["question"])})
            imgs = render_pages(PDF_DIR / doc_name, pages, out_dir / doc_name.split(".")[0])
            for q in use:
                if split == "train" and shuffle_train:
                    q = shuffle_options(q, rng)
                m = PAGE_RE.search(q["question"])
                img = imgs.get(int(m.group(1)))
                if not img:
                    continue
                rows[split].append({
                    "image": str(img),
                    "question": prompt_of(q),
                    "answer": answer_text(q),
                    "source": "imdr",
                    "doc": doc_name,
                    "term": term_of(q),
                    # term 非空 → 计入「专业术语准确性」；
                    # 其余（方位关系型）按 group 计入「图纸推理正确性」
                    "kind": kind_of(q),
                    "group": q.get("group"),
                    "letter": (q.get("answer") or "").strip()[:1],
                })
    print(f"  IMDR：渲染 {len(eval_set)} + {len(train_set)} 份 PDF，"
          f"训练 {len(rows['train'])} 条，评估 {len(rows['eval'])} 条")
    return rows


# 方位/关系词：带这些词的答案问的是「谁在谁的哪边」，不是术语
RELATION_WORDS = ("位于", "上方", "下方", "左侧", "右侧", "左边", "右边", "之间",
                  "平行", "垂直", "相邻", "之内", "之外", "内侧", "外侧",
                  "顶部", "底部", "中部", "同一", "重合", "对称")


def term_of(q):
    """从标准答案里抽出**工业术语** —— 评估「专业术语准确性」用。

    只保留名词短语型的答案（如「伸缩杆」「氧化反应器」「减震弹簧和按压弹簧杆」）。
    方位关系型的答案（「位于编号12的部件之内」「在部件57的上方」）**不算术语** ——
    它们考的是图纸上的位置关系，归到「图纸推理正确性」那一项去评。
    返回空串表示这条不计入术语指标。
    """
    ans = answer_text(q)
    body = re.sub(r"^[A-D][.、]\s*", "", ans)
    if any(w in body for w in RELATION_WORDS):
        return ""
    body = re.sub(r"(部件|编号|零件|图|第|销轴|螺栓|弹簧杆)\s*\d+", " ", body)
    body = re.sub(r"[，。；、,.;:（）()【】\[\]“”\"']", " ", body)
    body = re.sub(r"\s+", " ", body).strip()
    # 术语至少两个字，且不能只剩数字
    if len(body) < 2 or body.replace(" ", "").isdigit():
        return ""
    # 剥掉「部件N」之后剩下一句残句的（「需要 X和 的配合」），不是术语
    if re.search(r"可以|需要|同时|必须|应该|进行|不能|分别|都输出|的是", body):
        return ""
    return body


def kind_of(q):
    """这条题考的是什么：术语 or 图纸推理。"""
    return "term" if term_of(q) else "relation"


def build_ragflow(out_dir, base_url="http://localhost"):
    """RAGFlow 解析出的图纸块 → (当页渲染图, 「描述这张图纸」, 那段描述)。

    知识库是工单14/15 建的，图纸块正文以 `【技术图纸】图N｜第P页` 开头。
    """
    import urllib.request

    def api(path, token):
        req = urllib.request.Request(base_url + path,
                                     headers={"Authorization": f"Bearer {token}"})
        return json.loads(urllib.request.urlopen(req, timeout=60).read())

    rows = []
    # 哪个工单对应哪份 PDF 是确定的：工单14 用 CN100342976C，工单15 用 CN100347506C
    sources = [(Path(r"D:\Pycharm\yzq\zg6\工单15\测试\.api_token"), "CN100347506C.pdf"),
               (Path(r"D:\Pycharm\yzq\zg6\工单14\测试\.api_token"), "CN100342976C.pdf")]
    for token_file, pdf_name in sources:
        if not token_file.exists():
            continue
        token = token_file.read_text(encoding="utf-8").strip()
        ids_file = token_file.with_name(".ragflow_ids.json")
        if not ids_file.exists():
            continue
        ids = json.loads(ids_file.read_text(encoding="utf-8"))
        try:
            chunks = api(f"/api/v1/datasets/{ids['ds_id']}/documents/{ids['doc_id']}/chunks",
                         token).get("data", {}).get("chunks", [])
        except Exception as e:
            print(f"  RAGFlow({pdf_name}) 取块失败：{e}")
            continue

        pdf = PDF_DIR / pdf_name
        if not pdf.exists():
            print(f"  RAGFlow：找不到 {pdf_name}")
            continue
        for ch in chunks:
            content = ch.get("content", "")
            m = re.match(r"【技术图纸】([^\n]*)\n(.*)", content, re.S)
            if not m or len(m.group(2)) < 200:
                continue
            head, desc = m.group(1), m.group(2).strip()
            pm = re.search(r"第(\d+)页", head)
            if not (pdf and pm):
                continue
            page = int(pm.group(1))
            if page > 0 and page <= 200:
                imgs = render_pages(pdf, [page], out_dir / pdf.stem)
                img = imgs.get(page)
                if img:
                    rows.append({
                        "image": str(img),
                        "question": f"请描述这张技术图纸（{head}）。",
                        "answer": desc,
                        "source": "ragflow",
                        "doc": pdf.name,
                        "term": "",
                        "kind": "caption",
                        "group": None,
                        "letter": "",
                    })
    print(f"  RAGFlow：取到 {len(rows)} 条图文块样本")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data", help="输出目录")
    ap.add_argument("--eval-docs", type=int, default=60, help="划给评估集的文档数")
    ap.add_argument("--max-train", type=int, default=600, help="训练集上限（6GB 显存跑不动太多）")
    ap.add_argument("--ragflow", action="store_true", help="额外拉 RAGFlow 的图纸块")
    ap.add_argument("--shuffle-options", action="store_true",
                    help="训练集打乱选项顺序（消除位置先验），评估集不动")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    img_dir = out / "images"

    print("=== 1. IMDR QA 对 ===")
    rows = build_imdr(load_questions(), img_dir, args.eval_docs, args.max_train,
                    shuffle_train=args.shuffle_options)

    if args.ragflow:
        print("=== 2. RAGFlow 图文块 ===")
        rf = build_ragflow(img_dir)
        # 都进训练集：它们讲的就是训练文档里的图
        rows["train"].extend(rf)

    random.Random(0).shuffle(rows["train"])
    rows["train"] = rows["train"][: args.max_train]

    for split in ("train", "eval"):
        p = out / f"{split}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for r in rows[split]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  -> {p}  {len(rows[split])} 条")

    # 术语非空的条数（「专业术语准确性」指标只用得上这些）
    n_term = sum(1 for r in rows["eval"] if r.get("term"))
    print(f"\n评估集里带术语的题：{n_term}/{len(rows['eval'])}")


if __name__ == "__main__":
    main()
