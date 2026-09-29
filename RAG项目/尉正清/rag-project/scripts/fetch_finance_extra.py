# -*- coding: utf-8 -*-
"""从 Fin-Eva 数据集补充金融理财师知识库。

    git clone --depth 1 https://github.com/alipay/financial_evaluation_dataset /tmp/fineva
    .venv/bin/python -m scripts.fetch_finance_extra --src /tmp/fineva/data/Ant

数据来源：蚂蚁集团与上海财经大学联合发布的 Fin-Eva（CC BY 4.0）。
原始结构是四选一单选题，直接拿来当知识库价值不高，因此分两种方式提取：

    带 context 的  context 是保险条款原文、产品分析、事件解读这类**真知识**，
                   抽成独立知识文档（同一段条款在多个文件里重复出现，按内容去重）
    无 context 的   把「问题 + 正确选项原文」转成问答对，选项文字就是答案

刻意排除的目录：
    内容生成   营销文案、标题生成——是文案样本不是知识
    金融认知   情绪识别、槽位识别——是 NLP 标注任务，产出是标签不是知识
    执业医师 / 执业药师  与金融无关
"""
import argparse
import csv
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.dataset_io import did, write_jsonl

OUT_DIR = "data/financial_advisor"

# (相对路径, 标签, 是否把 context 抽成知识文档)
SELECTED = [
    # —— 带 context 的真知识 ——
    ("金融逻辑/保险条款解读.csv", "保险条款解读", True),
    ("金融逻辑/金融产品分析.csv", "金融产品分析", True),
    ("金融逻辑/金融事件解读.csv", "金融事件解读", True),
    ("金融逻辑/保险属性抽取.csv", "保险条款要素", True),
    ("金融知识/金融文档抽取.csv", "金融文档解读", True),
    # —— 无 context，转成问答对 ——
    ("金融知识/理财知识解读.csv", "理财知识", False),
    ("金融知识/金融术语解释.csv", "金融术语", False),
    ("金融知识/保险知识解读.csv", "保险知识", False),
    ("金融逻辑/金融数值计算.csv", "金融计算", False),
    ("安全合规/金融事实性.csv", "金融事实性", False),
    ("安全合规/金融合规性.csv", "金融合规", False),
    # —— 从业资格考试：职业资格知识，专业且成体系 ——
    ("金融知识/基金从业资格考试.csv", "基金从业资格", False),
    ("金融知识/证券从业资格考试.csv", "证券从业资格", False),
    ("金融知识/银行从业资格考试.csv", "银行从业资格", False),
    ("金融知识/期货从业资格考试.csv", "期货从业资格", False),
    ("金融知识/保险从业资格考试.csv", "保险从业资格", False),
    ("金融知识/会计从业资格考试.csv", "会计从业资格", False),
]

MIN_CONTEXT = 60        # 太短的 context 没有独立价值
MIN_QUESTION = 6

# 上财部分的学科名来自文件名（political_economy_val.csv），转成中文标签
SUFE_LABELS = {
    "accounting": "会计学", "advanced_financial_accounting": "高级财务会计",
    "auditing": "审计学", "banking_practitioner_qualification_certificate": "银行从业资格",
    "central_banking": "中央银行学", "certified_practising_accountant": "注册会计师",
    "commercial_bank_finance": "商业银行金融", "corporate_strategy_and_risk_management": "公司战略与风险管理",
    "econometrics": "计量经济学", "economic_law": "经济法", "finance": "金融学",
    "financial_derivatives": "金融衍生品", "financial_engineering": "金融工程",
    "financial_management": "财务管理", "financial_markets": "金融市场学",
    "financial_risk_management": "金融风险管理", "fund_qualification_certificate": "基金从业资格",
    "futures_practitioner_qualification": "期货从业资格", "insurance": "保险学",
    "intermediate_financial_accounting": "中级财务会计", "international_economics": "国际经济学",
    "international_finance": "国际金融", "investments": "投资学", "macroeconomics": "宏观经济学",
    "management_accounting": "管理会计", "microeconomics": "微观经济学",
    "monetary_finance": "货币金融学", "political_economy": "政治经济学",
    "public_finance": "财政学", "securities_practitioner_qualification": "证券从业资格",
    "statistics": "统计学", "tax_law": "税法",
}


def sfe_label(stem: str) -> str:
    """文件名 -> 中文标签；查不到就原样返回并把下划线换成空格。"""
    return SUFE_LABELS.get(stem, stem.replace("_", " "))


def read_csv(path):
    with open(path, encoding="utf-8-sig", errors="ignore") as f:
        return list(csv.DictReader(f))


def pick_answer(row):
    """单选题：把答案字母映射回选项原文。"""
    letter = (row.get("answer") or "").strip().upper()[:1]
    if letter not in ("A", "B", "C", "D", "E"):
        return None
    text = (row.get(letter) or "").strip()
    return text or None


# 上财部分：test 集把 answer 列删掉了（3340 行用不了），val 与 dev 保留了答案
SUFE_DIRS = ["SUFE/val", "SUFE/dev"]


def build_sufe(src_root):
    """把上财部分的 val/dev 转成问答对（这些文件的 answer 列是完整的）。"""
    qa, files = [], 0
    for sub in SUFE_DIRS:
        d = os.path.join(src_root, sub)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".csv"):
                continue
            rows = read_csv(os.path.join(d, fn))
            if not rows or "answer" not in rows[0]:
                continue
            files += 1
            stem = fn.replace("_%s.csv" % sub.split("/")[-1], "").replace(".csv", "")
            label = sfe_label(stem)
            n = 0
            for i, row in enumerate(rows):
                question = (row.get("question") or "").strip()
                answer = pick_answer(row)
                if len(question) < MIN_QUESTION or not answer:
                    continue
                # 该子集带解析，优先用解析作为答案补充
                explain = (row.get("explanation") or "").strip()
                body = answer + (("\n解析：%s" % explain) if explain else "")
                qa.append({
                    "doc_id": did("fineva", "sufe", sub, fn, i, question),
                    "role_id": "financial_advisor",
                    "source": "finance_exam_qa.jsonl",
                    "doc_type": "qa",
                    "embed_text": question,
                    "display_text": "【%s】\n问：%s\n答：%s" % (label, question, body),
                    "meta": {"category": label, "license": "CC BY 4.0",
                             "origin": "Fin-Eva SUFE (上海财经大学)"},
                })
                n += 1
            if n:
                print("  %-34s 问答 %4d 条" % (("%s/%s" % (sub, fn))[:34], n))
    print("  （上财部分共 %d 个文件，%d 条问答）" % (files, len(qa)))
    return qa


def build(src_dir):
    knowledge, qa = [], []
    seen_ctx = set()

    for rel, label, use_context in SELECTED:
        path = os.path.join(src_dir, rel)
        if not os.path.exists(path):
            print("  [跳过] 文件不存在: %s" % rel)
            continue
        rows = read_csv(path)
        n_ctx = n_qa = 0

        for i, row in enumerate(rows):
            ctx = (row.get("context") or "").strip()
            question = (row.get("question") or "").strip()
            answer = pick_answer(row)

            # 1) context 抽成知识文档（跨文件按内容去重）
            if use_context and len(ctx) >= MIN_CONTEXT:
                key = hashlib.md5(ctx.encode("utf-8")).hexdigest()
                if key not in seen_ctx:
                    seen_ctx.add(key)
                    title = ctx.split("\n")[0].strip()[:52]
                    # context 常以标题开头，直接前置会重复一遍，这里剥掉
                    body = ctx
                    if body.startswith(title):
                        body = body[len(title):].lstrip()
                    knowledge.append({
                        "doc_id": did("fineva", "ctx", key),
                        "role_id": "financial_advisor",
                        "source": "finance_extra_knowledge.jsonl",
                        "doc_type": "knowledge",
                        "embed_text": title,
                        "display_text": "【%s】%s\n%s" % (label, title, body or ctx),
                        "meta": {"title": title, "category": label,
                                 "license": "CC BY 4.0",
                                 "origin": "Fin-Eva (蚂蚁集团 / 上海财经大学)"},
                    })
                    n_ctx += 1

            # 2) 问 + 正确选项 -> 问答对
            if len(question) >= MIN_QUESTION and answer:
                qa.append({
                    "doc_id": did("fineva", "qa", rel, i, question),
                    "role_id": "financial_advisor",
                    "source": "finance_exam_qa.jsonl",
                    "doc_type": "qa",
                    "embed_text": question,
                    "display_text": "【%s】\n问：%s\n答：%s" % (label, question, answer),
                    "meta": {"category": label, "license": "CC BY 4.0",
                             "origin": "Fin-Eva (蚂蚁集团 / 上海财经大学)"},
                })
                n_qa += 1

        print("  %-34s 知识 %4d 条 | 问答 %4d 条" % (rel[:34], n_ctx, n_qa))

    return knowledge, qa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/tmp/fineva/data/Ant")
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        print("数据目录不存在: %s\n请先执行:\n"
              "  git clone --depth 1 "
              "https://github.com/alipay/financial_evaluation_dataset /tmp/fineva"
              % args.src)
        return 1

    print("从 Fin-Eva 提取（源: %s）\n" % args.src)
    knowledge, qa = build(args.src)

    # 上财部分：与蚂蚁部分同属 Fin-Eva，另一套学科体系
    root = os.path.dirname(args.src)
    if os.path.isdir(os.path.join(root, "SUFE")):
        print("\n上财部分:")
        qa += build_sufe(root)

    # 注意：不能写成 finance_knowledge.jsonl —— 那个文件名已被 DISC-FinLLM
    # 的 286 篇金融材料占用，会覆盖
    write_jsonl(os.path.join(OUT_DIR, "finance_extra_knowledge.jsonl"), knowledge)
    write_jsonl(os.path.join(OUT_DIR, "finance_exam_qa.jsonl"), qa)
    print("\n合计新增 %d 条（知识 %d + 问答 %d）"
          % (len(knowledge) + len(qa), len(knowledge), len(qa)))
    print("数据来源 Fin-Eva，许可 CC BY 4.0，使用时请保留署名")
    return 0


if __name__ == "__main__":
    sys.exit(main())
