# -*- coding: utf-8 -*-
"""把三个 GitHub 开源项目的原始数据清洗成入库友好的 JSONL。

    python -m scripts.prepare_dataset

输出 schema（每条一行 JSON）：
    doc_id       稳定唯一 ID
    role_id      所属角色
    source       来源文件（与同名 JSONL 一致）
    doc_type     qa / article / case / knowledge
    embed_text   向量化与 BM25 的检索目标（短、精准）
    display_text 拼进 prompt 给大模型看的完整上下文
    meta         附加信息

注意：源仓库（LaWGPT / MentalGLM / DISC-FinLLM）在提取完成后已删除，
本脚本保留作为数据血缘记录；若需重跑，请先重新 clone 到 data/ 下。
"""
import os
import re

from scripts.dataset_io import did, load_json, write_jsonl

DATA = "/root/rag-project/data"


# ============================================================
# 1. 法律 —— LaWGPT
# ============================================================
def build_lawyer():
    print("\n[律师] LaWGPT")
    src = os.path.join(DATA, "LaWGPT/resources")
    out = os.path.join(DATA, "lawyer")

    # 1.1 刑法罪名表 -> 法条知识条目
    recs = []
    for c in load_json(os.path.join(src, "criminal_charges.json")):
        title = (c.get("charge") or "").strip()
        if not title:
            continue
        chapter = (c.get("chapter") or "").strip()
        lv1 = (c.get("level1") or "").strip()
        lv2 = (c.get("level2") or "").strip()

        body = ["罪名：%s" % title]
        for label, val in (("法条依据", chapter), ("所属章节", lv1), ("子类", lv2)):
            if val:
                body.append("%s：%s" % (label, val))

        recs.append({
            "doc_id": did("lawyer", "charge", c.get("id"), title),
            "role_id": "lawyer",
            "source": "legal_articles.jsonl",
            "doc_type": "article",
            "embed_text": "%s %s %s" % (title, chapter, lv1),
            "display_text": "【刑法罪名条目】\n%s" % "\n".join(body),
            "meta": {"charge": title, "chapter": chapter},
        })
    write_jsonl(os.path.join(out, "legal_articles.jsonl"), recs)

    # 1.2 法律问答（含法条正文引用）
    recs = []
    for fn in ["example_instruction_tune.json", "example_infer_data.json"]:
        for i, it in enumerate(load_json(os.path.join(src, fn))):
            q = (it.get("instruction") or "").strip()
            a = (it.get("output") or "").strip()
            if not q or a in ("", "无"):
                continue
            recs.append({
                "doc_id": did("lawyer", "qa", fn, i, q),
                "role_id": "lawyer",
                "source": "legal_qa.jsonl",
                "doc_type": "qa",
                "embed_text": q,
                "display_text": "【法律问答】\n问：%s\n答：%s" % (q, a),
                "meta": {},
            })
    write_jsonl(os.path.join(out, "legal_qa.jsonl"), recs)

    # 1.3 裁判文书全文（长文，入库时会切块）
    recs = []
    for i, it in enumerate(load_json(os.path.join(src, "example_instruction_train.json"))):
        content = (it.get("content") or "").strip()
        if not content:
            continue
        head = content.split("\n")[0].strip()[:80]
        recs.append({
            "doc_id": did("lawyer", "case", i),
            "role_id": "lawyer",
            "source": "legal_cases.jsonl",
            "doc_type": "case",
            "embed_text": head,
            "display_text": "【裁判文书】\n%s" % content,
            "meta": {"title": head},
        })
    write_jsonl(os.path.join(out, "legal_cases.jsonl"), recs)

    # 1.4 法律词表 —— 保留为资源，供 BM25 自定义词典使用
    with open(os.path.join(src, "legal_vocab.txt"), encoding="utf-8") as f:
        vocab = [l.strip() for l in f if l.strip()]
    with open(os.path.join(out, "legal_vocab.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(vocab))
    print("  -> %-42s %6d 词" % ("legal_vocab.txt", len(vocab)))


# ============================================================
# 2. 心理 —— MentalGLM
# ============================================================
def build_psychologist():
    print("\n[心理医生] MentalGLM")
    src = os.path.join(DATA, "MentalGLM/Instruct data")
    out = os.path.join(DATA, "psychologist")

    for fn, outname, label in [
        ("Instruct_CD.json", "cognitive_distortion_qa.jsonl", "认知歪曲识别"),
        ("Instruct_suiside.json", "suicide_risk_qa.jsonl", "自杀风险识别"),
    ]:
        recs = []
        for i, it in enumerate(load_json(os.path.join(src, fn))):
            q = (it.get("instruction") or "").strip()
            extra = (it.get("input") or "").strip()
            a = (it.get("output") or "").strip()
            if not q or not a:
                continue
            if extra:
                q = "%s\n%s" % (q, extra)

            # 检索目标必须是**案例正文**，不能是「这个帖子体现了什么认知歪曲？」
            # ——后者对 3407 条记录是同一句，会让所有向量完全相同（实测余弦=1.0），
            # 检索彻底失效。用户描述的情形要跟案例做语义匹配。
            m = re.search(r"[“\"](.+?)[”\"]", q, re.S)
            embed = m.group(1).strip() if m else q
            if len(embed) < 6:
                embed = q

            recs.append({
                "doc_id": did("psych", outname, i, q),
                "role_id": "psychologist",
                "source": outname,
                "doc_type": "qa",
                "embed_text": embed,
                "summary": "案例：%s\n结论：%s" % (embed, a[:120]),
                "display_text": "【%s · 专业分析】\n案例：%s\n分析：%s" % (label, q, a),
                "meta": {"label": label},
            })
        write_jsonl(os.path.join(out, outname), recs)


# ============================================================
# 3. 金融 —— DISC-FinLLM
# ============================================================
CTX_PREFIX = re.compile(r"^请根据以下提供的上下文回答相应问题\s*[:：]\s*", re.S)
FIN_PREFIX = re.compile(r"^作为一名金融领域专家.*?[；;]\s*", re.S)
# 材料分隔符：不同子集用了不同写法
MATERIAL_MARK = re.compile(r"(?:参考材料\s*\d*|上下文|基于材料|输入|材料)\s*[:：]")
Q_MARK = re.compile(r"问题\s*[:：]")


def _last_after(text, pattern):
    ms = list(re.finditer(pattern, text))
    return text[ms[-1].end():].strip() if ms else None


def extract_title(text):
    """优先取 《...》 / 【...】 作为标题，否则取首行前 60 字。"""
    for pat in (r"《([^》]{4,60})》", r"【([^】]{4,60})】"):
        m = re.search(pat, text)
        if m:
            return m.group(1).strip()
    return text.split("\n")[0].strip("《》【】 ")[:60]


def extract_question(q):
    """抽出真正的问题：优先「问题:」之后，其次首行提问句，最后末尾问句。"""
    cand = _last_after(q, r"问题\s*[:：]")
    if cand and len(cand) >= 4:
        return cand
    lines = [l.strip() for l in q.split("\n") if l.strip()]
    if lines:
        first = lines[0]
        if re.match(r"^(请|如何|什么|为什么|哪|谁|怎么|试|判断|分析)", first) \
                or first.endswith("？"):
            return first
        m = re.search(r"([^。？\n]{4,80}？)", q)
        if m:
            return m.group(1).strip()
    return FIN_PREFIX.sub("", CTX_PREFIX.sub("", q)).strip()


def extract_materials(q):
    """抽出内嵌的金融材料（兼容多种分隔符），返回正文列表。"""
    out = []
    for p in MATERIAL_MARK.split(q)[1:]:
        p = Q_MARK.split(p)[0].strip()
        p = re.sub(r"^回答相应问题\s*[:：]\s*", "", p).strip()
        if len(p) >= 80:
            out.append(p)
    return out


def build_finance():
    print("\n[金融理财师] DISC-FinLLM")
    src = os.path.join(DATA, "DISC-FinLLM/data")
    out = os.path.join(DATA, "financial_advisor")
    knowledge = []
    seen_mat = set()          # 材料跨记录去重

    for fn, outname, label, min_ans in [
        ("consulting_part.json", "consulting_qa.jsonl", "金融咨询", 0),
        ("computing_part.json", "computing_qa.jsonl", "金融计算", 0),
        ("retrieval_part.json", "retrieval_qa.jsonl", "金融知识检索", 0),
        # task_part 是 NLP 任务数据（标题生成/实体识别），答案中位数仅 19 字，
        # 只保留答案充实的问答；其余价值在于内嵌材料，单独抽成知识文档
        ("task_part.json", "task_qa.jsonl", "金融文档问答", 30),
    ]:
        recs = []
        for i, it in enumerate(load_json(os.path.join(src, fn))):
            raw_q = (it.get("instruction") or "").strip()
            a = (it.get("output") or "").strip()
            if not raw_q or not a:
                continue

            # 先抽内嵌材料（必须在清洗 question 之前做）
            for m in extract_materials(raw_q):
                key = did(m)
                if key in seen_mat:
                    continue
                seen_mat.add(key)
                title = extract_title(m)
                knowledge.append({
                    "doc_id": did("fin", "mat", key),
                    "role_id": "financial_advisor",
                    "source": "finance_knowledge.jsonl",
                    "doc_type": "knowledge",
                    "embed_text": title,
                    "display_text": "【金融材料】%s\n%s" % (title, m),
                    "meta": {"title": title},
                })

            if len(a) < min_ans:
                continue
            q = extract_question(raw_q)
            if not q:
                continue

            hist_lines = []
            for turn in (it.get("history") or []):
                if isinstance(turn, (list, tuple)) and len(turn) == 2:
                    hist_lines.append("问：%s\n答：%s" % (turn[0], turn[1]))
            hist_txt = ("\n【历史对话】\n" + "\n".join(hist_lines)) if hist_lines else ""

            recs.append({
                "doc_id": did("fin", outname, i, q),
                "role_id": "financial_advisor",
                "source": outname,
                "doc_type": "qa",
                "embed_text": q,
                "display_text": "【%s】\n问：%s%s\n答：%s" % (label, q, hist_txt, a),
                "meta": {"label": label},
            })
        write_jsonl(os.path.join(out, outname), recs)

    write_jsonl(os.path.join(out, "finance_knowledge.jsonl"), knowledge)


def summarize():
    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    total = 0
    for role in ["lawyer", "psychologist", "financial_advisor"]:
        d = os.path.join(DATA, role)
        n = 0
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".jsonl"):
                with open(os.path.join(d, fn), encoding="utf-8") as f:
                    n += sum(1 for l in f if l.strip())
        print("  %-20s %6d 条" % (role, n))
        total += n
    print("  %-20s %6d 条" % ("合计", total))


if __name__ == "__main__":
    print("=" * 70)
    print("开始整理数据集")
    print("=" * 70)
    build_lawyer()
    build_psychologist()
    build_finance()
    summarize()
