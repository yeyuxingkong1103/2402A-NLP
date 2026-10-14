# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
"""eval_question.md / ground_truth.json 的结构与可检索性测试。"""
import io
import json
import os
import re

GT = os.path.join("data", "eval", "ground_truth.json")
CHUNKS = os.path.join("data", "chunks", "chunks.json")
OVERBROAD = 200


def _gt():
    with io.open(GT, encoding="utf-8") as f:
        return json.load(f)


def test_fourteen_questions():
    gt = _gt()
    assert len(gt) == 14
    assert {it["id"] for it in gt} == set(range(1, 15))


def test_contains_wo07_ten():
    """Q1~Q10 必须来自工单07，保证逐题可比。"""
    gt = {it["id"]: it for it in _gt()}
    for i in range(1, 11):
        assert gt[i].get("origin") == "wo07", f"Q{i} 应标记 origin=wo07"


def test_every_question_has_required_fields():
    for it in _gt():
        assert it["question"].strip(), f"Q{it['id']} 问题为空"
        assert it["reference"].strip(), f"Q{it['id']} 参考答案为空"
        assert it["key_facts"], f"Q{it['id']} 缺少 key_facts"


def test_eval_question_md_in_sync():
    """eval_question.md 必须包含全部题号，且表格结构合法。

    只做 `QXX` 子串匹配无法发现表格被空行截断：GFM 在空行处终止表格，
    14 行数据会全部退化成段落文本。故这里额外断言分隔行存在、且首个
    数据行（| Q01 ...）紧跟在分隔行之后的下一行。
    """
    with io.open("eval_question.md", encoding="utf-8") as f:
        lines = f.read().splitlines()
    md = "\n".join(lines)
    for i in range(1, 15):
        assert f"Q{i:02d}" in md, f"eval_question.md 缺少 Q{i:02d}"
    delim = [n for n, ln in enumerate(lines) if re.match(r"^\|\s*-{3,}", ln)]
    assert delim, "eval_question.md 缺少表格分隔行（| --- | ...）"
    n = delim[0]
    assert n + 1 < len(lines), "eval_question.md 分隔行后没有数据行"
    assert lines[n + 1].startswith("| Q01 "), (
        f"表格首个数据行未紧跟分隔行（分隔行第 {n + 1} 行，"
        f"下一行第 {n + 2} 行为 {lines[n + 1][:40]!r}）"
    )


def test_wo08_key_facts_hittable_in_corpus():
    """08 扩展题的 key_facts 同时在承担检索判定，必须逐字可检索。

    口径同 scripts/validate_ground_truth.py：0 命中 = 金标准写错/证据不在文本层；
    命中 > 200 = 无区分度。07 的题目沿用其 evidence_facts（见验证脚本），
    这里只对新增题把关。
    """
    with io.open(CHUNKS, encoding="utf-8") as f:
        chunks = json.load(f)
    texts = [(c.get("text", "") or "") + "\n" + (c.get("parent_text", "") or "")
             for c in chunks]
    for it in _gt():
        if it.get("origin") != "wo08":
            continue
        for fact in it["key_facts"]:
            n = sum(1 for t in texts if fact in t)
            assert 0 < n <= OVERBROAD, f"Q{it['id']} key_fact「{fact}」命中 {n} 块"
