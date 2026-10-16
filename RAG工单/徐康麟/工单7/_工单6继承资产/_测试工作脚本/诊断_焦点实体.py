# -*- coding: utf-8 -*-
"""人工智能NLP-RAG-功能测试及评估 —— 多轮指代消解根因诊断（本地纯逻辑，不调模型、不读索引）。

复现并定位 ``回归结果.json`` 中 TC-M-C2 失败的直接原因：
第 1 轮问句自带「平安银行」实体，但改写器把 ``focus_entity`` 写成了时间词「2019年」；
第 2 轮「它的拨备覆盖率是多少？」按 ``pronoun`` 策略把「它」替换成「2019年」，
检索因此丢失主体约束 ⇒ 命中并引用了**招商银行**年报（跨语料）。

输出：实体识别结果 / 指代识别结果 / ``pending_subject`` 主体正则的实测命中情况。
"""

from __future__ import annotations

import sys

sys.path.insert(0, r"E:\gao6gongdan\工单7\研发")

from app.core.conversation import _ANAPHORA, _SUBJECT          # noqa: E402
from app.core.query_rewriter import QueryRewriter, _ORG_RE      # noqa: E402

Q1 = ("平安银行2019年年度报告中，该行2019年实现营业收入和净利润分别是多少？"
      "同比增速分别是多少？年末不良贷款率和拨备覆盖率分别是多少？")
QUESTIONS = [
    "平安银行2019年营业收入是多少？",
    "那净利润呢？",
    "它的拨备覆盖率是多少？",
    "该行的不良贷款率是多少？",
    "那招商银行呢？",
]

#: 模拟真实 corpus_issuers（真实值来自索引块内容自动统计；此处只取本诊断需要的 3 家）
FAKE_ISSUERS = {"corpusA": ["平安银行股份有限公司", "招商银行股份有限公司",
                            "中国邮政储蓄银行股份有限公司"]}


def main() -> int:
    print("=" * 96)
    print("_ORG_RE =", _ORG_RE.pattern)
    print("_SUBJECT =", _SUBJECT.pattern, "｜_ANAPHORA =", _ANAPHORA.pattern)
    print("=" * 96)

    bare = QueryRewriter(corpus_issuers={})
    print("\n[1] 无实体表（只用通用机构名正则）detect_entities：")
    print("   ", bare.detect_entities(Q1))

    rw = QueryRewriter(corpus_issuers=FAKE_ISSUERS)
    print("\n[2] 带实体表（含「平安银行股份有限公司」）detect_entities：")
    for start, end, name in rw.detect_entities(Q1):
        print(f"    span=({start},{end}) text={Q1[start:end]!r} 归一化实体={name!r}")

    ents = rw.detect_entities(Q1)
    print(f"\n[3] focus_entity 取值规则（query_rewriter.py:346）= entities[-1][2] ⇒ "
          f"{ents[-1][2]!r}" if ents else "[3] 无实体")

    print("\n[4] 逐问句实体 / 指代：")
    for q in QUESTIONS:
        print(f"    {q!r}")
        print(f"        entities  = {[(s, e, n) for s, e, n in rw.detect_entities(q)]}")
        print(f"        anaphora  = {[x[2] for x in rw.detect_anaphora(q)]}")

    print("\n[5] QAEngine 单轮指代消解用的主体正则 _SUBJECT（要求「股份有限公司/有限公司」全称）：")
    for q in ("平安银行2019年营业收入是多少？", Q1):
        m = _SUBJECT.search(q)
        print(f"    {q[:34]!r}… ⇒ pending_subject={m.group(0) if m else None!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
