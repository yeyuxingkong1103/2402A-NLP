# -*- coding: utf-8 -*-
"""人工智能NLP-RAG-功能测试及评估 —— 焦点实体污染根因（读取真实索引的实体表）。

**只读索引与分词，不调用任何生成/嵌入模型**（``load_bundle`` 命中进程内缓存或读本地文件）。
目的：解释为什么多轮回归里第 1 轮问句含「平安银行」却把 ``focus_entity`` 记成了「2019年」。
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, r"E:\gao6gongdan\工单7\研发")

from app.core.config import get_config                       # noqa: E402
from app.core.query_rewriter import QueryRewriter, _short_names  # noqa: E402
from app.core.retriever import load_bundle                    # noqa: E402

Q1 = ("平安银行2019年年度报告中，该行2019年实现营业收入和净利润分别是多少？"
      "同比增速分别是多少？年末不良贷款率和拨备覆盖率分别是多少？")


def main() -> int:
    cfg = get_config()
    bundle = load_bundle(cfg)
    issuers = getattr(bundle, "corpus_issuers", {}) or {}
    print(f"corpus_issuers: {len(issuers)} 个语料键")
    for key, names in issuers.items():
        print(f"  - {key[:52]} ⇒ {json.dumps(list(names), ensure_ascii=False)[:220]}")

    rw = QueryRewriter(corpus_issuers=issuers, cfg=cfg)
    print("\n别名表里含「年」的别名（暴露短名派生的问题）：")
    for full, aliases in list(rw._aliases.items()):
        bad = [a for a in aliases if a.endswith("年") or a.endswith("年度报告")]
        if bad:
            print(f"  full={full!r} ⇒ 可疑简名={json.dumps(bad, ensure_ascii=False)}")

    print(f"\nQ1 问句：{Q1}")
    print("真实实体表下的 detect_entities：")
    for start, end, name in rw.detect_entities(Q1):
        print(f"  span=({start},{end}) 文本={Q1[start:end]!r} ⇒ 归一化实体={name!r}")
    ents = rw.detect_entities(Q1)
    print(f"focus_entity = entities[-1][2] = {ents[-1][2]!r}" if ents else "无实体")
    print(f"\n平安银行股份有限公司 的短名派生：{_short_names('平安银行股份有限公司')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
