# -*- coding: utf-8 -*-
"""
工单：人工智能NLP-RAG-图像内容解析及检索优化
文件：测试/评测骨架/三层命中口径自检.py（T3 骨架：验证三层命中口径的可分性）

用途：用 golden 的真实证据 + 诊断层证据，实测「① 权威层 / ② 诊断层A / ③ 诊断层B」三层在不同 chunk 上的取值，
      作为 T6 报告「检索命中率按通道分列 + 三层口径」的可复现自检。
      只读 golden 与诊断层文件，**不产生任何成绩数字**。

运行：pwsh -NoProfile -File run_py.ps1 测试/评测骨架/三层命中口径自检.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    spec = importlib.util.spec_from_file_location("golden_lib", ROOT / "测试" / "common" / "golden.py")
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)

    rec = G.load_golden_index()[5]
    diag = {5: G.load_diagnostic_evidence()[5]}
    sup = diag[5]["supporting"]

    def show(tag: str, chunks) -> None:
        r = G.check_retrieval_hit_layers(rec, chunks, diagnostic=diag)
        print(f"  {tag:<26s} 权威层={str(r['hit_complete']):<5s} 诊断层A={str(r['hit_any_evidence']):<5s} "
              f"诊断层B={str(r['hit_page_level']):<5s}")

    print("题 id 5：evidence_pages_complete =", rec["evidence_pages_complete"],
          "| evidence_pages =", rec["evidence_pages"],
          "| supporting =", rec["supporting_pages"])
    print("三层口径（① 权威=完整证据原文；② 诊断A=任证证据原文；③ 诊断B=页码 ∈ evidence_pages）：")
    show("页39 chunk（完整证据）", [{"page": 39, "text": rec["retrieval_evidence"][0]["text"]}])
    show("页40 chunk（整句）", [{"page": 40, "text": sup[0]["text"]}])
    show("页113 chunk（整句）", [{"page": 113, "text": sup[1]["text"]}])
    # captain 追踪到的真实索引分块事实：支持性枚举被切到 p40-x1 与 p41-x0 两块（单块不含整句）
    show("p40-x1 块(枚举前半)", [{"page": 40, "text": "销售部下设渠道销售部、电话及网络销售部、大客户销售部"}])
    show("p41-x0 块(枚举后半)", [{"page": 41, "text": "国际贸易部；负责海外市场销售"}])
    show("页113 chunk（内容无关）", [{"page": 113, "text": "（与证据无关的内容）"}])
    show("页38 chunk（页码+内容都无关）", [{"page": 38, "text": "（与证据无关的内容）"}])
    print("片段级说明：诊断层A 对 supporting 证据启用子句片段放宽（≥4 字），"
          "故真实索引里被切开的 p40-x1 / p41-x0 单块也可判为诊断层A 命中。")
    print("说明：三层不可合并——③ 只看页码，② 要求证据原文（或子句片段）真的落在 chunk 里；"
          "命中率主指标只用 ①（见 golden.HIT_LAYER_GUIDE）。")
    print("⚠️ 本自检不产出任何成绩数字；T6 的命中率必须来自 T6 自己的实跑。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
