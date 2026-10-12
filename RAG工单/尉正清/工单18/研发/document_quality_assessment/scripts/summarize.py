# 工单编号：人工智能NLP-RAG-实现文档质量评估Skill并集成至智能体工作流
"""把解析记录汇总成报告 —— 五大功能的「合流处」

`assess.py` 负责**跑**（遍历、解析、缓存、断点续跑），这个模块负责**算**
（把缓存里的一堆事实汇总成一份报告）。分开的理由有两条：

1. `assess.py` 要控制在 300 行以内（项目约定），合在一起会超；
2. 汇总这一步**不读任何文件正文**，纯内存计算，所以可以随便重跑 ——
   调阈值不用重新解析 1700 份 PDF，靠的就是这个边界。

调用方只需要 `build_report(root, records, cfg)`。
"""
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))   # 允许被单独 import（API 端点只调汇总这一步）

import collect      # noqa: E402
import common       # noqa: E402
import dedup        # noqa: E402
import pdf_type     # noqa: E402
import report       # noqa: E402
import sensitive    # noqa: E402
import tags as tagmod   # noqa: E402

def build_report(root, records, cfg):
    """把解析记录汇总成报告。**这一步不读任何文件正文**，纯算，所以快。"""
    paths = [Path(r["path"]) for r in records]

    # 1) 格式分布
    fmt = collect.format_stats(paths)

    # 2) PDF 文字型 / 扫描型分流
    pdf_rows = [(r["path"], pdf_type.classify(r["path"], cfg, r.get("pages"),
                                              r.get("covers"),
                                              r.get("total_pages") or 0))
                for r in records if r["ext"] == ".pdf"]
    pdf_sum = pdf_type.summarize(pdf_rows)
    pdf_by_path = dict(pdf_rows)

    # 3) 长度分布
    length = collect.length_stats(records, cfg)

    # 4) 重复检测：MD5 精确 + SimHash 近似
    md5_dup = dedup.md5_groups(paths, common.md5_of)
    min_chars = cfg["dedup"].get("simhash_min_chars", 0)
    sim_items, thin = [], 0
    for r in records:
        # 没算出指纹（无正文）和正文太短，都算「没参与比较」——
        # 两种情况都要计数，否则「0 对冲突」看着像查过了，其实根本没查
        if r.get("simhash") is None or (r.get("chars") or 0) < min_chars:
            thin += 1
            continue
        sim_items.append((r["path"], r["simhash"]))
    sim_pairs, sim_note = dedup.simhash_pairs_from_hashes(sim_items, cfg)
    if thin:
        why = (f"{thin} 份文档正文不足 {min_chars} 字符或没有文字层，未参与 SimHash 比较"
               f"（仅 {len(sim_items)} 份参与）；这批文档的版本冲突需先 OCR 出正文再比。")
        sim_note = f"{why} {sim_note}".strip()
    dup_sum = dedup.summarize(md5_dup, sim_pairs, len(records))
    dup_sum["simhash_compared"] = len(sim_items)
    dup_sum["simhash_skipped_thin"] = thin

    # 5) 敏感信息
    all_hits = [h for r in records for h in r.get("hits", [])]
    sens_sum = sensitive.summarize(all_hits, cfg)

    # 6) 分类标签（工作流决策节点读 route 决定送哪个解析器）
    tagged = []
    for r in records:
        facts = {"ext": r["ext"], "chars": r.get("chars"),
                 "pdf_kind": (pdf_by_path.get(r["path"]) or {}).get("kind")}
        tagged.append((r["path"], tagmod.classify(facts, cfg)))
    tag_sum = tagmod.summarize(tagged)
    tag_routes = {}
    for _, ts in tagged:
        for t in ts:
            tag_routes.setdefault(t["name"], set()).add(t["route"])
    tag_routes = {k: sorted(v) for k, v in tag_routes.items()}

    uncertain = [[p, pr["kind"], pr["scanned_ratio"], pr["reason"]]
                 for p, pr in pdf_rows if pr.get("uncertain")]

    summary = {
        "root": str(root),
        "total_files": len(records),
        "total_bytes": sum(r.get("size") or 0 for r in records),
        "parse_failed": length["failed_count"],
        "ext": fmt["by_ext"],
        "pdf": pdf_sum["counts"],
        "pdf_scanned_ratio": pdf_sum["ratios"].get("scanned", 0.0),
        "chars": {"total": length["total_chars"], "mean": length["mean"],
                  **length["percentiles"]},
        "exact_dup_groups": dup_sum["exact_dup_groups"],
        "redundant_copies": dup_sum["redundant_copies"],
        "similar_pairs": dup_sum["similar_pairs"],
        "simhash_compared": dup_sum["simhash_compared"],
        "simhash_skipped_thin": dup_sum["simhash_skipped_thin"],
        "sensitive_hits": sens_sum["total_hits"],
        "uncertain_pdf": len(uncertain),
        "unreadable_pdf": pdf_sum["unreadable_count"],
        "by_tag": tag_sum["by_tag"],
        "by_route": tag_sum["by_route"],
        "by_primary_route": tag_sum["by_primary"],   # 这份才是「多少份送去哪儿」
    }

    details = {
        "format": fmt,
        "pdf": {"counts": pdf_sum["counts"], "ratios": pdf_sum["ratios"],
                "total": len(pdf_rows), "unreadable": pdf_sum["_unreadable"],
                "uncertain": uncertain},
        "length": length,
        "dup": {
            "summary": dup_sum,
            # 每条都带路径 —— 验收要求「列表可操作性」
            "md5_groups": [[h[:8], len(ps), " ｜ ".join(ps)] for h, ps in md5_dup.items()],
            "sim_pairs": [[pr["a"], pr["b"], f"{pr['similarity']:.0%}", pr["distance"]]
                          for pr in (sim_pairs or [])],
            "note": sim_note,
        },
        "sensitive": {
            "summary": sens_sum,
            "hits": [[h["source"], h["type"], h["value"], h["context"]] for h in all_hits],
        },
        "tags": {"by_tag": tag_sum["by_tag"], "by_route": tag_sum["by_route"],
                 "by_primary": tag_sum["by_primary"], "tag_routes": tag_routes,
                 # 第三个字段是**主路由** = 第一条命中的标签的 route。
                 # 规则按「最确定→最兜底」排序，所以 tag[0] 就是最该听的那个；
                 # 工作流决策节点只读这一列，不然一份文档带着两个路由它没法选。
                 "docs": [[p, "、".join(t["name"] for t in ts), ts[0]["route"]]
                          for p, ts in tagged]},
        "errors": [[r["path"], r["error"]] for r in records if r.get("error")],
    }
    return report.build(summary, details, cfg)
