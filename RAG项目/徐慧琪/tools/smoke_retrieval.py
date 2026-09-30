"""检索链路冒烟集。

存在的理由：本期唯一的验收判据就是"链路通不通"，而链路的三个环节
（dense 路、sparse 路、RRF 融合 + 过滤）任一坏掉都不会抛错，只会安静地
返回不太相关的结果。用一组人工构造的、答案已知的查询去撞，是唯一能在
早期发现问题的办法。

**本冒烟集由 AI 照着法条内容构造，属于"看着答案出题"，只能证明链路通，
不能作为召回率依据。** 需求文档 6.3 要求的 50~100 条律师真实问题评估集
另立任务，两者不可互相替代。

用法：cd D:/xinzg6/fl/backend && python ../tools/smoke_retrieval.py
"""
from __future__ import annotations

import pathlib
import sys

# 冒烟脚本放在 tools/ 下，而 app 包在 backend/ 下，故把 backend 挂进搜寻路径
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

from app.db.milvus import (  # noqa: E402
    COLLECTION, DEFAULT_EF, FILTER_EXPR, entity_count, get_client, hybrid_search,
)
from app.ingest.embed import encode_texts, load_model  # noqa: E402

REPORT_PATH = pathlib.Path(__file__).resolve().parents[1] / "data" / "parsed" / "smoke_report.md"

# 期望条号取自 law_articles.jsonl 的实读值。刻意挑易混对：
# 语义相近但条款号不同——纯语义容易混、纯关键词抓不住改写，正好考验双路融合
QUERIES = [
    {"topic": "善意取得", "query": "买了别人无权处分的二手车，能取得所有权吗",
     "expect_articles": [311]},
    {"topic": "无权代理", "query": "没有代理权的人以我的名义签合同有效吗",
     "expect_articles": [171]},
    {"topic": "悬赏广告", "query": "悬赏寻找失物，找到后对方不给赏金怎么办",
     "expect_articles": [499]},
    {"topic": "买卖合同", "query": "卖方交付的货物质量不符合约定要承担什么责任",
     "expect_articles": [617]},
    {"topic": "借款利息", "query": "民间借贷没有约定利息，还能要利息吗",
     "expect_articles": [680]},
    {"topic": "保证责任", "query": "保证人被债权人起诉，保证期间是多久",
     "expect_articles": [692]},
    {"topic": "租赁押金", "query": "租房到期押金不退怎么办",
     "expect_articles": [733]},
    {"topic": "建设工程", "query": "建设工程未经竣工验收就使用，质量问题谁负责",
     "expect_articles": [799]},
    {"topic": "夫妻共同债务", "query": "一方在外面借的钱，另一方要一起还吗",
     "expect_articles": [1064]},
    {"topic": "离婚财产", "query": "离婚时房子怎么分割",
     "expect_articles": [1087]},
    {"topic": "法定继承", "query": "没有遗嘱的情况下遗产按什么顺序继承",
     "expect_articles": [1127]},
    {"topic": "遗嘱形式", "query": "打印出来签字的遗嘱有效吗",
     "expect_articles": [1136]},
    {"topic": "高空抛物", "query": "楼上掉东西砸伤人找不到是谁扔的怎么办",
     "expect_articles": [1254]},
    {"topic": "精神损害赔偿", "query": "被侵权受到严重精神痛苦能要求赔偿吗",
     "expect_articles": [1183]},
    {"topic": "诉讼时效", "query": "欠钱不还过了三年还能起诉吗",
     "expect_articles": [188]},
    {"topic": "担保物权", "query": "抵押的房屋可以转让吗",
     "expect_articles": [406]},
    {"topic": "不当得利", "query": "银行转账转错人，对方不还怎么办",
     "expect_articles": [985]},
    {"topic": "无因管理", "query": "替别人垫付了费用能不能要回来",
     "expect_articles": [979]},
]


def run_queries(model, client, top_k: int = 5) -> list[dict]:
    """跑全部查询，每条同时记录 hybrid 与前 5 条。

    额外单独跑一次 dense-only 的 top1，是为了能判断稀疏路有没有真的生效：
    只跑 hybrid 的话，稀疏路坏掉也看不出来——融合结果由 dense 主导时照样能
    返回看起来合理的东西。两路 top1 完全一致就是可疑信号。
    """
    results = []
    for q in QUERIES:
        dense, sparse = encode_texts(model, [q["query"]])[0]
        hits = hybrid_search(client, dense, sparse, top_k=top_k)
        # 条数是调用方唯一的预算闸门：检索超量返回不会报错，只会让下游 prompt 悄悄膨胀，
        # 而本项目无版本控制，事后没有任何东西能回溯。Task 6 审查发现 top_k 零覆盖
        # （变异 limit=top_k→10 后 9 条测试仍全绿），这里当场钉死
        if len(hits) != top_k:
            raise AssertionError(f"{q['topic']}：期望 {top_k} 条，实得 {len(hits)} 条")
        # 直接调 client.search 而不是给 hybrid_search 加开关：dense-only 只是
        # 冒烟期的诊断手段，不该把诊断参数塞进生产检索接口
        only = client.search(COLLECTION, data=[dense], anns_field="dense", limit=1,
                             search_params={"metric_type": "COSINE",
                                            "params": {"ef": DEFAULT_EF}},
                             filter=FILTER_EXPR, output_fields=["article_no"])
        dense_top1 = only[0][0]["entity"]["article_no"] if only and only[0] else None
        results.append({**q, "hits": hits, "dense_only_top1": dense_top1})
    return results


def render_report(results: list[dict]) -> str:
    """渲染 markdown 报告。

    报告必须写明它不是基准——这句话是防止后人把冒烟结果当召回率引用。
    """
    lines = [
        "# 检索链路冒烟报告",
        "",
        "> **这不是评估集，不代表召回率。** 本冒烟集的查询由 AI 照着法条内容构造，",
        "> 属于「看着答案出题」，仅用于确认链路能跑通（双路都活、RRF 融合有序、",
        "> 父块过滤生效）。需求文档 6.3 要求的律师真实问题评估集另立任务。",
        "",
        f"- 查询数：{len(results)}",
        "",
    ]
    for r in results:
        lines.append(f"## {r['topic']}")
        lines.append("")
        lines.append(f"- 查询：{r['query']}")
        lines.append(f"- 期望条号：{r.get('expect_articles') or '（不验答案）'}")
        # 两路 top1 并列，是为了让"稀疏路有没有生效"一眼可见
        if r.get("dense_only_top1") is not None:
            hybrid_top1 = r["hits"][0]["article_no"] if r["hits"] else None
            lines.append(f"- dense-only top1：第 {r['dense_only_top1']} 条；"
                         f"hybrid top1：第 {hybrid_top1} 条")
        lines.append("")
        if not r["hits"]:
            lines.append("- **无命中**")
            lines.append("")
            continue
        for i, h in enumerate(r["hits"], start=1):
            hit = h["article_no"] in (r.get("expect_articles") or [])
            mark = "✅" if hit else "  "
            text = h["text"][:60].replace("\n", " ")
            lines.append(f"{i}. {mark} 第 {h['article_no']} 条（{h['chunk_type']}）{text}…")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    client = get_client()
    stats = entity_count(client)
    if stats != 3388:
        print(f"集合实体数 {stats} ≠ 3388，需先跑 app.ingest.embed 灌满")
        return 1
    model = load_model()
    results = run_queries(model, client)
    REPORT_PATH.write_text(render_report(results), encoding="utf-8")
    hit_top1 = sum(1 for r in results
                   if r["hits"] and r["hits"][0]["article_no"] in (r["expect_articles"] or []))
    # 融合若从未改变 top1，说明稀疏路没起作用——这是本期最该盯的失败模式
    diverged = sum(1 for r in results
                   if r["hits"] and r.get("dense_only_top1") is not None
                   and r["hits"][0]["article_no"] != r["dense_only_top1"])
    print(f"冒烟完成：{len(results)} 条查询，top1 命中 {hit_top1} 条")
    print(f"融合改变 top1 的查询数：{diverged}/{len(results)}（为 0 说明稀疏路可能没生效）")
    print(f"报告：{REPORT_PATH}")
    print("提醒：这不是评估集，命中数不代表召回率")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
