# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 本脚本职责：把「减少送入重排的候选条数（FUSION_TOP_N）」这条杠杆，按项目自己的评测标准
#             （§3.1：固定 temperature=0、每问 3 轮、两组配对）做一次端到端验证。
#
# 背景：bench_rerank_onnx.py 的深度扫描发现 FUSION_TOP_N 从 60 降到 30 时，
#       重排耗时 0.889s → 0.445s，而「含答案片段仍在最终 top-8」仍是 10/10。
#       但那只是**代理指标**（片段有没有进上下文），不是验收指标（关键事实命中率）。
#       本脚本补上端到端那一步，避免拿代理指标当结论。
#
# 用法：python scripts/bench_fusion_depth_e2e.py --depths 60 30 --rounds 3

import argparse                        # 命令行参数解析
import json                            # 结果落盘
import statistics                      # 取中位数
import sys                             # 调整模块搜索路径
import time                            # 计时
from pathlib import Path               # 路径处理

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # 项目根目录入 sys.path

from app import config as C            # 全局配置（FUSION_TOP_N 从这里读）
from app import eval as E              # 复用官方的流式问答链路与关键事实判定


def run_once(depth: int, rounds: int) -> dict:                         # 跑一个深度配置
    """把 FUSION_TOP_N 设成 depth，跑 rounds 轮 × 全部题目，返回统计。"""
    C.FUSION_TOP_N = depth                                             # 改写配置（rrf_fuse 每次调用时读取）
    qs = E.load_questions()                                            # 题目集
    hits, total, firsts = 0, 0, []                                     # 累计命中 / 总数 / 首字样本
    per_round = []                                                     # 每轮命中
    for r in range(1, rounds + 1):                                     # 逐轮
        rh = 0                                                         # 本轮命中
        for item in qs:                                                # 逐题
            ans, _ctxs, lat = E._rag_stream(item["question"], use_rerank=True)   # 与线上一致的流式链路
            got = E._hits(ans, item.get("expect", []))                 # 关键事实命中
            rh += len(got)                                             # 累计
            hits += len(got)                                           # 累计
            total += len(item.get("expect", []))                       # 累计总数
            firsts.append(lat["first_token"])                          # 首字延迟
        per_round.append(rh)                                           # 记录本轮
        print(f"    深度={depth} 第 {r} 轮：{rh} 项命中")                  # 进度
    return {"depth": depth, "hits": hits, "total": total,              # 汇总
            "per_round": per_round,                                    # 每轮明细
            "first_token_median": round(statistics.median(firsts), 3)}  # 首字中位数


def main() -> int:                                                     # 脚本主函数
    ap = argparse.ArgumentParser(description="FUSION_TOP_N 端到端验证（配对设计）")   # 参数解析器
    ap.add_argument("--depths", type=int, nargs="+", default=[60, 30], help="待比较的送入条数")   # 深度
    ap.add_argument("--rounds", type=int, default=3, help="每个配置跑的轮数")   # 轮数
    ap.add_argument("--out", default="", help="结果落盘路径（默认写到 eval/results）")   # 输出
    args = ap.parse_args()                                             # 解析

    print(f"配对设计：同题目、同轮数、同生成参数（temperature=0），只变 FUSION_TOP_N。"
          f"共 {len(args.depths)} 组 × {args.rounds} 轮 × 10 题")            # 说明协议
    results = []                                                       # 结果集
    for d in args.depths:                                              # 逐配置
        print(f"[配置] FUSION_TOP_N={d}")                                # 进度
        t0 = time.time()                                               # 计时
        res = run_once(d, args.rounds)                                 # 执行
        res["seconds"] = round(time.time() - t0, 1)                    # 记录总耗时
        results.append(res)                                            # 收集
        print(f"  → 命中 {res['hits']}/{res['total']}，首字中位数 {res['first_token_median']}s")   # 小结

    print("\n配对结果")
    print(f"{'FUSION_TOP_N':>14}{'关键事实命中':>16}{'命中率':>10}{'首字中位数':>12}")   # 表头
    base = results[0]                                                  # 以第一组为基线
    for r in results:                                                  # 逐组打印
        rate = r["hits"] / r["total"] if r["total"] else 0             # 命中率
        print(f"{r['depth']:>14}{str(r['hits']) + '/' + str(r['total']):>14}{rate:>10.1%}"
              f"{r['first_token_median']:>11.3f}s")
    print(f"\n（基线 FUSION_TOP_N={base['depth']}；配对设计的差异只来自送入重排的条数）")

    out = Path(args.out) if args.out else (C.EVAL_DIR / "results" /
                                           f"fusion_depth_e2e_{time.strftime('%Y%m%d_%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)                      # 确保目录存在
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")   # 落盘
    print(f"结果已落盘：{out}")                                           # 提示
    return 0                                                           # 成功退出码


if __name__ == "__main__":            # 仅在直接运行时执行
    raise SystemExit(main())          # 用返回值作为退出码
