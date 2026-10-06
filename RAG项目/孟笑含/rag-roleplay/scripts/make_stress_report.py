# -*- coding: utf-8 -*-
"""把 stress_report/results/*.jtl 汇总成压测报告 stress_report/压测报告.md。"""
import sys
# 解析：导入路径
from datetime import datetime
# 解析：时间戳
from pathlib import Path
# 解析：路径

ROOT = Path(__file__).resolve().parent.parent
# 解析：项目根
sys.path.insert(0, str(ROOT))
# 解析：加导入路径

from app.stress.jtl import parse_jtl  # noqa: E402
# 解析：JTL 解析器

RESULTS_DIR = ROOT / "stress_report" / "results"
# 解析：结果目录

SCENARIOS = [
    # 解析：场景清单（名称、文件名列表、说明）
    ("S1 注册接口", ["S1_register_10t", "S1_register_50t", "S1_register_100t"], "并发 10/50/100 线程 × 10 循环，真实服务"),
    # 解析：S1 注册
    ("S2 对话接口（MOCK LLM，单实例 8001）", ["S2_chatmock_10t", "S2_chatmock_50t", "S2_chatmock_100t"], "并发 10/50/100 线程 × 10 循环，use_rag=false"),
    # 解析：S2 API 骨架
    ("S3 对话接口（真实 DeepSeek，单实例 8000）", ["S3_chatreal_5t"], "并发 5 线程 × 2 循环，端到端真实大模型"),
    # 解析：S3 端到端
    ("S4 对话接口（MOCK LLM，nginx 反代双实例 8001+8002）", ["S4_nginx_50t", "S4_nginx_100t"], "并发 50/100 线程 × 10 循环，轮询负载均衡"),
    # 解析：S4 负载均衡
]


# 读取单个 jtl 结果文件并解析统计
def load_stats(name: str) -> dict:
    # 解析：读 jtl 统计
    path = RESULTS_DIR / f"{name}.jtl"
    # 解析：文件路径
    if not path.exists():
        # 解析：不存在
        return None
        # 解析：返回 None（报告跳过）
    return parse_jtl(path.read_text(encoding="utf-8"))
    # 解析：解析返回统计


# 汇总 stress_report/results/*.jtl 生成压测报告
def main() -> None:
    # 解析：报告生成主流程
    lines = ["# 压测报告", ""]
    # 解析：标题
    lines.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    # 解析：时间戳
    lines.append("- 工具：Apache JMeter 5.6.3（JDK 17）")
    # 解析：工具
    lines.append("- 被测系统：rag-roleplay FastAPI（uvicorn 单 worker）")
    # 解析：被测系统
    lines.append("- 环境：Windows 11 本机，MySQL/Redis 本机，Milvus Docker（本轮压测不涉及 RAG 检索）")
    # 解析：环境
    lines.append("- 场景说明：S1 纯后端注册；S2 API 骨架（LLM_MOCK=true，use_rag=false）；S3 端到端真实 DeepSeek；S4 nginx 轮询负载均衡两个 uvicorn 实例")
    # 解析：场景说明

    lines += ["", "## 结果总览", ""]
    # 解析：总览段
    lines.append("| 场景 | 并发 | 请求数 | QPS | 平均延迟 | 中位延迟 | P95 | 最大 | 错误率 |")
    # 解析：表头
    lines.append("|---|---|---|---|---|---|---|---|---|")
    # 解析：分隔
    for title, names, _ in SCENARIOS:
        # 解析：逐场景
        for name in names:
            # 解析：逐并发档
            stats = load_stats(name)
            # 解析：读统计
            if stats is None:
                # 解析：无数据
                continue
                # 解析：跳过
            threads = name.rsplit("_", 1)[-1].rstrip("t")
            # 解析：从文件名解析并发数
            lines.append(
                # 解析：一行数据
                f"| {title} | {threads} | {stats['total']} | {stats['qps']:.1f} | "
                # 解析：场景/并发/请求数/QPS
                f"{stats['avg_ms']:.0f}ms | {stats['median_ms']:.0f}ms | {stats['p95_ms']:.0f}ms | "
                # 解析：延迟三值
                f"{stats['max_ms']:.0f}ms | {stats['error_rate']*100:.1f}% |"
                # 解析：最大延迟与错误率
            )

    lines += ["", "## 负载均衡对比（MOCK LLM，同参数）", ""]
    # 解析：对比段
    lines.append("| 并发 | 单实例 QPS | 双实例(nginx) QPS | 吞吐提升 | 单实例延迟 | 双实例延迟 |")
    # 解析：表头
    lines.append("|---|---|---|---|---|---|")
    # 解析：分隔
    for t in ("50", "100"):
        # 解析：两个并发档
        single = load_stats(f"S2_chatmock_{t}t")
        # 解析：单实例数据
        dual = load_stats(f"S4_nginx_{t}t")
        # 解析：双实例数据
        if single and dual:
            # 解析：数据齐全
            lines.append(
                # 解析：一行对比
                f"| {t} | {single['qps']:.1f} | {dual['qps']:.1f} | "
                # 解析：两路 QPS
                f"{(dual['qps']/single['qps']-1)*100:+.0f}% | {single['avg_ms']:.0f}ms | {dual['avg_ms']:.0f}ms |"
                # 解析：提升百分比与延迟
            )

    lines += ["", "## 瓶颈分析与压测发现", ""]
    # 解析：分析段
    lines += [
        # 解析：五项发现（数据来自实测）
        "1. **MySQL 连接池默认配置不足（已修复）**：SQLAlchemy 默认池 size=5/overflow=10，50 并发时出现 "
        # 解析：发现一
        "`QueuePool limit reached`，请求排队 30 秒等连接导致超时。已将 `app/models/db.py` 调整为 "
        # 解析：现象
        "pool_size=20、max_overflow=40、pool_timeout=10，修复后 50/100 并发 0 错误。",
        # 解析：修复
        "2. **单实例吞吐平台约 48 req/s**：100 并发时延迟升至 1.5s。原因是同步 MySQL/Redis 依赖调用走 anyio "
        # 解析：发现二
        "线程池（默认 40 线程），并发超过线程池容量后排队。生产优化方向：改用 async SQLAlchemy/redis.asyncio，"
        # 解析：原因与方向
        "或起多 worker 进程。",
        # 解析：方向
        "3. **端到端延迟由外部大模型主导**：S3 真实 DeepSeek 平均 2.0s（1.0~4.7s），QPS≈1.5（5 并发）。"
        # 解析：发现三
        "在线 API 场景下系统吞吐上限 = 并发数 / LLM 延迟，本地骨架不是瓶颈。",
        # 解析：结论
        "4. **负载均衡收益显著**：nginx 轮询反代两个实例后，100 并发吞吐 +111%（50.4→106.3 req/s）、"
        # 解析：发现四
        "延迟 -68%（1522→494ms）；50 并发吞吐 +33%（57.7→76.9 req/s）、延迟 -29%（395→279ms）。"
        # 解析：数据
        "继续横向扩展实例可继续提升（受 MySQL/Redis 单点制约）。",
        # 解析：边界
        "5. **BGE 模型懒加载竞态（已修复）**：压测暴露 use_rag=false 的请求也会在首次对话时加载 BGE 模型"
        # 解析：发现五
        "（约 23s/2.5GB），并发下曾出现多线程同时加载。已改为 LazyModelProxy 线程安全懒加载，"
        # 解析：现象与修复
        "仅首次真正检索时加载一次。",
        # 解析：修复效果
    ]

    lines += ["", "## 建议", ""]
    # 解析：建议段
    lines += [
        # 解析：建议清单
        "- 生产部署：uvicorn 多 worker（或 nginx 多实例）+ 数据库连接池按并发调优",
        # 解析：部署建议
        "- 若主要流量走真实 LLM：压测重点应放在并发数上限与 API 配额，而非本地 QPS",
        # 解析：测什么
        "- 后续可测：RAG 全链路（use_rag=true，BGE CPU 推理约为每请求 1-3s）",
        # 解析：后续
    ]

    out = ROOT / "stress_report" / "压测报告.md"
    # 解析：输出路径
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # 解析：写报告
    print(f"[已保存] {out}")
    # 解析：提示
    print("\n".join(lines[:40]))
    # 解析：预览前 40 行


if __name__ == "__main__":
    # 解析：入口
    main()
    # 解析：执行
