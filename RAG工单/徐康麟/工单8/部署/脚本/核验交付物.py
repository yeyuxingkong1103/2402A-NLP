# -*- coding: utf-8 -*-
"""工单8 交付物核验：清单齐全性 + 源码工单编号注释 + 真实数据可追溯 + 禁用写法自检。

工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答

核验项（不合格即退出码非 0，可直接用于 CI/交付前自检）：
    1. 五大目录与关键交付物是否齐全（设计 / 研发 / 测试 / 优化 / 部署）；
    2. **所有源码文件头部是否含工单编号注释**（`人工智能NLP-RAG-基于Graph RAG 实现金融问答`）；
    3. **禁用写法自检**：`multiprocessing` / `ProcessPoolExecutor` / `Pool(` / `joblib` / `n_jobs`
       （`_共享约定.md` §5.5 事故后禁令）；
    4. 关键数字是否有产物文件支撑（图谱统计 / 评估汇总 / 对比表 / 截图）；
    5. 缓存目录是否已清理（`__pycache__` / `.pytest_cache`）；
    6. 外链自检：图谱 SVG/HTML 是否自包含（无 http(s) 外链）。

用法：
    pwsh -NoProfile -File run_py.ps1 部署/脚本/核验交付物.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "研发") not in sys.path:
    sys.path.insert(0, str(_ROOT / "研发"))

#: 工单编号（源码头部注释必须含此串）
WORK_ORDER = "人工智能NLP-RAG-基于Graph RAG 实现金融问答"

#: 源码扫描范围（相对仓库根）
SOURCE_DIRS = ("研发/app/graphrag", "研发/scripts", "研发/评估", "测试", "优化/脚本", "部署/脚本")
#: 需要豁免注释检查的文件（第三方/复用文件不强制）
EXEMPT_PREFIXES = ("研发/app/core/", "研发/app/storage/", "研发/app/prompts/", "研发/app/ui/serve_fallback.py",
                   "研发/app/main.py", "研发/app/chat_cli.py", "研发/app/multiturn_engine.py",
                   "研发/app/__init__.py", "研发/scripts/build_index", "研发/scripts/parse_corpus",
                   "研发/scripts/extract_images", "研发/scripts/download_clip")
#: 禁用写法自检的**自我豁免**：本核验脚本自身需要写出这些关键词（BANNED 常量 + 说明文字），
#: 否则会自证违规（实测第一次运行就报了自己）。
BANNED_SCAN_EXEMPT = ("部署/脚本/核验交付物.py",)

#: 交付物清单：(路径, 说明, 是否必需)
DELIVERABLES: list[tuple[str, str, bool]] = [
    # 设计
    ("设计/工单需求原文.md", "需求原文存档", True),
    ("设计/需求分析.md", "需求分析与设计取舍", True),
    ("设计/GraphRAG系统架构.md", "系统架构", True),
    ("设计/知识图谱Schema设计.md", "实体/关系类型表", True),
    ("设计/接口设计.md", "Python/CLI/HTTP 接口", True),
    ("设计/技术选型.md", "技术选型与论证", True),
    ("设计/环境事实.md", "实测环境事实", True),
    ("设计/验收标准.md", "验收标准与结果", True),
    ("设计/README.md", "设计文档索引", True),
    # 研发
    ("研发/app/graphrag/config.py", "配置与阈值（配置化）", True),
    ("研发/app/graphrag/schema.py", "图谱 Schema（唯一真源）", True),
    ("研发/app/graphrag/extractor.py", "规则/词典/正则抽取器", True),
    ("研发/app/graphrag/store.py", "图谱存储与邻域扩展", True),
    ("研发/app/graphrag/engine.py", "Graph RAG 检索与问答引擎", True),
    ("研发/app/graphrag/llm.py", "Ollama 客户端", True),
    ("研发/app/graphrag/visualize.py", "图谱可视化（PIL/SVG/HTML）", True),
    ("研发/app/ui/serve_graph_web.py", "纯标准库 Web 界面（8124）", True),
    ("研发/scripts/build_graph.py", "图谱构建脚本", True),
    ("研发/scripts/render_graph_viz.py", "图谱可视化脚本", True),
    ("研发/scripts/build_comparison.py", "与工单7 对比脚本", True),
    ("研发/评估/run_eval.py", "17 题评估脚本", True),
    ("研发/评估/判分口径.py", "判分口径（复用工单7）", True),
    ("研发/data/graph/entities.jsonl", "实体表", True),
    ("研发/data/graph/relations.jsonl", "关系表", True),
    ("研发/data/graph/graph.json", "图谱单文件", True),
    ("研发/data/graph/graph.graphml", "GraphML 导出", True),
    ("研发/data/graph/graph.svg", "自包含 SVG", True),
    ("研发/data/graph/graph.html", "自包含 HTML", True),
    ("研发/data/graph/图谱构建统计.json", "构建实测数字", True),
    ("研发/data/raw", "9 份年报 PDF（语料存档）", True),
    ("研发/requirements.txt", "依赖清单", True),
    ("研发/README.md", "研发说明", True),
    # 测试
    ("测试/test_图谱schema.py", "Schema 校验用例", True),
    ("测试/test_检索回归.py", "检索回归用例", True),
    ("测试/test_多语言.py", "多语言用例", True),
    ("测试/test_图谱可视化.py", "可视化用例", True),
    ("测试/测试数据/GraphRAG评估结果.jsonl", "逐题评估明细", True),
    ("测试/测试数据/GraphRAG评估汇总.json", "评估汇总", True),
    ("测试/测试数据/与工单7对比.json", "并排对比表", True),
    ("测试/测试数据/与工单7对比.csv", "逐题对比 CSV", True),
    ("测试/测试报告.md", "测试报告", True),
    ("测试/测试截图", "中文命名 PNG 截图目录", True),
    # 优化
    ("优化/图谱抽取优化说明.md", "抽取规则迭代说明", True),
    ("优化/评估结果/优化前后对比.json", "优化前后对比数据", True),
    # 部署
    ("部署/脚本/一键跑通.ps1", "一键跑通全流程", True),
    ("部署/脚本/启动界面.ps1", "启动 Web 界面（8124）", True),
    ("部署/README.md", "部署说明", True),
    ("部署/交付物核验.md", "交付物核验报告", True),
]

#: 截图清单（中文按测试内容命名，硬性要求）
SHOTS: list[tuple[str, str]] = [
    ("测试/测试截图/知识图谱可视化.png", "知识图谱可视化（≥120 节点 + 关系 + 图例）"),
    ("测试/测试截图/实体关系抽取统计.png", "实体/关系抽取统计"),
    ("测试/测试截图/GraphRAG与工单7检索结果对比.png", "与工单7 检索结果对比"),
    ("测试/测试截图/17题逐题评分.png", "17 题逐题评分"),
    ("测试/测试截图/响应时间对比.png", "响应时间对比"),
]

BANNED = ("multiprocessing", "ProcessPoolExecutor", "concurrent.futures.ProcessPool", "joblib",
          "n_jobs=-1", "n_jobs = -1", "Pool(")


def check_deliverables(root: Path) -> tuple[list[str], list[str]]:
    ok: list[str] = []
    bad: list[str] = []
    for rel, desc, required in DELIVERABLES:
        path = root / rel
        exists = path.exists()
        if exists and path.is_file() and path.stat().st_size == 0:
            exists = False
        if exists:
            ok.append(f"{'✅' if required else '➖'} {rel} —— {desc}")
        else:
            bad.append(f"{'❌' if required else '⚠️'} {rel} —— {desc}（缺失）")
    return ok, bad


def check_work_order_comment(root: Path) -> tuple[list[str], list[str]]:
    ok: list[str] = []
    bad: list[str] = []
    for rel_dir in SOURCE_DIRS:
        directory = root / rel_dir
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*.py")):
            rel = path.relative_to(root).as_posix()
            if any(rel.startswith(prefix) for prefix in EXEMPT_PREFIXES):
                continue
            try:
                head = path.read_text(encoding="utf-8", errors="replace")[:1200]
            except OSError as exc:
                bad.append(f"❌ {rel} —— 读取失败：{type(exc).__name__}: {exc}")
                continue
            if WORK_ORDER in head:
                ok.append(f"✅ {rel}")
            else:
                bad.append(f"❌ {rel} —— 头部未包含工单编号注释")
    return ok, bad


def check_banned(root: Path) -> tuple[list[str], list[str]]:
    ok: list[str] = []
    bad: list[str] = []
    for rel_dir in SOURCE_DIRS:
        directory = root / rel_dir
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*.py")):
            rel = path.relative_to(root).as_posix()
            if rel in BANNED_SCAN_EXEMPT:          # 核验脚本自身豁免（见常量注释）
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            hits = [token for token in BANNED if token in text]
            # 允许「注释里说明禁令」的写法：出现 banned 词但同行的行首是 # 或属于文档字符串则不算违规
            real: list[str] = []
            for token in hits:
                for line in text.splitlines():
                    if token in line and not line.strip().startswith(("#", "*", '"', "'")):
                        real.append(f"{token}@「{line.strip()[:70]}」")
                        break
            if real:
                bad.append(f"❌ {rel} —— 命中禁用写法：{'; '.join(real)}")
            else:
                ok.append(f"✅ {rel}（无进程池写法）")
    return ok, bad


def check_shots(root: Path) -> tuple[list[str], list[str]]:
    ok: list[str] = []
    bad: list[str] = []
    for rel, desc in SHOTS:
        path = root / rel
        if path.is_file() and path.stat().st_size > 10_000:
            ok.append(f"✅ {rel}（{path.stat().st_size // 1024} KB）—— {desc}")
        else:
            bad.append(f"❌ {rel} —— {desc}（缺失或过小）")
    extra = [p.name for p in (root / "测试" / "测试截图").glob("*.png")
             if f"测试/测试截图/{p.name}" not in {s[0] for s in SHOTS}]
    for name in extra:
        if not re.search(r"[\u4e00-\u9fff]", name):
            bad.append(f"❌ 测试/测试截图/{name} —— 截图文件名必须中文按测试内容命名")
        else:
            ok.append(f"✅ 测试/测试截图/{name}（额外截图，命名合规）")
    return ok, bad


def check_real_data(root: Path) -> tuple[list[str], list[str]]:
    ok: list[str] = []
    bad: list[str] = []
    stats_path = root / "研发" / "data" / "graph" / "图谱构建统计.json"
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        for key in ("entities", "relations", "mentions", "chunks_scanned", "elapsed_ms"):
            if key in stats:
                ok.append(f"✅ 图谱统计 {key} = {stats[key]}")
            else:
                bad.append(f"❌ 图谱统计缺少字段 {key}")
    except Exception as exc:  # noqa: BLE001
        bad.append(f"❌ 图谱构建统计不可读：{type(exc).__name__}: {exc}")
    summary_path = root / "测试" / "测试数据" / "GraphRAG评估汇总.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        ok.append(f"✅ 评估有效题数 = {summary.get('有效题数')}，异常题数 = {summary.get('异常题数')}")
        ok.append(f"✅ 严格准确率 = {summary.get('准确率', {}).get('严格准确率')}")
        ok.append(f"✅ 平均 recall@5 = {summary.get('检索召回', {}).get('平均recall@5')}")
    except Exception as exc:  # noqa: BLE001
        bad.append(f"❌ 评估汇总不可读：{type(exc).__name__}: {exc}")
    for rel in ("测试/测试数据/与工单7对比.json", "研发/data/graph/可视化统计.json"):
        path = root / rel
        if path.is_file():
            ok.append(f"✅ {rel}（{path.stat().st_size // 1024} KB）")
        else:
            bad.append(f"❌ {rel} 缺失")
    # 自包含检查
    for rel in ("研发/data/graph/graph.html", "研发/data/graph/graph.svg"):
        path = root / rel
        if not path.is_file():
            bad.append(f"❌ {rel} 缺失，无法检查外链")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        refs = text.count("http://") + text.count("https://") - text.count("http://www.w3.org/2000/svg")
        if refs > 0:
            bad.append(f"❌ {rel} —— 存在 {refs} 处外链（要求自包含）")
        else:
            ok.append(f"✅ {rel} 自包含（0 外链）")
    return ok, bad


def check_caches(root: Path) -> tuple[list[str], list[str]]:
    ok: list[str] = []
    bad: list[str] = []
    for pattern in ("__pycache__", ".pytest_cache"):
        found = [p for p in root.rglob(pattern) if p.is_dir()]
        if found:
            bad.append(f"❌ 发现 {len(found)} 个 {pattern} 目录（需清理）："
                       + "、".join(str(p.relative_to(root)) for p in found[:5]))
        else:
            ok.append(f"✅ 无 {pattern} 目录")
    return ok, bad


def main(argv: list[str] | None = None) -> int:
    root = _ROOT
    sections: list[tuple[str, tuple[list[str], list[str]]]] = [
        ("1. 交付物清单", check_deliverables(root)),
        ("2. 源码工单编号注释", check_work_order_comment(root)),
        ("3. 禁用写法自检（进程池）", check_banned(root)),
        ("4. 测试截图（中文命名）", check_shots(root)),
        ("5. 真实数据可追溯 + 自包含", check_real_data(root)),
        ("6. 缓存目录清理", check_caches(root)),
    ]
    total_ok = total_bad = 0
    lines = [f"# 工单8 交付物核验报告", "",
             f"> 工单编号：{WORK_ORDER}", f"> 核验方式：`python 部署/脚本/核验交付物.py`（自动生成，勿手改）", ""]
    print(f"# 工单8 交付物核验（{WORK_ORDER}）")
    for title, (ok, bad) in sections:
        total_ok += len(ok)
        total_bad += len(bad)
        lines += [f"## {title}", "", f"通过 {len(ok)} 项，未通过 {len(bad)} 项", ""]
        for item in ok:
            lines.append(f"- {item}")
        for item in bad:
            lines.append(f"- {item}")
        lines.append("")
        print(f"\n== {title} == 通过 {len(ok)} / 未通过 {len(bad)}")
        for item in bad:
            print("   " + item)
    verdict = "✅ 全部通过" if total_bad == 0 else f"❌ 有 {total_bad} 项未通过"
    lines += ["## 结论", "", f"- 通过 **{total_ok}** 项，未通过 **{total_bad}** 项 —— {verdict}", ""]
    out = root / "部署" / "交付物核验.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n结论：{verdict}（通过 {total_ok} / 未通过 {total_bad}）")
    print(f"报告已写入：{out}")
    return 0 if total_bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
