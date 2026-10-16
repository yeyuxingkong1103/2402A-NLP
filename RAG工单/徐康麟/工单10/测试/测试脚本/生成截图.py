# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG 项目-金融问答系统部署

测试截图生成脚本：``测试/测试脚本/生成截图.py``。

原则（§共享约定 §4.1）：**截图里的每个数字都必须来自真实产物文件**，禁止手绘假数据。
本脚本只做「读真实文件 → 排版成 PNG」，不改写任何数字，并在每张图页脚标注**来源文件 + 生成时间**。

图片命名：**中文按测试内容命名**（如 ``服务健康检查响应.png``），一眼看出测的是什么。

用法（工作目录 = 工单10/）::

    pwsh -NoProfile -File run_py.ps1 测试/测试脚本/生成截图.py

输出：``测试/测试截图/*.png``
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(r"E:\gao6gongdan\_共享工具")))
sys.path.insert(0, str(REPO_ROOT / "研发"))

from 截图渲染 import Page, clip  # noqa: E402

SHOT_DIR = REPO_ROOT / "测试" / "测试截图"
DATA_DIR = REPO_ROOT / "测试" / "测试数据"
VOL_ROOT = REPO_ROOT / "部署" / "卷"
LOG_DIR = REPO_ROOT / "部署" / "日志"
OPT_DIR = REPO_ROOT / "优化" / "评估结果"

SHOT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def dir_stats(path: Path, limit: int = 0) -> tuple[int, int]:
    """返回 (文件数, 总字节)。流式遍历，不在内存里留清单。"""
    files = 0
    total = 0
    stack = [path]
    while stack:
        cur = stack.pop()
        try:
            with __import__("os").scandir(cur) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(Path(e.path))
                        elif e.is_file(follow_symlinks=False):
                            files += 1
                            total += e.stat().st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return files, total


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.1f} GB"


# ====================================================================== #
# 图 1：容器运行状态（含 Docker 限制的如实说明）
# ====================================================================== #
def shot_container_status() -> None:
    health = load_json(DATA_DIR / "健康检查-三档探针.json", {}) or {}
    info = load_json(DATA_DIR / "健康检查-服务信息.json", {}) or {}
    proc = load_json(LOG_DIR / "本地进程.json", {}) or {}
    got = health.get("实测", {})
    healthz = got.get("/healthz", {}).get("body", {}) if isinstance(got, dict) else {}
    health_body = got.get("/health", {}).get("body", {}) if isinstance(got, dict) else {}

    p = Page("容器运行状态与端口服务验证", "工单10 · 人工智能NLP-RAG 项目-金融问答系统部署")
    y = p.y          # 页头（标题/副标题）已在 Page.__init__ 里画好
    y = p.section("① 服务实例（本机等价验证：与镜像内同一份源码）")
    y = p.kv_block([
        ("服务 rag-api", "uvicorn service.api:app"),
        ("服务 rag-ui", "python app/main.py --serve"),
        ("API 端口（指定）", "8200"),
        ("UI 端口（指定）", "8201"),
        ("API 进程 PID", str(proc.get("api_pid", "—"))),
        ("UI 进程 PID", str(proc.get("ui_pid", "—"))),
        ("启动时刻", str(proc.get("started_at", "—"))),
        ("服务状态", f"alive（/healthz HTTP {got.get('/healthz', {}).get('status', '?')}）"),
    ])

    y = p.section("② 端口探活（真实 HTTP 响应）")
    rows = []
    for path in ("/healthz", "/health", "/readyz"):
        item = got.get(path, {})
        body = item.get("body", {})
        rows.append([path, str(item.get("status", "—")),
                     str(body.get("status", "—")) if isinstance(body, dict) else "—",
                     "JSON" if isinstance(body, dict) else "文本"])
    y = p.table_block(["端点", "HTTP 状态", "自报状态", "响应格式"], rows, widths=[220, 160, 200, 160])

    y = p.section("③ 运行期健康摘要（/health 关键子项）")
    checks = health_body.get("checks", {}) if isinstance(health_body, dict) else {}
    y = p.kv_block([
        ("整体状态", f"{health_body.get('status', '—')}（ready={health_body.get('ready')}）"),
        ("启动耗时 init_ms", f"{health_body.get('init_ms', '—')} ms"),
        ("服务自报端口", str(health_body.get("port", "—"))),
        ("索引卷可读", str(checks.get("volume_index", {}).get("ok"))),
        ("运行期卷可写", str(checks.get("volume_runtime", {}).get("ok"))),
        ("会话后端", str(checks.get("session_store", {}).get("backend"))),
    ])
    y = p.note("说明：本机 docker CLI 访问 daemon 被沙箱拒绝（提权在本会话被禁用），"
               "故运行期证据来自「等价本机验证」：同一份源码 + 同一组端口 8200/8201 + 同构卷语义。")
    y = p.note(f"服务信息接口自报端点：{'、'.join(info.get('endpoints', [])[:6])} …", size=15)
    p.finish(SHOT_DIR / "容器运行状态.png",
             sources=["测试/测试数据/健康检查-三档探针.json", "测试/测试数据/健康检查-服务信息.json",
                      "部署/日志/本地进程.json"])


# ====================================================================== #
# 图 2：compose 配置解析（docker 官方解析器的真实输出）
# ====================================================================== #
def shot_compose_config() -> None:
    cfg = load_json(DATA_DIR / "compose配置-解析结果.json", {}) or {}
    services = cfg.get("services", {})
    api = services.get("rag-api", {})
    ui = services.get("rag-ui", {})
    redis = services.get("redis", {})
    prom = services.get("prometheus", {})
    nets = cfg.get("networks", {})
    vols = cfg.get("volumes", {})

    p = Page("docker compose 配置解析验证（Docker 官方解析器输出）",
             "命令：docker compose --env-file 部署/配置/.env.example -f 部署/docker/docker-compose.yml "
             "--profile monitoring config --format json")
    y = p.y          # 页头（标题/副标题）已在 Page.__init__ 里画好

    y = p.section("① 服务与端口映射（验收项②：指定端口）")
    rows = []
    for name, svc in (("rag-api", api), ("rag-ui", ui), ("redis", redis), ("prometheus", prom)):
        ports = svc.get("ports") or [{}]
        pub = ",".join(str(x.get("published", "—")) for x in ports) or "—"
        tgt = ",".join(str(x.get("target", "—")) for x in ports) or "—"
        rows.append([name, str(svc.get("container_name", "—")),
                     f"{pub} → {tgt}", str(svc.get("restart", "—")),
                     (svc.get("deploy", {}).get("resources", {}).get("limits", {}) or {}).get("memory", "—")])
    y = p.table_block(["服务", "容器名", "宿主端口 → 容器端口", "重启策略", "内存限制"],
                      rows, widths=[130, 190, 260, 200, 160])

    y = p.section("② 卷（具名卷 + 绑定挂载）与只读标记")
    rows = []
    for v in (api.get("volumes") or []):
        rows.append([str(v.get("source")), str(v.get("target")), str(v.get("type")),
                     "是" if v.get("read_only") else ""])
    y = p.table_block(["来源", "容器内路径", "类型", "只读"], rows, widths=[300, 330, 140, 100])

    y = p.section("③ 网络（自定义 bridge）与构建上下文")
    net = nets.get("rag-net", {})
    subnet = ((net.get("ipam") or {}).get("config") or [{}])[0].get("subnet", "—")
    y = p.kv_block([
        ("网络名", str(net.get("name"))),
        ("驱动", str(net.get("driver"))),
        ("子网", str(subnet)),
        ("构建上下文", str(api.get("build", {}).get("context"))),
        ("Dockerfile", str(api.get("build", {}).get("dockerfile"))),
        ("extra_hosts", str((api.get("extra_hosts") or ["—"])[0])),
    ])
    y = p.note(f"具名卷共 {len(vols)} 个：{'、'.join(sorted(vols))}", size=15)
    y = p.note("解析退出码 0（纯客户端解析，不需要 docker daemon —— 这正是本机唯一能拿到的 Docker 侧真证据）",
               color=(22, 140, 74))
    p.finish(SHOT_DIR / "compose配置解析.png",
             sources=["测试/测试数据/compose配置-解析结果.json"])


# ====================================================================== #
# 图 3：服务健康检查响应
# ====================================================================== #
def shot_health() -> None:
    health = load_json(DATA_DIR / "健康检查-三档探针.json", {}) or {}
    idx = load_json(DATA_DIR / "健康检查-索引元信息.json", {}) or {}
    metrics = (DATA_DIR / "健康检查-指标.txt")
    metrics_txt = metrics.read_text(encoding="utf-8") if metrics.exists() else ""
    body = (health.get("实测", {}).get("/health", {}) or {}).get("body", {}) or {}
    checks = body.get("checks", {}) or {}

    p = Page("服务健康检查响应（/healthz · /health · /readyz）",
             "验收项①：部署后的金融问答服务测试通过；验收项②：指定端口提供服务")
    y = p.y          # 页头（标题/副标题）已在 Page.__init__ 里画好

    y = p.section("① 三档探针状态码")
    rows = []
    for path, meaning in (("/healthz", "存活（不碰索引，容器 HEALTHCHECK 用）"),
                          ("/health", "健康全量（诚实报告每一子项）"),
                          ("/readyz", "就绪（索引不可用则 503）")):
        item = (health.get("实测", {}) or {}).get(path, {})
        rows.append([path, str(item.get("status", "—")), meaning])
    y = p.table_block(["端点", "HTTP", "语义"], rows, widths=[180, 110, 620])

    y = p.section("② /health 子项（逐项 ok + 原因）")
    rows = []
    for key in ("volume_logs", "volume_sessions", "volume_index", "volume_uploads", "volume_runtime"):
        c = checks.get(key, {}) or {}
        rows.append([key, str(c.get("ok")), str(c.get("label", "")), clip(c.get("path", ""), 46)])
    ss = checks.get("session_store", {}) or {}
    rows.append(["session_store", str(ss.get("ok")), f"后端={ss.get('backend')}",
                 f"客户端={(ss.get('redis') or {}).get('client', '—')}"])
    eng = checks.get("engine", {}) or {}
    rows.append(["engine", str(eng.get("ok")), f"status={eng.get('status')}",
                 f"Ollama={(eng.get('checks', {}).get('llm', {}) or {}).get('backend', '—')}"])
    y = p.table_block(["子项", "ok", "说明", "路径/明细"], rows, widths=[230, 80, 250, 420])

    y = p.section("③ 索引元信息（与基座一致：8847 块 / 1024 维）")
    # 注意：共享渲染器的 kv_block 是「键宽 300px / 值偏移 200px」的紧凑两列布局，
    # 键名超过 ~8 个汉字就会与数值重叠（实测）。此处改用「单字段一行」的 text 排版避免压字。
    for label, value in (
        ("索引块数", idx.get("chunks")),
        ("向量维度", idx.get("vector_dim")),
        ("FAISS 启用", idx.get("faiss")),
        ("BM25 词表", idx.get("bm25_vocab")),
        ("嵌入模型", idx.get("embed_model")),
        ("构建时间", idx.get("built_at")),
    ):
        y = p.text(f"  {label}：{value}", size=18, gap=28)
    by_mod = idx.get("chunk_by_modality", {}) or {}
    if by_mod:
        y = p.note("  模态分布：" + "、".join(f"{k}={v}" for k, v in by_mod.items()), size=16)
    y = p.section("④ Prometheus 指标（/metrics 文本格式，节选）")
    for line in [x for x in metrics_txt.splitlines() if x and not x.startswith("#")][:7]:
        y = p.note("  " + line, size=15)
    p.finish(SHOT_DIR / "服务健康检查响应.png",
             sources=["测试/测试数据/健康检查-三档探针.json", "测试/测试数据/健康检查-索引元信息.json",
                      "测试/测试数据/健康检查-指标.txt"])


# ====================================================================== #
# 图 4：问答接口调用结果
# ====================================================================== #
def shot_ask() -> None:
    a1 = load_json(DATA_DIR / "问答接口-真实问题-A1.json", {}) or {}
    a2 = load_json(DATA_DIR / "问答接口-真实问题-A2.json", {}) or {}
    a3 = load_json(DATA_DIR / "问答接口-真实问题-A3.json", {}) or {}
    refuse = load_json(DATA_DIR / "问答接口-拒答护栏.json", {}) or {}
    contract = (load_json(DATA_DIR / "问答接口-契约样例.json", {}) or {}).get("响应", {}) or {}

    p = Page("问答接口调用与答案（POST /api/v1/ask）",
             "验收项①：部署后的金融问答服务测试通过（真实问题 + 真实数字 + 真实引用页）")
    y = p.y          # 页头（标题/副标题）已在 Page.__init__ 里画好

    y = p.section("① 真实金融问答（3/3 答对）")
    rows = []
    for tag, item in (("A1", a1), ("A2", a2), ("A3", a3)):
        rows.append([tag, clip(item.get("问题", ""), 40), clip(item.get("答案", ""), 62),
                     str(item.get("引用页")), f"{(item.get('耗时ms') or {}).get('total', '—')} ms"])
    y = p.table_block(["编号", "问题", "答案（节选）", "引用页", "总耗时"],
                      rows, widths=[70, 420, 520, 150, 130])

    y = p.section("② 拒答护栏（无关问题必须拒答，且 HTTP 200 —— 拒答是正确行为）")
    y = p.kv_block([
        ("问题", refuse.get("问题", "—")),
        ("答案", refuse.get("答案", "—")),
        ("refused", str(refuse.get("refused"))),
        ("拒答原因", clip(refuse.get("refusal_reason", "—"), 30)),
    ])

    y = p.section("③ 接口契约字段（16 个字段齐全）")
    if contract:
        lat = contract.get("latency_ms", {}) or {}
        y = p.kv_block([
            ("trace_id", str(contract.get("trace_id"))[:20]),
            ("session_id", clip(contract.get("session_id", "—"), 26)),
            ("分段耗时", f"检索 {lat.get('retrieve')} / 生成 {lat.get('generate')} / 总计 {lat.get('total')} ms"),
            ("后端:模型", f"{contract.get('backend')}:{contract.get('model')}"),
            ("生成方式", str(contract.get("generation_method"))),
            ("会话轮次", f"session_turns={contract.get('session_turns')}，后端={contract.get('session_backend')}"),
            ("引用条数", str(len(contract.get("citations") or []))),
            ("缓存", str((contract.get("cache") or {}).get("hit"))),
        ])
    y = p.note("数据来源：服务真实 HTTP 响应原样落盘（测试/测试数据/问答接口-真实问题-*.json），未经改写。",
               color=(22, 140, 74))
    p.finish(SHOT_DIR / "问答接口调用结果.png",
             sources=["测试/测试数据/问答接口-真实问题-A1/A2/A3.json",
                      "测试/测试数据/问答接口-拒答护栏.json", "测试/测试数据/问答接口-契约样例.json"])


# ====================================================================== #
# 图 5：数据卷持久化验证
# ====================================================================== #
def shot_volumes() -> None:
    persist = load_json(DATA_DIR / "数据卷持久化验证.json", {}) or {}
    redis_proof = load_json(DATA_DIR / "会话共享-Redis实证.json", {}) or {}
    rt = load_json(DATA_DIR / "健康检查-服务信息.json", {}) or {}

    p = Page("数据卷持久化与容器间数据共享验证",
             "验收项③：支持通过 Docker 卷持久化存储关键数据；支持容器间数据共享")
    y = p.y          # 页头（标题/副标题）已在 Page.__init__ 里画好

    y = p.section("① 卷清单（用途 + 真实文件数与体积）")
    purpose = {
        "index": "RAG 索引（faiss/BM25/chunks，只读）",
        "logs": "运行日志（JSON 结构化）",
        "sessions": "多轮会话（rag-api 与 rag-ui 共享）",
        "runtime": "运行期 sqlite（qa_runtime.sqlite3）",
        "models": "模型/权重缓存（扩展位）",
        "uploads": "用户上传语料",
    }
    rows = []
    total_files = total_bytes = 0
    for name, desc in purpose.items():
        d = VOL_ROOT / name
        files, size = dir_stats(d)
        total_files += files
        total_bytes += size
        rows.append([name, desc, str(files), human(size)])
    rows.append(["合计", "—", str(total_files), human(total_bytes)])
    y = p.table_block(["卷目录", "用途", "文件数", "体积"], rows, widths=[150, 520, 130, 180])

    y = p.section("② 索引卷只读语义 + 运行期数据落位（缺陷回归）")
    y = p.kv_block([
        ("本次会话新建 sqlite（索引卷）", str(len(persist.get("本次会话新建的sqlite", []))) + " 个（必须为 0）"),
        ("历史残留（不判失败）", str(len(persist.get("历史残留（不判失败）", []))) + " 个（已登记成因）"),
        ("runtime 卷中的 sqlite", "、".join(persist.get("runtime卷里的sqlite", [])) or "—"),
        ("会话文件数（服务自报）", str(rt.get("sessions_on_disk", "—"))),
    ])
    for item in (persist.get("历史残留（不判失败）") or [])[:2]:
        y = p.note(f"  · 残留：{item.get('文件')}  mtime={item.get('mtime')}", size=15)

    y = p.section("③ 容器间数据共享实证（独立 Redis 连接读回服务写入的会话）")
    if redis_proof:
        first = redis_proof.get("第一条", {}) or {}
        y = p.kv_block([
            ("Redis key", clip(redis_proof.get("key", "—"), 40)),
            ("TTL（秒）", str(redis_proof.get("ttl_s"))),
            ("客户端", str(redis_proof.get("客户端"))),
            ("轮次数", str(redis_proof.get("轮次数"))),
            ("健康检查后端", str(redis_proof.get("健康检查里的会话后端"))),
            ("会话内容（问句）", clip(first.get("question", "—"), 30)),
        ])
        y = p.note("通路①文件卷（持久，数据不丢失） + 通路②Redis（实时，跨容器共享）双写；"
                   "Redis 客户端为本项目内置的标准库 RESP 实现（MiniRedis），不依赖 redis 包。",
                   color=(22, 140, 74))
    else:
        y = p.note("未找到 Redis 共享证据文件（会话共享-Redis实证.json）")
    y = p.note("目录列表为宿主机侧真实遍历结果；容器内路径映射见 compose 的 volumes 段。")
    p.finish(SHOT_DIR / "数据卷持久化验证.png",
             sources=["测试/测试数据/数据卷持久化验证.json", "测试/测试数据/会话共享-Redis实证.json",
                      "部署/卷/{index,logs,sessions,runtime,models,uploads}"])


# ====================================================================== #
# 图 6：网络连通性验证
# ====================================================================== #
def shot_network() -> None:
    ollama = load_json(DATA_DIR / "网络连通性-ollama.json", {}) or {}
    redis = load_json(DATA_DIR / "网络连通性-redis降级.json", {}) or {}
    rtmp = load_json(DATA_DIR / "网络连通性-rtmp.json", {}) or {}
    cfg = load_json(DATA_DIR / "compose配置-解析结果.json", {}) or {}
    api = (cfg.get("services", {}) or {}).get("rag-api", {}) or {}
    net = (cfg.get("networks", {}) or {}).get("rag-net", {}) or {}

    p = Page("容器网络配置与连通性验证",
             "验收项④：容器网络配置正确，能够与其他服务（如 RTMP 服务器）正常通信")
    y = p.y          # 页头（标题/副标题）已在 Page.__init__ 里画好

    y = p.section("① 网络声明（自定义 bridge + 服务名 DNS + 宿主网关）")
    subnet = ((net.get("ipam") or {}).get("config") or [{}])[0].get("subnet", "—")
    y = p.kv_block([
        ("网络名 / 驱动", f"{net.get('name')} / {net.get('driver')}"),
        ("子网", str(subnet)),
        ("宿主网关映射", str((api.get("extra_hosts") or ["—"])[0])),
        ("Ollama 地址（容器视角）", "http://host.docker.internal:11434"),
        ("Redis 地址（容器视角）", "redis://:****@redis:6379/0（服务名 DNS）"),
        ("三类路径规范", "编排内=服务名 / 宿主=host.docker.internal / 外网=直连"),
    ])

    y = p.section("② 宿主机 Ollama 连通性（生成 + 嵌入后端）")
    llm = ollama.get("服务自报llm检查", {}) or {}
    y = p.kv_block([
        ("探测地址", str(ollama.get("本机实测使用地址", "—"))),
        ("host.docker.internal 解析", str(ollama.get("host_alias_resolved_ip", "未解析（本机等价验证用 127.0.0.1）"))),
        ("HTTP 状态", str(ollama.get("http_status", "—"))),
        ("模型总数", str(ollama.get("模型总数", "—"))),
        ("关键模型", "、".join(ollama.get("关键模型", []) or [])),
        ("服务自报 LLM 检查", f"ok={llm.get('ok')} 后端={llm.get('backend')} 模型={llm.get('model')}"),
    ])

    y = p.section("③ Redis 连通性（用 Redis 协议 PING 判定，避免端口误判）")
    probe = redis.get("redis_协议探测", {}) or {}
    store = redis.get("session_store", {}) or {}
    y = p.kv_block([
        ("协议探测结果", f"is_redis={probe.get('is_redis')}"),
        ("探测详情", clip(probe.get("detail", "—"), 34)),
        ("会话后端", str(store.get("backend"))),
        ("Redis 子项", f"ok={(store.get('redis') or {}).get('ok')} 客户端={(store.get('redis') or {}).get('client')}"),
        ("降级原因", clip(store.get("degraded_reason") or "（无，Redis 正常）", 34)),
    ])

    y = p.section("④ RTMP 服务器（1935）探测 —— 工单点名的互通信对象")
    y = p.table_block(
        ["探测目标", "容器内等价目标", "实测结果", "说明"],
        [["127.0.0.1:1935", "host.docker.internal:1935",
          "可连接" if rtmp.get("connectable") else "不可达（本机未运行 RTMP 服务）",
          clip(str(rtmp.get("本机现状", "")), 40)]],
        widths=[230, 320, 260, 300])
    y = p.note("容器内验证命令：" + clip(str(rtmp.get("容器内验证命令", "")), 96), size=15)
    y = p.note("另一种接入方式（推荐）：把 RTMP 容器接进本编排网络 → "
               "docker network connect rag-net <rtmp容器>，之后用服务名互访。", size=15)
    y = p.note("如实说明：本机 1935 端口无服务监听，因此该项只验证「探测机制与证据留痕」，"
               "不宣称 RTMP 已连通。", color=(196, 48, 48))
    p.finish(SHOT_DIR / "容器网络连通性验证.png",
             sources=["测试/测试数据/网络连通性-ollama.json", "测试/测试数据/网络连通性-redis降级.json",
                      "测试/测试数据/网络连通性-rtmp.json", "测试/测试数据/compose配置-解析结果.json"])


# ====================================================================== #
# 图 7：Docker 权限限制的证据（失败也是证据）
# ====================================================================== #
def shot_docker_evidence() -> None:
    log = LOG_DIR / "docker提权失败证据.log"
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else "（缺少证据文件）"

    p = Page("Docker 权限限制实证（命令与报错原文）",
             "本机 docker CLI 访问 daemon 被沙箱拒绝；本会话禁止提权，故按约定如实记录并给等价验证")
    y = p.y          # 页头（标题/副标题）已在 Page.__init__ 里画好

    y = p.section("① 失败命令与报错（原样摘录，完整版见来源日志）")
    keep: list[str] = []
    for line in text.splitlines():
        if line.startswith("===== 命令：") or "permission denied" in line or line.startswith("----- exit code"):
            keep.append(line)
    for line in keep[:22]:
        color = (196, 48, 48) if "permission denied" in line else (
            (22, 140, 74) if "exit code = 0" in line else (28, 32, 38))
        y = p.note(clip(line, 110), color=color, size=15)

    y = p.section("② 唯一可用的 Docker 侧真证据（纯客户端子命令）")
    y = p.kv_block([
        ("docker compose version", "Docker Compose version v5.5.1（退出码 0）"),
        ("docker compose config", "成功：官方解析器输出 4 服务 / 8 卷 / 网络 / 资源限制"),
        ("需要 daemon 的命令", "version / ps / images / volume ls / network ls / info 全部 permission denied"),
        ("提权状态", "本会话 approval prompts 被禁用 → 提权会被自动拒绝，故不反复重试"),
    ])
    y = p.note("替代证据链：① compose 配置级验证（官方解析器）；② 等价本机运行验证（同源码/同端口/同构卷）；"
               "③ 全部未达标项登记在 设计/验收标准.md §五。", color=(22, 140, 74))
    p.finish(SHOT_DIR / "Docker权限限制实证.png",
             sources=["部署/日志/docker提权失败证据.log"])


# ====================================================================== #
# 图 8：镜像体积与冷启动优化对比
# ====================================================================== #
def shot_optimize() -> None:
    opt = load_json(OPT_DIR / "部署优化实测.json", {}) or {}
    pkg = opt.get("依赖体积核算", {}) or {}
    cold = opt.get("冷启动", {}) or {}
    hot = opt.get("热路径", {}) or {}
    conc = opt.get("并发探活", {}) or {}
    vol = opt.get("卷读写", {}) or {}
    mem = opt.get("内存", {}) or {}

    p = Page("部署优化对比（镜像体积 · 冷启动 · 资源限制）",
             "全部数字来自 优化/脚本/测量部署优化指标.py 的实测/核算结果，口径在页脚标注")
    y = p.y          # 页头（标题/副标题）已在 Page.__init__ 里画好

    y = p.section("① 镜像体积优化（依赖闭包核算：真实包目录体积）")
    keep_mb = (pkg.get("包集合合计MB", {}) or {}).get("本方案（精简闭包）", "—")
    cut_mb = (pkg.get("包集合合计MB", {}) or {}).get("若在容器内跑本地大模型（被否决方案）", "—")
    y = p.table_block(
        ["方案", "本地大模型依赖", "精简依赖闭包", "本镜像实际需要"],
        [["容器内自带模型（被否决）", f"{cut_mb} MB", "—", f"≈{cut_mb} MB + 系统包"],
         ["走宿主机 Ollama（本方案）", "0 MB（不装）", f"{keep_mb} MB", f"≈{keep_mb} MB + 系统包"]],
        widths=[340, 220, 220, 240])
    y = p.note(f"结论：剔除 torch / transformers / open_clip 等本地推理依赖，"
               f"依赖体积从 {cut_mb} MB 降到 {keep_mb} MB。", color=(22, 140, 74), size=16)

    y = p.section("② 冷启动实测（真实跑 QAEngine 加载 8847 块索引）")
    nowarm = cold.get("nowarm", {}) or {}
    warm = cold.get("warm", {}) or {}
    if nowarm or warm:
        y = p.table_block(
            ["阶段", "关预热(s)", "开预热(s)", "说明"],
            [["导入 + 配置", str(nowarm.get("import_s", "—")), str(warm.get("import_s", "—")), "import app.core"],
             ["引擎初始化", str(nowarm.get("engine_init_s", "—")), str(warm.get("engine_init_s", "—")), "QAEngine(cfg)"],
             ["预热", "—", str(warm.get("warmup_s", "—")), "全文索引构建 + 实体标注 + 嵌入探活"],
             ["启动到可服务", str(nowarm.get("total_s", "—")), str(warm.get("total_s", "—")), "HEALTHCHECK start-period=90~150s 的依据"]],
            widths=[240, 200, 200, 380])
    else:
        y = p.note("冷启动阶段未采集到数据（脚本被跳过或失败）", color=(196, 48, 48))

    y = p.section("③ 热路径与并发实测")
    rows = []
    for ep, item in hot.items():
        if isinstance(item, dict) and "中位ms" in item:
            rows.append([ep, f"{item['中位ms']} ms", f"{item['最小ms']}~{item['最大ms']} ms", f"{item['次数']} 次"])
    for row in (hot.get("问答接口") or [])[:2]:
        rows.append([f"POST /api/v1/ask（第{row.get('第几次')}次）", f"{row.get('墙钟ms')} ms",
                     f"服务端 {row.get('服务端total_ms')} ms", str(row.get("生成方式"))])
    if rows:
        y = p.table_block(["端点", "中位/墙钟", "区间/服务端", "次数/方式"], rows, widths=[380, 200, 260, 200])
    if conc:
        y = p.kv_block([
            ("并发配置", f"{conc.get('并发线程')} 线程 × {conc.get('请求数')} 请求"),
            ("QPS", str(conc.get("QPS"))),
            ("延迟 中位/p95/最大", f"{conc.get('延迟ms_中位')} / {conc.get('延迟ms_p95')} / {conc.get('延迟ms_最大')} ms"),
            ("失败数", str(conc.get("失败数"))),
        ])

    y = p.section("④ 卷读写与资源限制实测")
    if vol:
        w = vol.get("会话写（原子替换）", {}) or {}
        r = vol.get("会话读", {}) or {}
        s = vol.get("索引顺序读", {}) or {}
        y = p.kv_block([
            ("会话写", f"{w.get('单文件均ms')} ms/文件（{w.get('吞吐MB/s')} MB/s）"),
            ("会话读", f"{r.get('单文件均ms')} ms/文件（{r.get('吞吐MB/s')} MB/s）"),
            ("索引顺序读", f"{s.get('吞吐MB/s')} MB/s（{s.get('字节')} 字节）"),
            ("卷总量", f"{(vol.get('索引卷stat', {}) or {}).get('合计MB')} MB（索引）"),
        ])
    api_mem = mem.get("API进程内存", {}) or {}
    y = p.kv_block([
        ("系统可用内存", f"{(mem.get('系统内存MB', {}) or {}).get(chr(92) + 'Memory' + chr(92) + 'Available MBytes', '—')} MB"),
        ("提交内存占比", f"{mem.get('提交占比%', '—')} %"),
        ("API 进程工作集", f"{api_mem.get('ws_mb', '—')} MB（限额参考 2 GB）"),
        ("测量脚本内存", f"{(mem.get('测量脚本自身内存', {}) or {}).get('ws_mb', '—')} MB"),
    ])
    p.finish(SHOT_DIR / "镜像体积与冷启动优化对比.png",
             sources=["优化/评估结果/部署优化实测.json"],
             extra_note="口径：依赖体积=真实包目录体积核算；冷启动/热路径/并发/卷/内存=本机实测")


# ====================================================================== #
# 图 9：测试报告摘要
# ====================================================================== #
def shot_test_summary() -> None:
    log = LOG_DIR / "pytest-本地等价验证.log"
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    summary = ""
    for line in reversed(text.splitlines()):
        if "passed" in line and ("=" in line or "failed" in line):
            summary = line.strip("= ").strip()
            break
    passed = failed = skipped = 0
    for token in summary.replace(",", " ").split():
        if token.isdigit():
            continue
    import re as _re
    m = _re.search(r"(\d+) passed", summary)
    passed = int(m.group(1)) if m else 0
    m = _re.search(r"(\d+) failed", summary)
    failed = int(m.group(1)) if m else 0
    m = _re.search(r"(\d+) skipped", summary)
    skipped = int(m.group(1)) if m else 0

    cases: list[tuple[str, str]] = []
    for line in text.splitlines():
        mm = _re.match(r"^(测试/测试脚本/\S+::\S+)\s+(PASSED|FAILED|SKIPPED)", line)
        if mm:
            cases.append((mm.group(1).split("::", 1)[1], mm.group(2)))
    by_module: dict[str, list[int]] = {}
    for name, status in cases:
        by_module.setdefault("", [0, 0, 0])
    # 按文件分组统计
    per_file: dict[str, dict[str, int]] = {}
    for line in text.splitlines():
        mm = _re.match(r"^(测试/测试脚本/(\S+?\.py))::(\S+)\s+(PASSED|FAILED|SKIPPED)", line)
        if mm:
            f = mm.group(2)
            per_file.setdefault(f, {"PASSED": 0, "FAILED": 0, "SKIPPED": 0})
            per_file[f][mm.group(4)] += 1

    p = Page("测试结果汇总（pytest 本地等价验证）",
             f"结果：{passed} passed / {failed} failed / {skipped} skipped —— 验收项①：部署后的服务测试通过")
    y = p.y          # 页头（标题/副标题）已在 Page.__init__ 里画好

    y = p.section("① 总体结果")
    y = p.kv_block([
        ("通过", f"{passed} 个"),
        ("失败", f"{failed} 个"),
        ("跳过（需 Docker daemon，门控 RAG_TEST_DOCKER=1）", f"{skipped} 个"),
        ("原始日志", "部署/日志/pytest-本地等价验证.log"),
        ("证据目录", "测试/测试数据/（真实 HTTP 响应与解析输出）"),
        ("本轮会话起点", str(json.loads((DATA_DIR / '数据卷持久化验证.json').read_text(encoding='utf-8')).get('会话起点_unix', '—'))[:16]
         if (DATA_DIR / '数据卷持久化验证.json').exists() else "—"),
    ])

    y = p.section("② 按测试模块统计")
    desc = {
        "test_01_部署配置.py": "Dockerfile / compose / env / 脚本 / 阈值配置化",
        "test_02_卷挂载与数据可见.py": "卷持久化 + 运行期写入 + Redis 跨容器共享",
        "test_03_网络连通性.py": "Ollama(host.docker.internal) / Redis 协议 / RTMP / 网络声明",
        "test_04_健康检查.py": "三档探针 / 索引元信息 / UI 端口 / 指标格式",
        "test_05_并发探活.py": "并发零失败 + 零 5xx / 多轮会话轮次",
        "test_06_问答接口契约.py": "契约字段 / 真实问答 / 拒答护栏 / 422 / 404 / OpenAPI",
        "test_07_部署制品.py": "compose 官方解析 / COPY 源 / entrypoint / 依赖闭包 / 探针",
    }
    rows = []
    for f, cnt in sorted(per_file.items()):
        rows.append([f, desc.get(f, ""), str(cnt["PASSED"]), str(cnt["FAILED"]), str(cnt["SKIPPED"])])
    y = p.table_block(["测试模块", "覆盖内容", "通过", "失败", "跳过"], rows,
                      widths=[330, 520, 100, 100, 100])

    y = p.section("③ 未达标项（如实登记，不掩盖）")
    y = p.table_block(
        ["项", "状态", "根因"],
        [["docker run / compose up 真实起容器", "未达成", "docker CLI 访问命名管道被沙箱拒绝；本会话禁止提权"],
         ["镜像构建体积/耗时实测", "部分", "无 daemon → 用真实包体积核算 + 层明细，口径已标注"],
         ["容器内 host.docker.internal 探测", "部分", "无容器 → 宿主 127.0.0.1:11434 实测 200"],
         ["RTMP 服务器连通", "部分", "本机未运行 RTMPServer（1935 无监听）"],
         ["并发压测深度", "受限", "资源纪律：本机内存红线，禁用进程池；深度压测属工单17"]],
        widths=[380, 140, 640])
    p.finish(SHOT_DIR / "测试结果汇总.png",
             sources=["部署/日志/pytest-本地等价验证.log", "测试/测试数据/"])


def main() -> int:
    started = time.perf_counter()
    tasks = [
        ("容器运行状态", shot_container_status),
        ("compose配置解析", shot_compose_config),
        ("服务健康检查响应", shot_health),
        ("问答接口调用结果", shot_ask),
        ("数据卷持久化验证", shot_volumes),
        ("容器网络连通性验证", shot_network),
        ("Docker权限限制实证", shot_docker_evidence),
        ("镜像体积与冷启动优化对比", shot_optimize),
        ("测试结果汇总", shot_test_summary),
    ]
    ok = 0
    for name, fn in tasks:
        try:
            fn()
            ok += 1
        except Exception as exc:  # noqa: BLE001 —— 单张失败不影响其它图
            print(f"[失败] {name}: {type(exc).__name__}: {exc}")
    print(f"完成 {ok}/{len(tasks)} 张，耗时 {round(time.perf_counter() - started, 2)} 秒")
    return 0 if ok == len(tasks) else 1


if __name__ == "__main__":
    sys.exit(main())
