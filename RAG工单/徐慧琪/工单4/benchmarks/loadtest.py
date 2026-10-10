# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""并发压测：10 / 50 / 100 并发下的响应时间、错误率与资源占用。

用法：
  python benchmarks/loadtest.py                      # 默认打 127.0.0.1:8000
  python benchmarks/loadtest.py http://host:8000 10,50,100 300

地址必须写 IPv4 字面量：服务端 uvicorn 绑 ``0.0.0.0``（仅 IPv4），而本机
``localhost`` 先解析到 ``::1``，客户端每次新建连接都要先等 IPv6 连接失败再回退
—— 实测同一端点 ``localhost:8000`` 2065/2047/2053ms vs ``127.0.0.1:8000``
46/46/15ms，会把这 2 秒算进「客户端墙钟」并污染时延结论。

每档请求数缺省为「最大并发 × 3」；显式传入小于最大并发的值会被上调并告警 ——
请求数少于并发数时队列在毫秒级被抢空，实际在飞请求数达不到标称并发，报告就成了
「声称 100 并发、实际 60」。每档另记**在飞请求峰值**（``max_inflight``）并逐级
展示，低于标称并发时报告会给出告警行。

时延口径（Task 19 复核结论，勿混用）：
  - **主口径 = 客户端墙钟**：``time.perf_counter()`` 包住整个 HTTP 往返，含 HTTP
    开销、连接建立与线程池排队。工单硬约束「端到端 ≤3 秒」按此判定。
  - **服务端上报 = ``latency_ms``**：流水线内自测，不含 HTTP 与排队，高并发下必然
    小于客户端观测值；只作对照（``server_*`` 字段），**不参与达标判定**。
  - **达标判定 = ``rag04.config.latency_verdict``**：先把客户端墙钟取到 1 位小数
    再与预算比较，与 ``api/server.py`` 的 ``within_budget``、``ui/app.py`` 的显示
    口径同一实现，三处对同一次请求给出同一结论。

本模块的单测（tests/test_loadtest.py）只碰纯函数：``_percentile`` / ``_summarize``
/ ``_server_latency`` / ``write_report``；网络请求与进程内存测量只发生在
``run_level`` 实跑时，导入本模块不会建连接、不开索引。
"""
from __future__ import annotations

import json
import statistics
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from queue import Empty, Queue

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import requests  # noqa: E402

try:
    # 达标判定只有一份实现（先取到 1 位小数再比较）：与 api/server.py、ui/app.py
    # 共用，压测报告与界面/接口对同一次请求给出同一个结论。
    from rag04.config import latency_verdict
except Exception:  # noqa: BLE001 - 脱离仓库运行本脚本时的自足兜底
    def latency_verdict(ms: float, budget_ms: float) -> tuple[float, bool]:
        shown = round(float(ms), 1)
        return shown, shown <= float(budget_ms)

DEFAULT_QUESTIONS = [
    "武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成，其中大客户销售部有几个销售处构成？",
    "从2008年中国IC市场应用结构与增长图中可以看出，增长率最快的是哪个行业？",
    "武汉力源信息技术股份有限公司本次发行股数是多少？",
    "武汉兴图新科电子股份有限公司的法定代表人是谁？",
    "武汉兴图新科电子股份有限公司的注册资本是多少？",
]

# requests 的 timeout 是**每次 socket 操作**（连接/读取）的上限，不是整个请求的
# 总时限：响应只要持续有数据就可能远超此值。工单预算的 40 倍仅用于识别「连接或
# 读取彻底卡死」，若需硬性总时限，得另加看门狗（当前未加，报告也不据此截断）。
REQUEST_TIMEOUT_S = 120.0
DEFAULT_PER_LEVEL_FACTOR = 3     # 每档请求数缺省 = 最大并发 × 3（保证施加得出去）
_FALLBACK_BUDGET_MS = 3000.0     # rag04.config 不可用时的自足兜底（工单硬指标）
_BUDGET_SOURCE_SETTINGS = "Settings.latency_budget_ms"
_BUDGET_SOURCE_FALLBACK = f"兜底常量（未能读取 rag04.config，值 {_FALLBACK_BUDGET_MS:g}ms）"


def budget_source() -> tuple[float, str]:
    """返回 ``(预算毫秒, 来源说明)``：来源如实区分配置项与兜底常量。

    报告据此标注阈值出处 —— 兜底时不能声称「来自 ``Settings.latency_budget_ms``」，
    那会把「没读到配置」写成「配置就是这么定的」。
    """
    try:
        from rag04.config import get_settings
        return float(get_settings().latency_budget_ms), _BUDGET_SOURCE_SETTINGS
    except Exception:
        return _FALLBACK_BUDGET_MS, _BUDGET_SOURCE_FALLBACK


def default_budget_ms() -> float:
    """端到端时延预算（毫秒）：优先读 ``Settings.latency_budget_ms``。

    读不到（例如脱离仓库运行本脚本）时退回工单硬指标的 3000ms，保证报告自足、
    不因缺配置而丢掉预算判定；出处由 :func:`budget_source` 如实标注。
    """
    return budget_source()[0]


@dataclass
class BenchResult:
    """单档并发的压测结果。

    前 12 个字段与 brief 的位置参数顺序**逐位一致**（外部契约，勿重排）。
    ``p50_ms`` / ``p95_ms`` / ``p99_ms`` / ``mean_ms`` 一律是**客户端墙钟**（主
    口径，含 HTTP 与排队）：工单的「端到端 ≤3 秒」只认这组数。服务端上报的
    ``latency_ms`` 存于 ``server_*``，仅作对照；``as_dict()`` 把两组数据分别
    命名为 ``client_*`` / ``server_*``，避免任何场合下被误读成同一口径。
    """

    concurrency: int
    n_requests: int
    ok: int
    errors: int
    error_rate: float
    qps: float
    p50_ms: float                  # 客户端墙钟 P50（主口径）
    p95_ms: float                  # 客户端墙钟 P95（主口径）
    p99_ms: float                  # 客户端墙钟 P99（主口径）
    mean_ms: float                 # 客户端墙钟均值（主口径）
    rss_mb: float
    gpu_mb: float | None           # 测不到显存时一律 None，绝不填 0 冒充
    # --- 以下为 brief 之外的新增，均带默认值，不影响 brief 的按位构造 ---
    budget_ms: float = field(default_factory=default_budget_ms)
    n_within_budget: int | None = None   # 客户端墙钟达标数（latency_verdict 判定）
    server_p50_ms: float | None = None
    server_p95_ms: float | None = None
    server_p99_ms: float | None = None
    server_mean_ms: float | None = None
    error_kinds: dict[str, int] = field(default_factory=dict)
    # 在飞请求峰值：真正同时施加的并发上限。0 = 未记录（手工构造的结果）。
    max_inflight: int = 0
    # 成功但被拒答（``refused=true``）的条数：拒答是正确服务行为，单列不藏
    refused: int = 0
    # 服务端列（server_*）的样本数：不是每个成功响应都带数值 latency_ms
    n_server_samples: int = 0

    @property
    def client_p50_ms(self) -> float:
        """客户端墙钟 P50（主口径，等价于 ``p50_ms``）。"""
        return self.p50_ms

    @property
    def client_p95_ms(self) -> float:
        """客户端墙钟 P95（主口径，等价于 ``p95_ms``）。"""
        return self.p95_ms

    @property
    def client_p99_ms(self) -> float:
        """客户端墙钟 P99（主口径，等价于 ``p99_ms``）。"""
        return self.p99_ms

    @property
    def client_mean_ms(self) -> float:
        """客户端墙钟均值（主口径，等价于 ``mean_ms``）。"""
        return self.mean_ms

    @property
    def within_budget_rate(self) -> float:
        """达标率 = 预算内的成功请求数 / 总请求数（无样本时为 0.0）。"""
        if not self.n_requests or self.n_within_budget is None:
            return 0.0
        return self.n_within_budget / self.n_requests

    def as_dict(self) -> dict:
        """带口径标注的字典：客户端/服务端时延分开命名后一并导出。"""
        d = asdict(self)
        d["client_p50_ms"] = self.client_p50_ms
        d["client_p95_ms"] = self.client_p95_ms
        d["client_p99_ms"] = self.client_p99_ms
        d["client_mean_ms"] = self.client_mean_ms
        d["within_budget_rate"] = round(self.within_budget_rate, 4)
        return d


def _percentile(data: list[float], p: float) -> float:
    """线性插值分位数。空列表返回 0。"""
    if not data:
        return 0.0
    s = sorted(data)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


def _server_latency(payload: object) -> float | None:
    """取服务端上报的 ``latency_ms``；缺失或非数值一律 None，不猜也不填 0。"""
    if not isinstance(payload, dict):
        return None
    value = payload.get("latency_ms")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _server_refused(payload: object) -> bool:
    """响应体 ``refused`` 严格为 True 才算拒答（缺字段/非布尔一律 False）。"""
    return isinstance(payload, dict) and payload.get("refused") is True


def _summarize(concurrency: int, n_requests: int, ok_count: int,
               client_ms: list[float], client_ok_ms: list[float],
               server_ms: list[float], elapsed_s: float, rss_mb: float,
               gpu_mb: float | None, budget_ms: float,
               error_kinds: dict[str, int] | None = None,
               max_inflight: int = 0, refused_count: int = 0) -> BenchResult:
    """把原始样本汇总成 :class:`BenchResult`（纯函数，便于单测）。

    ``client_ms`` 是全部请求的客户端墙钟（含失败请求，分位数用全套样本）；
    ``client_ok_ms`` 只含成功请求，预算达标数从这里数 —— 失败的请求没有产出
    答案，不能算「满足 ≤3 秒」。

    样本数必须与计数自洽（丢失或重复记账会直接歪曲错误率与达标率），不一致即
    抛 ``ValueError``：宁可让压测档位失败，也不产出对不上的报告。
    """
    if len(client_ms) != n_requests:
        raise ValueError(
            f"客户端样本数 {len(client_ms)} != 请求数 {n_requests}：记账丢失或重复")
    if len(client_ok_ms) != ok_count:
        raise ValueError(
            f"成功样本数 {len(client_ok_ms)} != 成功计数 {ok_count}：记账不一致")
    if len(server_ms) > ok_count:
        raise ValueError(
            f"服务端样本数 {len(server_ms)} > 成功计数 {ok_count}：样本来源异常")
    errors = n_requests - ok_count
    return BenchResult(
        concurrency=concurrency, n_requests=n_requests, ok=ok_count, errors=errors,
        error_rate=round(errors / n_requests, 4) if n_requests else 0.0,
        qps=round(n_requests / elapsed_s, 2) if elapsed_s > 0 else 0.0,
        p50_ms=round(_percentile(client_ms, 50), 1),
        p95_ms=round(_percentile(client_ms, 95), 1),
        p99_ms=round(_percentile(client_ms, 99), 1),
        mean_ms=round(statistics.mean(client_ms), 1) if client_ms else 0.0,
        rss_mb=round(rss_mb, 1),
        gpu_mb=round(gpu_mb, 1) if gpu_mb is not None else None,
        budget_ms=float(budget_ms),
        # 达标判定走 rag04.config.latency_verdict（先取 1 位小数再比较）：与
        # api/server.py 的 within_budget、ui/app.py 的显示口径完全一致，
        # 否则 3000.04ms 在接口里算达标、在报告里算超标。
        n_within_budget=sum(1 for ms in client_ok_ms
                            if latency_verdict(ms, budget_ms)[1]),
        server_p50_ms=round(_percentile(server_ms, 50), 1) if server_ms else None,
        server_p95_ms=round(_percentile(server_ms, 95), 1) if server_ms else None,
        server_p99_ms=round(_percentile(server_ms, 99), 1) if server_ms else None,
        server_mean_ms=round(statistics.mean(server_ms), 1) if server_ms else None,
        error_kinds=dict(error_kinds or {}),
        max_inflight=int(max_inflight),
        refused=int(refused_count),
        n_server_samples=len(server_ms),
    )


def _measure_mem() -> tuple[float, float | None]:
    """本进程 RSS(MB) 与显存(MB)；显存测不到就回 None（不编造 0）。"""
    try:
        from rag04.obs.logging import mem_snapshot
        snap = mem_snapshot()
        gpu = snap.get("gpu_mb")
        return float(snap.get("rss_mb") or 0.0), (None if gpu is None else float(gpu))
    except Exception:
        return 0.0, None


@dataclass(frozen=True)
class _Sample:
    """单次请求的原始样本。``kind`` 为 ``ok`` 或失败类型（``timeout``/``http_5xx``…）。"""

    ok: bool
    client_ms: float
    server_ms: float | None
    kind: str
    refused: bool = False


def _one_request(url: str, question: str, session: requests.Session | None = None,
                 timeout_s: float = REQUEST_TIMEOUT_S) -> _Sample:
    """打一次 ``POST /api/ask``，返回 :class:`_Sample`。

    客户端时延从发请求前开始计时，含连接与排队；服务端时延与拒答标记从响应体
    解析，解析不到即 None/False。``timeout_s`` 只约束单次 socket 操作（见
    ``REQUEST_TIMEOUT_S`` 注释），不是整请求总时限。
    ``session`` 复用时同一线程走 keep-alive，避免把 TCP 握手算进每次请求，也避免
    100 线程各自新建连接耗尽临时端口。
    """
    poster = session.post if session is not None else requests.post
    t0 = time.perf_counter()
    try:
        resp = poster(f"{url.rstrip('/')}/api/ask",
                      json={"question": question}, timeout=timeout_s)
        client_ms = (time.perf_counter() - t0) * 1000.0
        if resp.status_code != 200:
            return _Sample(False, client_ms, None, f"http_{resp.status_code}")
    except requests.exceptions.Timeout:
        return _Sample(False, (time.perf_counter() - t0) * 1000.0, None, "timeout")
    except Exception as e:                                   # 连接拒绝/DNS/SSL 等
        return _Sample(False, (time.perf_counter() - t0) * 1000.0, None,
                       type(e).__name__)
    try:
        payload = resp.json()
    except Exception:
        return _Sample(False, client_ms, None, "bad_json")
    return _Sample(True, client_ms, _server_latency(payload), "ok",
                   _server_refused(payload))


def run_level(url: str, concurrency: int, n_requests: int,
              questions: list[str], budget_ms: float | None = None) -> BenchResult:
    """在指定并发下打 n_requests 次（线程池 = 并发数，请求按轮转分配问题）。

    ``budget_ms`` 缺省时读 ``Settings.latency_budget_ms``。**需真实服务在跑**：
    本函数会建 socket，属集成范畴，不在单测覆盖内。
    返回结果里的 ``max_inflight`` 是实际在飞请求峰值：``n_requests < concurrency``
    时该档的标称并发不可能达到（队列被抢空后线程即退出），报告会据此告警。
    """
    if not questions:
        raise ValueError("questions 不能为空：压测必须带真实问题")
    budget = float(budget_ms) if budget_ms is not None else default_budget_ms()

    pending: Queue[str] = Queue()
    for i in range(n_requests):
        pending.put(questions[i % len(questions)])

    lock = threading.Lock()
    client_ms: list[float] = []      # 全部请求（含失败）的客户端墙钟
    client_ok_ms: list[float] = []   # 成功请求的客户端墙钟（预算达标数的分子）
    server_ms: list[float] = []      # 成功响应里服务端上报的 latency_ms
    error_kinds: dict[str, int] = {}
    ok_count = 0
    refused_count = 0
    # 在飞请求峰值 = 真正同时施加过的并发上限（标称并发可能达不到：请求数不足、
    # 线程启动错峰、上游限流都会压低它），报告逐级展示。
    inflight_lock = threading.Lock()
    inflight = 0
    max_inflight = 0

    def worker() -> None:
        nonlocal ok_count, refused_count, inflight, max_inflight
        session = requests.Session()
        try:
            while True:
                try:
                    question = pending.get_nowait()
                except Empty:
                    return
                with inflight_lock:
                    inflight += 1
                    max_inflight = max(max_inflight, inflight)
                try:
                    sample = _one_request(url, question, session)
                finally:
                    with inflight_lock:
                        inflight -= 1
                with lock:
                    client_ms.append(sample.client_ms)
                    if sample.ok:
                        ok_count += 1
                        client_ok_ms.append(sample.client_ms)
                        if sample.server_ms is not None:
                            server_ms.append(sample.server_ms)
                        if sample.refused:
                            refused_count += 1
                    else:
                        error_kinds[sample.kind] = error_kinds.get(sample.kind, 0) + 1
        finally:
            session.close()

    start = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(max(1, concurrency))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - start

    rss_mb, gpu_mb = _measure_mem()
    return _summarize(concurrency=concurrency, n_requests=n_requests,
                      ok_count=ok_count, client_ms=client_ms,
                      client_ok_ms=client_ok_ms, server_ms=server_ms,
                      elapsed_s=elapsed, rss_mb=rss_mb, gpu_mb=gpu_mb,
                      budget_ms=budget, error_kinds=error_kinds,
                      max_inflight=max_inflight, refused_count=refused_count)


def _fmt(value: float | None, nd: int = 1) -> str:
    """数值渲染：None 用 ``-``（不把「没测到」伪装成 0）。"""
    return "-" if value is None else f"{value:.{nd}f}"


def _rss_cell(value: float | None) -> str:
    """RSS 渲染：没测到（``None`` 或 ``0``，见 ``mem_snapshot``）写「未测得」，不填数。

    ``mem_snapshot()`` 在 psutil/resource 都不可用时返回 0.0 表示「未测得」；
    报告必须显式呈现为文字，否则读者会把占位数字当成真实占用。
    """
    return "未测得" if value is None or value <= 0 else f"{value:.1f}"


def _budget_verdict(r: BenchResult) -> str:
    """逐级达标判定：只认客户端墙钟的逐请求达标数，服务端数字不参与。"""
    if r.n_within_budget is None:
        return "未记录"
    if r.n_requests and r.n_within_budget >= r.n_requests:
        return "全部达标"
    return "部分达标" if r.n_within_budget > 0 else "未达标"


def _inflight_note(r: BenchResult) -> str | None:
    """该档是否真的施加了标称并发；没达到就给出可读告警（宁缺毋滥）。

    成因只写代码里真实存在的东西：请求数不足，或线程启动错峰。**不写「上游
    限流」** —— 本服务对 ``/api/ask`` 没有任何请求限流（见下方能力表），
    把施加不足归因于限流会与本报告自相矛盾。
    """
    if r.max_inflight <= 0:
        return (f"- 并发 {r.concurrency}：**在飞峰值未记录**，该档是否真实施加 "
                f"{r.concurrency} 并发无法证实。")
    if r.max_inflight < r.concurrency:
        cause = ("请求数少于并发数" if r.n_requests < r.concurrency
                 else "线程启动错峰（本服务对 /api/ask 没有请求限流，见能力表）")
        return (f"- 并发 {r.concurrency}：**实际在飞峰值仅 {r.max_inflight}**，"
                f"该档未真正施加标称并发（{cause}）；其 QPS 与时延只代表 "
                f"{r.max_inflight} 并发下的表现。")
    return None


def write_report(results: list[BenchResult], path: Path,
                 notes: list[str] | None = None,
                 budget_ms: float | None = None) -> Path:
    """把压测结果写成 Markdown 报告。

    阈值口径：**每档都用该档自己记录的 ``budget_ms``**（即 ``run_level`` 计数时
    用的那个数），报告不再另设一个全局阈值 —— 否则标题与逐行会出现两个互相矛盾
    的阈值。仅当所有档位阈值一致时才在标题里写出该数字；``budget_ms`` 参数只在
    无结果时充当兜底，显式传入且与档位记录冲突时会写入备注。
    报告同时给出两组时延：客户端墙钟（主口径，达标判定只用它）与服务端上报
    （对照，另附样本数）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    explicit_budget = None if budget_ms is None else float(budget_ms)
    if explicit_budget is not None:
        fallback_budget = explicit_budget
    else:
        fallback_budget, _ = budget_source()
    thresholds = sorted({r.budget_ms for r in results})
    if len(thresholds) == 1:
        heading = f"## 三、{thresholds[0]:g} ms 预算逐级判定（客户端墙钟）"
    elif thresholds:
        heading = "## 三、时延预算逐级判定（客户端墙钟，逐级阈值见下表）"
    else:
        heading = f"## 三、{fallback_budget:g} ms 预算逐级判定（客户端墙钟）"

    _, source_label = budget_source()
    ladder = " / ".join(str(r.concurrency) for r in results) or "（无结果）"
    if len(thresholds) == 1:
        budget_line = f"- 端到端预算：{thresholds[0]:g} ms（来源：{source_label}）"
    elif thresholds:
        per_level = "、".join(f"{r.concurrency}→{r.budget_ms:g}" for r in results)
        budget_line = (f"- 端到端预算：逐级不同（{per_level}），判定一律用各档记录值")
    else:
        budget_line = f"- 端到端预算：{fallback_budget:g} ms（来源：{source_label}）"
    lines = [
        "# 高并发压测报告",
        "",
        "## 一、压测方案",
        "",
        "- 压测目标：FastAPI `POST /api/ask`（与前端解耦，避免 UI 开销污染数据）",
        f"- 并发梯度：{ladder}（按实际执行档位自动生成）",
        "- 指标：响应时间（P50/P95/P99）、错误率、QPS、在飞并发峰值、内存与显存占用",
        budget_line,
        "- **时延口径**：主口径为**客户端墙钟**（进程内计时，含 HTTP 开销、连接与"
        "线程池排队）；服务端上报的 `latency_ms` 只测量流水线内部，两项并列展示，"
        "**达标判定一律以客户端墙钟为准**。",
        "",
        "## 二、结果",
        "",
        "| 并发 | 在飞峰值 | 请求数 | 成功 | 拒答 | 错误 | 错误率 | QPS | 客户端 P50(ms)"
        " | 客户端 P95(ms) | 客户端 P99(ms) | 客户端均值(ms) | 服务端 P50(ms)"
        " | 服务端 P95(ms) | 服务端样本数 | RSS(MB) | 显存(MB) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---"
        " | --- | --- | --- | --- |",
    ]
    for r in results:
        inflight = r.max_inflight if r.max_inflight > 0 else "-"
        lines.append(
            f"| {r.concurrency} | {inflight} | {r.n_requests} | {r.ok} | {r.refused} "
            f"| {r.errors} | {r.error_rate} | {r.qps} | {r.client_p50_ms} "
            f"| {r.client_p95_ms} | {r.client_p99_ms} | {r.client_mean_ms} "
            f"| {_fmt(r.server_p50_ms)} | {_fmt(r.server_p95_ms)} "
            f"| {r.n_server_samples} | {_rss_cell(r.rss_mb)} | {_fmt(r.gpu_mb)} |"
        )

    inflight_notes = [n for n in (_inflight_note(r) for r in results) if n]
    if inflight_notes:
        lines += ["", "### 并发施加校验（标称并发 ≠ 实际并发）", ""] + inflight_notes

    lines += [
        "",
        "> 口径：客户端 P50/P95/P99 的样本**含失败请求**（快速失败如连接拒绝、"
        "http_5xx 会拉低分位数，故分位数不可单独当作「成功请求的响应时间」读）；"
        "服务端分位数只含**返回了数值 `latency_ms` 的成功响应**，其分母见"
        "「服务端样本数」列。",
        "> 成功 = HTTP 200 且响应体可解析。其中**拒答**（`refused=true`，服务在"
        "证据不足时的正确行为）单列：拒答仍计入成功与达标，若只看有实质答案的请求，"
        "达标数应再减去拒答数。",
        "",
        heading,
        "",
        "达标 = 该次请求**成功**且**客户端墙钟 ≤ 该档记录的预算**；失败的请求不产出"
        "答案，即使很快也不算达标。",
        "",
        "| 并发 | 预算(ms) | 达标数 | 总请求 | 达标率 | 失败数 | 判定 | 客户端 P95(ms) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        if r.n_within_budget is None:
            count, rate = "-", "-"
        else:
            count = f"{r.n_within_budget}/{r.n_requests}"
            rate = f"{r.within_budget_rate * 100:.1f}%"
        lines.append(
            f"| {r.concurrency} | {r.budget_ms:g} | {count} | {r.n_requests} | {rate} "
            f"| {r.errors} | {_budget_verdict(r)} | {r.client_p95_ms} |"
        )

    lines += [
        "",
        "## 四、资源与并发控制（逐条对应代码实现）",
        "",
        "> 本表只写代码里**确实存在**的机制；预定义但未接线的能力一律显式标注，"
        "不按设计意图写成已生效。",
        "",
        "| 手段 | 说明 |",
        "| --- | --- |",
        "| Qdrant 连接池 | 嵌入式/服务端双模式；服务端模式单客户端 `pool_size=16` "
        "长连接复用，避免每请求握手（嵌入式模式没有连接池，进程独占目录锁） |",
        "| LLM 并发控制 | **没有请求限流**：`llm_max_concurrency`（默认 8）是预留"
        "配置项、**未接线**（全仓只有定义，没有任何读取方）。每请求同步调用 LLM："
        "显式 `llm_timeout_s` 超时、SDK 侧 `max_retries=0`（不重试；仅当服务端不"
        "识别 thinking 参数时剥离后重发一次），无排队、无熔断，并发由服务线程池"
        "（anyio，默认 40）承载 |",
        "| embedding 批处理 | bge-m3 经 Ollama `/api/embed`，批量 16 条/请求，"
        "失败按 16→8→4→2→1 降批次重试 |",
        "| 结果缓存 | 仅 VLM 图像解析结果按「图像+标题」落盘缓存（`vlm_cache_dir`），"
        "重复图片零计算；**查询侧无结果缓存**，每次提问仍重新嵌入、检索与生成 |",
        "| 模型共享 | bge-m3 由 Ollama 统一托管（HTTP 调用），服务进程不各自加载"
        "该模型；reranker / CLIP 是**进程内**模块级缓存，每个进程各一份 |",
        "| 显存管理 | CLIP 图像塔只在入库阶段逐图推理（无批处理）、**不做推理后释放**；"
        "查询期还要用 CLIP 文本塔（`clip_search` 把问题编码进同一向量空间），模型经"
        "启动预热后**常驻**（`_CLIP_CACHE`），是本进程显存的固定占用项 |",
        "",
        "## 五、备注",
        "",
        "- RSS/显存为**压测进程自身**的占用（`mem_snapshot()` 在客户端进程内测量），"
        "不是服务进程的占用；服务端资源需按 PID 另行采集（`/health` 只报组件可用性，"
        "不返回内存/显存）。两项都测不到时如实写「未测得」/`-`，绝不填占位数字。",
    ]
    for n in (notes or []):
        lines.append(f"- {n}")
    if explicit_budget is not None and any(t != explicit_budget for t in thresholds):
        lines.append(
            f"- 调用方传入的预算 {explicit_budget:g} ms 与部分档位记录的阈值不同："
            "本报告采用各档记录值（达标数即按它统计），未按传入值重判。")
    for r in results:
        if r.error_kinds:
            kinds = "、".join(f"{k}×{v}" for k, v in sorted(r.error_kinds.items()))
            lines.append(f"- 并发 {r.concurrency} 失败构成：{kinds}")
    if any(r.rss_mb is None or r.rss_mb <= 0 for r in results):
        lines.append("- 部分档位 RSS **未测得**：本进程无法测量内存（psutil/resource "
                     "都不可用），表中以「未测得」呈现，不是真实占用为 0")
    if not notes:
        lines.append("- 无")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path


def resolve_per_level(requested: int | None, levels: list[int]) -> tuple[int, str]:
    """定每档请求数：缺省 = 最大并发 × 3；显式值小于最大并发时上调。

    返回 ``(每档请求数, 告警文案或空串)``。请求数少于并发数时队列会被前若干线程
    瞬间抢空，其余线程立即退出，标称并发根本施加不出去 —— 此时报告若仍写「100
    并发」，等于夸大压测强度、并顺着「时延更好看」的方向失真，必须上调或告警。
    """
    ceiling = max(levels) if levels else 1
    if requested is None:
        return DEFAULT_PER_LEVEL_FACTOR * ceiling, ""
    if requested < ceiling:
        return ceiling, (f"每档请求数 {requested} < 最大并发 {ceiling}，已上调为 {ceiling}："
                         "否则该档无法真正施加标称并发（在飞请求数不足）")
    return requested, ""


def main() -> int:
    """命令行入口：``loadtest.py [url] [并发档位] [每档请求数]``。

    每档请求数缺省为最大并发 × 3（10/50/100 时为 300）；显式给的小值会被上调。
    实跑前需先启动服务（见 README/工单）：
    ``uvicorn rag04.api.server:_get_app --factory --port 8000 --app-dir src``
    """
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    levels = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2
                               else ["10", "50", "100"])]
    per_level, warning = resolve_per_level(
        int(sys.argv[3]) if len(sys.argv) > 3 else None, levels)
    if warning:
        print(f"[压测] {warning}", flush=True)

    budget = default_budget_ms()
    reports_dir = Path("docs/reports")
    try:
        from rag04.config import get_settings
        s = get_settings()
        budget = float(s.latency_budget_ms)
        reports_dir = Path(s.reports_dir)
    except Exception:                    # 脱库运行：用兜底预算与相对路径
        print("[压测] 读取 rag04.config 失败，预算与报告目录使用兜底值", flush=True)

    results: list[BenchResult] = []
    for c in levels:
        print(f"[压测] 并发 {c} …", flush=True)
        r = run_level(url, c, per_level, DEFAULT_QUESTIONS, budget_ms=budget)
        results.append(r)
        print(json.dumps(r.as_dict(), ensure_ascii=False), flush=True)

    out = write_report(results, reports_dir / "高并发压测报告.md",
                       notes=["压测期间服务未重启，模型常驻"] if results else None,
                       budget_ms=budget)
    print(f"报告：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
