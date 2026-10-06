"""微调报告：把**记录下来的数据**自动拼成一份详细报告（纯函数，可单测）。

为什么要它：用户要求「微调记得监控、记录数据，给我一个详细的微调报告」。
手写报告容易漏项、也容易事后找理由，所以这里把四类**原始留痕**直接拼成 markdown：

1. 数据侧：`data/sft/stats.json`（本轮生成/接受/拒绝、四个闸门各拦下多少、train/val 条数）；
2. 训练侧：`<run>/train_log.jsonl`（逐步 loss/lr/grad_norm/显存）与 `train_report.json`
   （超参、环境、可训练参数、吞吐、峰值显存、耗时）——**曲线与峰值来自实测，不是估计**；
3. 评测侧：`eval/results/<tag>.jsonl` 的逐题判分 —— 与基线逐题对比，给出**转正/转负**清单
   与四个指标差值；
4. 判定：按**预注册标准**（`docs/FINETUNE-REPORT.md` §六）给出"是否达标"，并如实写清
   噪声口径（107 题 ±3 题）。

缺哪一项就在报告里显式写「缺失（未记录）」，**不猜、不留白**。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Iterable

#: 预注册的验收标准（与 docs/FINETUNE-REPORT.md §六 一致，改这里必须同步改文档）
P8_ITEMS = ("L07", "L13", "L22", "L23")
NOISE_NOTE = "107 题的波动约 ±3 题；只有同一批题上的一致改善才算有效"
METRIC_KEYS = ("ok_rate", "source_hit_rate", "claim_coverage", "citation_fidelity", "route_accuracy")

#: 一次训练步不可能短于这个秒数；比它短的记录是"收尾/评估"打出来的，吞吐算不出来就该写 None
MIN_STEP_SECONDS = 0.05
#: 吞吐的物理上限（4090 上 4B 模型实测约 7e3 tok/s）。超过它只可能是坏记录——
#: 2026-09-28 就出现过 6.5e8 tok/s（收尾记录与上一条只差 1ms），必须挡在统计之外并留痕。
THROUGHPUT_CEILING = 1e6


class StepLogger:
    """训练步监控：把每一步的指标写成 JSONL（**torch-free**，因此本地可单测）。

    为什么要它：用户要求"微调记得监控、记录数据"。这里记录 step/epoch/loss/lr/grad_norm/
    显存/吞吐/时间戳；`scripts/train_lora.py` 里的 `TrainerCallback` 只是薄薄一层调用。

    ``vram_fn`` 是注入的显存读取函数（``() -> GB`` 或 ``None``），所以本类不依赖 torch。
    """

    def __init__(self, path: str | Path, *, batch_size: int = 1, grad_accum: int = 1,
                 avg_tokens: int = 512, vram_fn: Callable[[], float | None] | None = None,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.batch_size = max(1, int(batch_size))
        self.grad_accum = max(1, int(grad_accum))
        self.avg_tokens = max(1, int(avg_tokens))
        self.vram_fn = vram_fn
        self.clock = clock
        self.tokens_seen = 0
        self.peak_vram_gb: float | None = None
        self.points: list[dict] = []
        self._last_time = self.clock()
        self._last_tokens = 0
        self._last_step = 0

    def record(self, *, step: int | None = None, epoch: float | None = None,
               logs: dict[str, Any] | None = None,
               vram_gb: float | None = None) -> dict:
        """记一条（追加写盘并返回该条）。缺的字段写 ``None``，不猜。

        吞吐的口径（2026-09-28 修，**两次**）：**本区间的 token 数 / 本区间的秒数**。

        * 第一版是"**累计** token / 本区间秒数"，越训越大；Trainer 收尾时还会在最后一条
          训练记录后 1ms 再打一条（`train_runtime`），除出 6.5e8 tok/s。
        * 第二版改成了"1 个优化步的 token / 本区间秒数"，但一条记录实际跨
          ``logging_steps`` 个优化步（实测少了 10 倍）——所以现在按 **step 跨度** 累计：

              delta_steps  = 本条 step - 上一条 step（第一条按 1..step 算）
              delta_tokens = delta_steps × batch × grad_accum × avg_tokens
              rate         = delta_tokens ÷ 区间秒数

          只有带 ``loss`` 的记录才算训练步（Trainer 的评估/收尾记录不带 ``loss``，
          所以 `delta_steps` 为 0，不会重复计数）；间隔 < `MIN_STEP_SECONDS` 的写 ``None``。
        """
        logs = logs or {}
        now = self.clock()
        elapsed = max(now - self._last_time, 1e-6)
        self._last_time = now
        is_train_step = logs.get("loss") is not None
        delta_steps = 0
        if is_train_step and step is not None:
            # 第一条记录覆盖 1..step；之后按相邻 step 差。收尾/评估记录 step 不变 ⇒ 0。
            delta_steps = max(0, int(step) - self._last_step)
        if step is not None:
            self._last_step = max(self._last_step, int(step))
        delta_tokens = 0
        if delta_steps > 0:
            delta_tokens = delta_steps * self.batch_size * self.grad_accum * self.avg_tokens
            self.tokens_seen += delta_tokens
        if vram_gb is None and self.vram_fn is not None:
            try:
                vram_gb = self.vram_fn()
            except Exception:  # noqa: BLE001 - 读显存失败不该影响训练
                vram_gb = None
        if vram_gb is not None:
            self.peak_vram_gb = max(self.peak_vram_gb or 0.0, float(vram_gb))
        rate = None
        if delta_tokens > 0 and elapsed >= MIN_STEP_SECONDS:
            rate = round(delta_tokens / elapsed, 1)
        point = {
            "step": step,
            "epoch": epoch,
            "loss": logs.get("loss"),
            "learning_rate": logs.get("learning_rate"),
            "grad_norm": logs.get("grad_norm"),
            "eval_loss": logs.get("eval_loss"),
            "vram_gb": round(float(vram_gb), 2) if vram_gb is not None else None,
            "tokens_seen": self.tokens_seen,
            "tokens_per_second": rate,
            "delta_steps": delta_steps,
            # 单调时钟原值（首末相减 = 训练时长；绝对值本身没有意义，别当时间戳读）
            "seconds": round(now, 3),
            "interval_seconds": round(elapsed, 3),
        }
        self._last_tokens = self.tokens_seen
        self.points.append(point)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(point, ensure_ascii=False) + "\n")
        return point

    def summary(self) -> dict:
        # 每步 token 训练侧一定知道（batch × grad_accum × 平均长度）⇒ 交给 summarize_curve 精确折算
        curve = summarize_curve(
            self.points, per_step_tokens=self.batch_size * self.grad_accum * self.avg_tokens)
        curve["peak_vram_gb"] = round(self.peak_vram_gb, 2) if self.peak_vram_gb else None
        curve["tokens_seen"] = self.tokens_seen
        return curve


def token_stats(records: Iterable[dict[str, Any]]) -> dict:
    """样本长度与证据规模统计（字符数近似 token 规模；不引入 tokenizer）。"""
    lengths: list[int] = []
    evidence_counts: list[int] = []
    for record in records:
        messages = record.get("messages") or []
        answer = ""
        for message in reversed(messages):
            if message.get("role") == "assistant":
                answer = str(message.get("content") or "")
                break
        lengths.append(len(answer))
        evidence_counts.append(len(record.get("evidence") or []))
    if not lengths:
        return {"samples": 0, "answer_chars_mean": 0, "answer_chars_p95": 0,
                "evidence_mean": 0}
    ordered = sorted(lengths)
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    return {
        "samples": len(lengths),
        "answer_chars_mean": round(sum(lengths) / len(lengths), 1),
        "answer_chars_p95": p95,
        "evidence_mean": round(sum(evidence_counts) / len(evidence_counts), 2),
    }


def _percentile(values: list[float], q: float) -> float | None:
    """线性插值分位数（**不引入 numpy**：报告脚本要能在任何环境跑）。"""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    frac = position - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def throughput_estimate(run: dict | None, curve: dict | None) -> dict | None:
    """端到端吞吐**估计**：步数 × 每步 token ÷ 墙钟秒数。

    为什么不能直接读 ``train_report.json`` 的 ``tokens_per_second``：早期 run 的
    ``train_log.jsonl`` 用的是错口径（累计 token ÷ 本区间秒数），所以那个字段虚高到
    1e7 量级（实测 v5c 报 17,259,602 tok/s，物理上不可能）。留痕里**可信**的是步数、
    墙钟、平均每条 token、样本数、epoch 数；由

        batch × grad_accum ≈ train_samples × epochs ÷ steps

    反推每步 token（向上取整造成的偏差用 ``fit_residual`` 如实标出），再乘步数得总 token。
    报告里必须写明这是估计，不能冒充实测。
    """
    if not run or not curve:
        return None
    steps = curve.get("steps")
    wall = curve.get("wall_seconds_from_log") or run.get("elapsed_seconds")
    per_sample = run.get("avg_tokens_per_sample")
    samples = run.get("train_samples")
    epochs = run.get("epochs")
    if not all(isinstance(x, (int, float)) and x > 0
               for x in (steps, wall, per_sample, samples, epochs)):
        return None
    raw = samples * epochs / steps
    batch_accum = max(1, round(raw))
    total = steps * batch_accum * per_sample
    return {
        "tokens_total": round(total),
        "tokens_per_second": round(total / wall, 1),
        "batch_x_accum_inferred": batch_accum,
        "fit_residual": round(abs(raw - batch_accum), 4),
        "steps": steps,
        "wall_seconds": round(wall, 1),
        "formula": "steps × round(samples×epochs/steps) × avg_tokens_per_sample ÷ wall_seconds",
    }


def _normalize_throughput(points: list[dict[str, Any]],
                          per_step_tokens: float | None = None) -> tuple[list[dict], bool]:
    """把每一点的吞吐折算成真值，返回 (新列表, 是否改过东西)。

    **两种历史约定都要能救**（2026-09-28 一天里踩了两次）：

    * **旧**（无 ``interval_seconds``）：每记录点只累计 1 个优化步的 token，又是拿
      "累计 token ÷ 区间秒数"当吞吐 ⇒ 表里 78 → 6482 一路爬、最后 4.6e7。
    * **新但没乘 stride**（有 ``interval_seconds``，但 rate = 1 步 token ÷ 区间秒数）：
      `logging_steps=10` 时**少算 10 倍**（v5d 的日志就是这样，237 tok/s 明显不对）。

    判据不靠猜：给定 ``per_step_tokens``（= ``batch×grad_accum×avg_tokens``，训练侧一定知道）
    就精确重算

        rate = per_step_tokens × 本区间跨过的优化步数 ÷ 区间秒数

    这个式子对**两种约定都对**（新版正确日志重算后数值不变，是幂等的）。
    没给 ``per_step_tokens`` 时退回旧约定的折算；新版日志缺参数则原样不动，免得把对的改错。
    """
    has_intervals = any(isinstance(p.get("interval_seconds"), (int, float)) for p in points)
    has_token_totals = any(isinstance(p.get("tokens_seen"), (int, float)) for p in points)
    legacy = has_token_totals and not has_intervals
    if not per_step_tokens and not legacy:
        # 没有可靠依据（手写点、或新口径日志但没给每步 token）⇒ **原样保留**，
        # 绝不"折算"成 None（那会把好好的测量值抹掉；真踩过）。
        return [dict(point) for point in points], False

    normalized: list[dict[str, Any]] = []
    changed = False
    for index, point in enumerate(points):
        current = dict(point)
        if index == 0:
            # 第一条：区间 = 从 logger 构造到本条（`interval_seconds` 就是它），
            # 跨过的步数 = step（训练从 0 步开始）。有依据就折算，没有就原样留着。
            step0 = current.get("step") or 0
            interval0 = current.get("interval_seconds")
            rate0 = current.get("tokens_per_second")
            if (per_step_tokens and step0 > 0
                    and isinstance(interval0, (int, float)) and interval0 >= MIN_STEP_SECONDS):
                rate0 = round(per_step_tokens * step0 / interval0, 1)
                if rate0 != current.get("tokens_per_second"):
                    changed = True
                current["tokens_per_second"] = rate0
            normalized.append(current)
            continue
        previous = points[index - 1]
        delta_tokens = (current.get("tokens_seen") or 0) - (previous.get("tokens_seen") or 0)
        delta_steps = (current.get("step") or 0) - (previous.get("step") or 0)
        interval = (current.get("seconds") or 0) - (previous.get("seconds") or 0)
        rate = None
        if delta_steps > 0 and interval >= MIN_STEP_SECONDS:
            if per_step_tokens:
                rate = round(per_step_tokens * delta_steps / interval, 1)
            elif delta_tokens > 0:
                rate = round(delta_tokens * delta_steps / interval, 1)
        if rate != current.get("tokens_per_second"):
            changed = True
        current["tokens_per_second"] = rate
        normalized.append(current)
    return normalized, changed
    return normalized


def summarize_curve(points: list[dict[str, Any]], *,
                    per_step_tokens: float | None = None) -> dict:
    """把逐步日志折成一条可读的曲线摘要。

    除了 loss（首/末/最低）与吞吐，**还给出显存的实测统计**（用户明确要看"过程中的显存占用"）：
    ``vram_first/last/min/mean/p95/max`` 取自 `train_log.jsonl` 每一步的实测值，
    并把曲线**采样**成 ``vram_points``（默认 ≤12 个点，给报告里的时间线表格用）。

    旧口径日志（无 ``interval_seconds``）或**新口径但没乘 logging stride** 的日志都会先做吞吐折算，
    见 `_normalize_throughput`：给了 ``per_step_tokens``（训练侧一定知道）就精确重算且幂等。
    """
    has_intervals = any(isinstance(p.get("interval_seconds"), (int, float)) for p in points)
    has_token_totals = any(isinstance(p.get("tokens_seen"), (int, float)) for p in points)
    points, changed = _normalize_throughput(points, per_step_tokens) if points else (points, False)
    legacy_throughput = changed or (bool(points) and has_token_totals and not has_intervals)
    losses = [float(p["loss"]) for p in points if isinstance(p.get("loss"), (int, float))]
    raw_speeds = [float(p["tokens_per_second"]) for p in points
                  if isinstance(p.get("tokens_per_second"), (int, float))]
    # 只统计"物理上说得通"的吞吐，并把被剔除的条数**写进摘要**（不静默丢弃）
    speeds = [rate for rate in raw_speeds if 0 < rate <= THROUGHPUT_CEILING]
    vrams = [float(p["vram_gb"]) for p in points if isinstance(p.get("vram_gb"), (int, float))]
    base = {
        "points": len(points),
        "steps": points[-1].get("step") if points else None,
        "epochs": points[-1].get("epoch") if points else None,
        "loss_first": round(losses[0], 4) if losses else None,
        "loss_last": round(losses[-1], 4) if losses else None,
        "loss_min": round(min(losses), 4) if losses else None,
        "tokens_per_second_mean": round(sum(speeds) / len(speeds), 1) if speeds else None,
        "tokens_per_second_max": round(max(speeds), 1) if speeds else None,
        "throughput_points_used": len(speeds),
        "throughput_points_skipped": len(raw_speeds) - len(speeds),
        # --- 显存（实测）---
        "vram_measured_points": len(vrams),
        "vram_first": round(vrams[0], 2) if vrams else None,
        "vram_last": round(vrams[-1], 2) if vrams else None,
        "vram_min": round(min(vrams), 2) if vrams else None,
        "vram_mean": round(sum(vrams) / len(vrams), 2) if vrams else None,
        "vram_p95": round(_percentile(vrams, 0.95), 2) if vrams else None,
        "vram_max": round(max(vrams), 2) if vrams else None,
        "tokens_seen_last": points[-1].get("tokens_seen") if points else None,
        "grad_norm_last": points[-1].get("grad_norm") if points else None,
        "throughput_normalized": legacy_throughput,
    }
    # 采样时间线（报告里要能"看见过程"，但又不能把几千行日志贴进去）
    if points:
        wanted = 12
        stride = max(1, len(points) // wanted)
        sampled = points[::stride]
        if sampled[-1] is not points[-1]:
            sampled.append(points[-1])
        base["vram_points"] = [
            {"step": p.get("step"), "epoch": p.get("epoch"),
             "loss": round(float(p["loss"]), 4) if isinstance(p.get("loss"), (int, float)) else None,
             "vram_gb": round(float(p["vram_gb"]), 2) if isinstance(p.get("vram_gb"), (int, float)) else None,
             "tokens_per_second": (round(float(p["tokens_per_second"]), 1)
                                   if isinstance(p.get("tokens_per_second"), (int, float)) else None)}
            for p in sampled]
    # 墙钟：日志里的 seconds 是 perf_counter，首末相减就是训练时长
    stamps = [float(p["seconds"]) for p in points if isinstance(p.get("seconds"), (int, float))]
    if len(stamps) >= 2:
        base["wall_seconds_from_log"] = round(stamps[-1] - stamps[0], 1)
        base["seconds_per_step"] = round((stamps[-1] - stamps[0]) / max(1, len(stamps) - 1), 2)
        steps_total = base.get("steps") or 0
        # 端到端吞吐：优先用「每步 token × 步数 ÷ 墙钟」——它不依赖 tokens_seen 的累计口径，
        # 所以新旧两种日志都对（v5d 的 tokens_seen 被少算了 10 倍，只有这条式子救得回来）。
        if per_step_tokens and steps_total and base["wall_seconds_from_log"]:
            base["tokens_total_from_steps"] = round(per_step_tokens * steps_total)
            base["tokens_per_second_overall"] = round(
                per_step_tokens * steps_total / base["wall_seconds_from_log"], 1)
        elif not legacy_throughput and base.get("tokens_seen_last") and base["wall_seconds_from_log"]:
            base["tokens_per_second_overall"] = round(
                base["tokens_seen_last"] / base["wall_seconds_from_log"], 1)
    return base


def compare_eval(baseline: dict[str, dict], lora: dict[str, dict]) -> dict:
    """逐题对比基线（27B）与微调模型（LoRA）的判分。

    ``baseline``/``lora`` 形如 ``{"L07": {"ok": False, "expect": "answerable", ...}}``。
    """
    from .eval_set import summarize

    base_ids, lora_ids = set(baseline), set(lora)
    shared = sorted(base_ids & lora_ids)
    base_results = [baseline[key] for key in shared]
    lora_results = [lora[key] for key in shared]

    turned_green = [key for key in shared
                    if not baseline[key].get("ok") and lora[key].get("ok")]
    turned_red = [key for key in shared
                  if baseline[key].get("ok") and not lora[key].get("ok")]
    p8_green = [key for key in turned_green if key in P8_ITEMS]
    p8_rank = {key: {"baseline": baseline.get(key, {}).get("ok"),
                     "lora": lora.get(key, {}).get("ok")} for key in P8_ITEMS if key in shared}

    base_summary = summarize(base_results) if base_results else {}
    lora_summary = summarize(lora_results) if lora_results else {}
    deltas = {key: round((lora_summary.get(key) or 0) - (base_summary.get(key) or 0), 4)
              for key in METRIC_KEYS}
    return {
        "questions": len(shared),
        "only_in_baseline": sorted(base_ids - lora_ids),
        "only_in_lora": sorted(lora_ids - base_ids),
        "baseline": {key: base_summary.get(key) for key in METRIC_KEYS},
        "lora": {key: lora_summary.get(key) for key in METRIC_KEYS},
        "deltas": deltas,
        "turned_green": turned_green,
        "turned_red": turned_red,
        "p8_turned_green": p8_green,
        "p8_status": p8_rank,
    }


def verdict(comparison: dict) -> dict:
    """按**预注册标准**判定（不做事后放宽）。"""
    checks: list[dict] = []
    checks.append({
        "name": "P8 相关题至少 3 道转正",
        "passed": len(comparison.get("p8_turned_green") or []) >= 3,
        "detail": f"转正 {comparison.get('p8_turned_green')}",
    })
    checks.append({
        "name": "四个答案侧指标不倒退",
        "passed": all((comparison.get("deltas") or {}).get(key, 0) >= 0
                      for key in ("source_hit_rate", "claim_coverage", "citation_fidelity",
                                  "route_accuracy")),
        "detail": str(comparison.get("deltas")),
    })
    checks.append({
        "name": "引用可信度保持 1.0",
        "passed": (comparison.get("lora") or {}).get("citation_fidelity") == 1.0,
        "detail": f"lora={((comparison.get('lora') or {}).get('citation_fidelity'))}",
    })
    return {"passed": all(check["passed"] for check in checks), "checks": checks,
            "noise_note": NOISE_NOTE}


def _env_of(run: dict) -> dict:
    """取环境信息：训练器把它写在**嵌套**的 ``report["env"]`` 里。

    ⚠️ 这里曾经读 ``run.get("torch")``（顶层）⇒ 报告里环境那一行一直打印 ``None``
    （2026-09-28 生成报告时才发现）。现在两者都认，老报告也不会丢。
    """
    env = run.get("env")
    if isinstance(env, dict) and env:
        return env
    return {key: run.get(key) for key in ("torch", "transformers", "peft", "gpu")}


def render_vram_section(curve: dict | None, run: dict | None, *,
                        vram_total_gb: float | None = None,
                        inference: list[dict] | None = None) -> list[str]:
    """**显存与吞吐**小节：过程里的实测显存（用户明确要求要看这个）。

    ``inference`` 是推理侧的占用（每项 ``{"name","vram_gb"}``），单独列，
    免得把"训练占多少"与"服务占多少"混成一个数。``vram_total_gb`` 给了就算百分比。
    """
    lines: list[str] = ["", "## 三、显存与吞吐（实测留痕）", ""]
    if not curve and not run:
        lines.append("- 缺失（未记录训练留痕）")
        return lines
    if curve:
        lines += [
            f"- 实测显存（`train_log.jsonl`，{curve.get('vram_measured_points')} 个记录点）："
            f"首 {curve.get('vram_first')} / 均 **{curve.get('vram_mean')}** / p95 "
            f"{curve.get('vram_p95')} / **峰 {curve.get('vram_max')}** / 末 {curve.get('vram_last')} GB"
            + (f"（GPU 总显存 {vram_total_gb} GB ⇒ 训练峰值占 "
               f"{round(100 * (curve.get('vram_max') or 0) / vram_total_gb, 1)}%）"
               if vram_total_gb else ""),
            f"- 吞吐：平均 {curve.get('tokens_per_second_mean')} tok/s；"
            f"日志跨度 {curve.get('wall_seconds_from_log')}s / "
            f"{curve.get('seconds_per_step')}s 每记录点；"
            f"累计 token {curve.get('tokens_seen_last')}",
        ]
        timeline = curve.get("vram_points") or []
        if timeline:
            lines += ["", "| step | epoch | loss | 显存 GB | tok/s |", "|---|---|---|---|---|"]
            for point in timeline:
                lines.append(f"| {point.get('step')} | "
                             f"{round(float(point['epoch']), 2) if point.get('epoch') is not None else '-'} | "
                             f"{point.get('loss')} | {point.get('vram_gb')} | "
                             f"{point.get('tokens_per_second')} |")
            lines += ["", "> 上表是**采样**（每份日志最多 12 行）；完整逐步数据见 "
                          "`train_log.jsonl`。"]
    if run:
        lines.append(f"- 训练器记录的峰值显存：{run.get('peak_vram_gb')} GB；"
                     f"训练总耗时 {run.get('elapsed_seconds')}s")
    if inference:
        lines += ["", "### 推理侧显存（同一张卡，供对照）", "",
                  "| 服务 | 显存 GB |", "|---|---|"]
        for item in inference:
            lines.append(f"| {item.get('name')} | {item.get('vram_gb')} |")
        lines.append("")
        lines.append("> 训练要**先停掉 27B 服务**腾显存（它自己就占 ~40 GB）；"
                     "训完再起回来 —— 这段『停机-训练-重启』是本项目的固定流程。")
    return lines


def summarize_scores(scores: dict[str, dict]) -> dict:
    """把一份逐题判分汇总成指标（``legal_rag.eval_set.summarize`` 的薄封装）。"""
    from .eval_set import summarize

    return summarize(list(scores.values())) if scores else {}


def _error_reason(error: str) -> str:
    """把一坨 HTTP 错误正文压成一句能读的原因（报告里要能一眼看懂）。

    例：``HTTP 503: {"detail":"请求被大模型服务拒绝（bad_request）。这次没有生成回答…"}``
    → ``HTTP 503 请求被大模型服务拒绝（bad_request）``
    """
    text = error.strip()
    marker = '"detail":"'
    if marker in text:
        text = text.split(marker, 1)[1]
    text = text.split("。", 1)[0].split("，", 1)[0].strip().strip('"').rstrip('"}').strip()
    prefix = error.split(":", 1)[0].strip()
    if prefix and prefix not in text:
        text = f"{prefix} {text}"
    return text[:70]


def arm_quality(records: list[dict]) -> dict:
    """一条臂的**过程质量**：空答案与错误各多少、都是什么错误。

    为什么要单独统计：判分只看四指标，**空回复**只会记成"答错了"。而空回复/错误是
    线上真实故障形态（本例：G10 因 `max_model_len` 不够被 vLLM 拒），
    报告里必须能一眼看见"这条 FAIL 是模型答错，还是根本没答"。
    """
    empty = 0
    errors: dict[str, int] = {}
    for record in records:
        answer = str(record.get("answer") or "")
        if not answer.strip():
            empty += 1
        error = str(record.get("error") or "")
        if error:
            reason = _error_reason(error)
            errors[reason] = errors.get(reason, 0) + 1
    return {"rows": len(records), "empty": empty, "errors": errors,
            "errors_total": sum(errors.values())}


def arm_table(arms: dict[str, dict[str, dict]], *, baseline: str = "",
              noise_items: float = 4.0,
              quality: dict[str, dict] | None = None) -> dict:
    """各臂指标表 + 相对基线的差值（**用题数表达**，并判定是否小于抖动地板）。

    为什么强调题数：本项目实测过"同一条件重复测量"的抖动是 **4 题 / 3.7 分**
    （107 题）。只看百分比很容易把 1–2 题的差别当成"提升"。
    """
    order = [name for name in arms]
    if baseline and baseline in order:
        order = [baseline] + [name for name in order if name != baseline]
    summaries = {name: summarize_scores(arms[name]) for name in order}
    n = max((len(arms[name]) for name in order), default=0)
    rows = []
    for name in order:
        summary = summaries[name]
        delta_items = None
        verdict_text = "（基线）"
        if baseline and name != baseline and baseline in summaries:
            base_rate = summaries[baseline].get("ok_rate") or 0.0
            rate = summary.get("ok_rate") or 0.0
            delta_items = round((rate - base_rate) * n, 1)
            if abs(delta_items) < noise_items:
                verdict_text = f"Δ{delta_items:+} 题 < 抖动地板 {noise_items:g} ⇒ **不可辨**"
            elif delta_items > 0:
                verdict_text = f"Δ{delta_items:+} 题 ⇒ 超过抖动地板（可能是真的）"
            else:
                verdict_text = f"Δ{delta_items:+} 题 ⇒ 低于抖动地板（变差可能为真）"
        rows.append({
            "tag": name, "n": len(arms[name]),
            "ok_rate": summary.get("ok_rate"),
            "source_hit_rate": summary.get("source_hit_rate"),
            "claim_coverage": summary.get("claim_coverage"),
            "citation_fidelity": summary.get("citation_fidelity"),
            "route_accuracy": summary.get("route_accuracy"),
            "empty_answers": (quality or {}).get(name, {}).get("empty"),
            "errors": (quality or {}).get(name, {}).get("errors_total"),
            "delta_items_vs_baseline": delta_items, "verdict": verdict_text,
        })
    return {"baseline": baseline, "n": n, "noise_items": noise_items,
            "order": order, "rows": rows, "summaries": summaries}


def render_fix_sweep(sweep: dict | None) -> list[str]:
    """「修好之后，`max_len` 该开多大」的实测扫描表。

    为什么必须有这一节：光说"修好了"没用，得给出**能训练的那个窗口**。
    判据与 `scripts/audit_sft_tokens.py` 的闸门一致（证据被裁 >10% ⇒ 拒绝；
    监督被吃 >5% ⇒ 拒绝）。
    """
    if not sweep:
        return []
    lines = ["", "**修复后扫描（当前实现：只裁证据、保留问题与答案）**", "",
             "| max_len | 证据被裁样本 | 只剩 <=1 监督 token | 监督 token 中位 | 可装下的样本 "
             "| 闸门判定 |", "|---|---|---|---|---|---|"]
    for key in sorted(sweep, key=lambda value: int(value)):
        item = sweep[key]
        rows = item.get("rows") or 1
        trimmed = item.get("prompt_trimmed_rows")
        starved = item.get("supervised_starved_rows")
        fit = item.get("fit_in_rows") or {}
        fit_text = ("/".join(f"{fit.get(k)}" for k in ("2048", "4096", "8192"))
                    if fit else "-")
        lines.append(
            f"| {key} | {trimmed}（{round(100 * (item.get('prompt_trimmed_share') or 0), 1)}%） | "
            f"{starved}（{round(100 * (starved or 0) / rows, 1)}%） | "
            f"{item.get('supervised_median')} | {fit_text} | {item.get('gate')} |")
    lines += ["", "> 读法：修好之后**答案不再被切**（监督中位数从 1 升到 "
                  f"{(sweep.get('2048') or {}).get('supervised_median')}），代价是长提示要裁证据 ⇒ "
                  "窗口越大裁得越少。**重训前先跑这张表**：闸门给 exit 0 的那个 `max_len` 才能用。"]
    return lines


def render_supervision_audit(audit: dict | None) -> list[str]:
    """「训练信号有没有被截断吃掉」的实测表（2026-09-28 事故的核心证据）。

    这一节存在的理由：数据、超参、显存全都正常，但 912 条里 514 条只剩 ≤1 个监督
    token —— 不看这一眼就永远找不到"为什么微调没效果"。
    """
    if not audit:
        return []
    legacy = any(isinstance(item, dict) and item.get("legacy_right_truncate")
                 for item in audit.values())
    title = ("**训练信号审计（真 tokenizer 实测；**旧实现·右截断**，复现 2026-09-28 事故）**"
             if legacy else "**训练信号审计（真 tokenizer 实测）**")
    lines = ["", title, "",
             "| 数据 | 行数 | 提示 token 均值 | p95 | 提示超 max_len | 只剩 <=1 监督 token "
             "| 监督 token 中位 | max_len |",
             "|---|---|---|---|---|---|---|---|"]
    for name, item in audit.items():
        if not isinstance(item, dict) or not item.get("present", True) or not item.get("rows"):
            continue
        rows = item["rows"]
        over = item.get("prompt_over_maxlen")
        starved = item.get("supervised_starved_rows")
        lines.append(
            f"| `{name}` | {rows} | {item.get('prompt_tokens_mean')} | "
            f"{item.get('prompt_tokens_p95')} | "
            f"{over}（{round(100 * over / rows, 1)}%） | "
            f"{starved}（{round(100 * starved / rows, 1)}%） | "
            f"{item.get('supervised_median')} | {item.get('max_len')} |")
    lines += ["", "> 读法：**提示超 max_len 的样本会被截断**。旧实现是右截断（`full_ids[:max_len]`），"
                  "提示在长、答案在尾 ⇒ 答案被切光，只剩 1 个监督 token。"
                  "`train.legacy.jsonl` 提示只有 17.7 token 所以一条都没被切；"
                  "换成「证据进提示」的同形数据后提示涨到 3000+ token，同一个 `max_len` 就把监督吃掉了。"]
    return lines


def render_suite_report(*, tag: str, runs: dict[str, tuple[dict | None, dict | None]],
                        arms: dict[str, dict[str, dict]], baseline: str = "",
                        noise_items: float = 4.0, env: dict | None = None,
                        headline: list[str] | None = None,
                        data: dict | None = None, data_stats: dict | None = None,
                        data_files: list[dict] | None = None,
                        data_notes: list[str] | None = None,
                        audit: dict | None = None,
                        fix_sweep: dict | None = None,
                        eval_notes: list[str] | None = None,
                        quality: dict[str, dict] | None = None,
                        inference: list[dict] | None = None,
                        reproduce: list[str] | None = None) -> str:
    """**汇总**报告：一次把多个训练 run 与多个评测臂写进同一份 markdown。

    ``runs`` 是 ``{tag: (train_report, curve)}``；``arms`` 是 ``{tag: {id: score}}``；
    ``data_files`` 是**实测**的数据文件表（行数/正负/提示是否含证据），优先于 ``data``
    里的摘要——因为摘要文件可能来自**另一次**运行（本项目就踩过：``stats.json`` 描述的是
    没被训练用到的那一次渲染）。
    """
    lines: list[str] = [f"# 微调过程报告 · {tag}", "",
                        "> 本报告由 `scripts/report_finetune.py` 从**原始留痕自动生成**："
                        "数据统计、逐步训练日志（loss/显存/吞吐）、逐题评测结果。"
                        "缺项一律写「缺失（未记录）」，不猜。", ""]
    if headline:
        lines += ["## 结论（先看这里）", ""]
        lines += [f"- {item}" for item in headline]
        lines.append("")

    lines += ["## 〇、环境与硬件（实测）", ""]
    if env:
        for key, value in env.items():
            lines.append(f"- {key}：{value}")
    else:
        lines.append("- 缺失（未提供 `--env-json`）")
    if inference:
        lines += ["", "| 推理服务 | 显存 GB |", "|---|---|"]
        for item in inference:
            lines.append(f"| {item.get('name')} | {item.get('vram_gb')} |")

    lines += ["", "## 一、数据（自蒸馏 + 护栏 + 自检）", ""]
    if data_files:
        lines += ["**实测**（回到 jsonl 里数出来的，不是抄摘要文件）：", "",
                  "| 文件 | 行数 | 正/负样本 | 提示含证据 | 平均提示字符 | 平均答案字符 |",
                  "|---|---|---|---|---|---|"]
        for entry in data_files:
            kinds = entry.get("kinds") or {}
            kind_text = (f"{kinds.get('positive', 0)}/{kinds.get('negative', 0)}"
                         if isinstance(kinds, dict) else str(kinds))
            rows = entry.get("rows")
            evidence = entry.get("with_evidence_in_prompt")
            lines.append(
                f"| `{entry.get('name')}` | {rows} | {kind_text} | "
                f"{evidence}/{rows if rows is not None else '?'} | "
                f"{entry.get('user_chars_mean')} | {entry.get('answer_chars_mean')} |")
        lines.append("")
    if data:
        lines += [
            f"- 摘要文件 `stats.json`：pending {data.get('pending')} / 接受 {data.get('accepted')} / "
            f"拒绝 {data.get('rejected')} / 端点错误 {data.get('errors')}",
            f"- 摘要文件声称 训练/验证 = **{data.get('train')} / {data.get('val')}**；"
            f"去重丢弃 {data.get('deduped')}；语义闸门丢弃 {data.get('semantic_dropped')}",
            f"- 提示形状：**{data.get('prompt_shape')}**；"
            f"摘要声称训练集里含证据 {data.get('train_with_evidence_in_prompt')} 条",
        ]
        negatives = data.get("negatives") or {}
        if negatives:
            lines.append(f"- 负样本（摘要）：{negatives}")
    elif not data_files:
        lines.append("- 缺失（未记录 `data/sft/stats.json`）")
    if data_notes:
        lines += ["", "**留痕冲突与说明（不掩盖、不猜）**：", ""]
        lines += [f"- {note}" for note in data_notes]
    lines += render_supervision_audit(audit)
    lines += render_fix_sweep(fix_sweep)
    if data_stats:
        lines.append(f"- 答案长度：均值 {data_stats.get('answer_chars_mean')} 字 / "
                     f"p95 {data_stats.get('answer_chars_p95')} 字")

    lines += ["", "## 二、训练留痕（每个 run 一节）", ""]
    if not runs:
        lines.append("- 缺失（未提供 `--run`）")
    for run_tag, (run, curve) in runs.items():
        lines += [f"### {run_tag}", ""]
        if run:
            env_of_run = _env_of(run)
            lora = run.get("lora") or {}
            estimate = throughput_estimate(run, curve)
            lines += [
                f"- 数据：{run.get('train_samples')} 训练 / {run.get('val_samples')} 验证；"
                f"每条平均 **{run.get('avg_tokens_per_sample')}** token"
                f"（受监督 {run.get('avg_supervised_tokens')}）",
                f"- 超参：epochs={run.get('epochs')} lr={run.get('lr')} "
                f"r={lora.get('r')} α={lora.get('alpha')} dropout={lora.get('dropout')} "
                f"max_len={run.get('max_len')}",
                f"- 可训练参数 {run.get('trainable_params')} / {run.get('total_params')}"
                + (f"（{round(100 * run['trainable_params'] / run['total_params'], 3)}%）"
                   if run.get("trainable_params") and run.get("total_params") else ""),
                f"- 耗时 **{run.get('elapsed_seconds')}s**；峰值显存 "
                f"**{run.get('peak_vram_gb')} GB**；train_loss "
                f"{round(float(run['train_loss']), 4) if isinstance(run.get('train_loss'), (int, float)) else run.get('train_loss')}",
                f"- 环境：torch {env_of_run.get('torch')} / transformers "
                f"{env_of_run.get('transformers')} / peft {env_of_run.get('peft')} / "
                f"GPU {env_of_run.get('gpu')}",
            ]
            if estimate:
                lines.append(
                    f"- 端到端吞吐**估计** {estimate['tokens_per_second']} tok/s"
                    f"（总 {estimate['tokens_total']} token ÷ {estimate['wall_seconds']}s；"
                    f"每步 batch×grad_accum={estimate['batch_x_accum_inferred']}"
                    f"，拟合残差 {estimate['fit_residual']}；"
                    f"公式 {estimate['formula']}）")
        else:
            lines.append("- 缺失（未记录 `train_report.json`）")
        if curve:
            normalized = curve.get("throughput_normalized")
            throughput = (f"吞吐：逐步均 {curve.get('tokens_per_second_mean')} tok/s"
                          f"（用 {curve.get('throughput_points_used')} 个点"
                          f"{('，另 ' + str(curve.get('throughput_points_skipped')) + ' 个点数值不合理已剔除') if curve.get('throughput_points_skipped') else ''}）"
                          + ("；**旧口径已按 step 跨度折算**（原始 tokens_seen 低估约一个 logging 周期）"
                             if normalized else ""))
            lines += [
                f"- 显存（{curve.get('vram_measured_points')} 点实测）：均 "
                f"{curve.get('vram_mean')} / p95 {curve.get('vram_p95')} / "
                f"峰 **{curve.get('vram_max')}** GB",
                f"- {throughput}；每记录点 {curve.get('seconds_per_step')}s",
                "",
                "| step | epoch | loss | 显存 GB | tok/s |",
                "|---|---|---|---|---|",
            ]
            for point in curve.get("vram_points") or []:
                epoch = point.get("epoch")
                rate = point.get("tokens_per_second")
                lines.append(f"| {point.get('step')} | "
                             f"{round(float(epoch), 2) if epoch is not None else '-'} | "
                             f"{point.get('loss')} | {point.get('vram_gb')} | "
                             f"{rate if rate is not None else '-'} |")
        lines.append("")

    lines += ["## 三、评测（同一套题，各臂独立会话）", ""]
    if arms:
        table = arm_table(arms, baseline=baseline, noise_items=noise_items, quality=quality)
        has_quality = any(row.get("empty_answers") is not None for row in table["rows"])
        lines += [f"题数 **{table['n']}**；抖动地板 **{noise_items:g} 题**"
                  f"（本项目 107 题同条件重复测量的实测抖动 ≈ 4 题 / 3.7 分）", "",
                  ("| 臂 | 通过率 | 来源命中 | 要点覆盖 | 引用保真 | 分流 | 空答案 | 错误 "
                   "| 相对基线 | 判读 |" if has_quality else
                   "| 臂 | 通过率 | 来源命中 | 要点覆盖 | 引用保真 | 分流 | 相对基线 | 判读 |"),
                  ("|---|---|---|---|---|---|---|---|---|---|" if has_quality else
                   "|---|---|---|---|---|---|---|---|")]
        for row in table["rows"]:
            delta = row["delta_items_vs_baseline"]
            delta_text = "—" if delta is None else f"{delta:+} 题"
            common = (f"| **{row['tag']}** | {row['ok_rate']} | {row['source_hit_rate']} | "
                      f"{row['claim_coverage']} | {row['citation_fidelity']} | "
                      f"{row['route_accuracy']} | ")
            if has_quality:
                common += (f"{row.get('empty_answers')} | {row.get('errors')} | ")
            lines.append(common + f"{delta_text} | {row['verdict']} |")
        lines += ["", "> 判读规则：**只看题数差**，差值小于抖动地板的一律标『不可辨』——"
                      "本项目曾把 1–2 题的差别当成「提升」来过，后来量出抖动地板才发现站不住。"]
        if has_quality:
            quality_notes = []
            for tag, item in (quality or {}).items():
                for reason, count in (item.get("errors") or {}).items():
                    quality_notes.append(f"`{tag}`：{count} 条 —— {reason}")
            if quality_notes:
                lines += ["", "**过程失败明细（有原文，不是「答错」）**：", ""]
                lines += [f"- {note}" for note in quality_notes]
        if eval_notes:
            lines += [""] + [f"> {note}" for note in eval_notes]
    else:
        lines.append("- 缺失（未提供 `--arm`）")

    lines += ["", "## 四、复现命令", ""]
    if reproduce:
        lines += ["```bash"] + list(reproduce) + ["```"]
    else:
        lines.append("- 缺失（未提供 `--reproduce`）")
    return "\n".join(lines) + "\n"


def render_report(*, tag: str, data: dict | None, run: dict | None, curve: dict | None,
                  comparison: dict | None, data_stats: dict | None = None,
                  env_extra: dict | None = None, inference: list[dict] | None = None) -> str:
    """拼出报告 markdown。缺项显式写「缺失（未记录）」。"""
    lines: list[str] = [f"# 微调报告 · {tag}", ""]

    lines += ["## 一、数据（自蒸馏 + 四关护栏）", ""]
    if data:
        lines += [
            f"- 处理证据块 {data.get('pending')} 个：**接受 {data.get('accepted')}** / "
            f"拒绝 {data.get('rejected')} / 端点错误 {data.get('errors')}",
            f"- 去重丢弃 {data.get('deduped')}；**语义闸门**丢弃 {data.get('semantic_dropped')}",
            f"- 训练/验证 = **{data.get('train')} / {data.get('val')}**",
            f"- 防泄漏基准：评测问题 {data.get('eval_questions')} 条；教师 {data.get('teacher')}",
            f"- 耗时 {data.get('elapsed_seconds')}s；随机种子 {data.get('seed')}",
        ]
        if data_stats:
            lines.append(f"- 答案长度：均值 {data_stats.get('answer_chars_mean')} 字 / "
                         f"p95 {data_stats.get('answer_chars_p95')} 字；每条证据 "
                         f"{data_stats.get('evidence_mean')} 段")
    else:
        lines.append("- 缺失（未记录：`data/sft/stats.json` 不存在）")

    lines += ["", "## 二、训练（监控留痕）", ""]
    if run:
        env = _env_of(run)
        lines += [
            f"- 基座 `{run.get('base_model')}`；训练/验证样本 "
            f"{run.get('train_samples')} / {run.get('val_samples')}",
            f"- 超参：epochs={run.get('epochs')} lr={run.get('lr')} "
            f"LoRA(r={ (run.get('lora') or {}).get('r') }, α={ (run.get('lora') or {}).get('alpha') }, "
            f"dropout={ (run.get('lora') or {}).get('dropout') }) max_len={run.get('max_len')}",
            f"- 可训练参数 {run.get('trainable_params')} / 总参数 {run.get('total_params')}"
            + (f"（占 {round(100 * run['trainable_params'] / run['total_params'], 3)}%）"
               if run.get("trainable_params") and run.get("total_params") else ""),
            f"- 每条样本平均 token **{run.get('avg_tokens_per_sample')}**"
            f"（其中受监督 {run.get('avg_supervised_tokens')}）"
            "—— ⚠️ 这个数**能反映证据有没有进提示**："
            "本项目曾因它只有 79.6 而查出『证据从未进提示』的数据事故（修好后 1700+）",
            f"- 耗时 {run.get('elapsed_seconds')}s；峰值显存 {run.get('peak_vram_gb')} GB；"
            f"吞吐 {run.get('tokens_per_second')} tok/s",
            f"- 环境：torch {env.get('torch')} / transformers {env.get('transformers')} / "
            f"peft {env.get('peft')} / GPU {env.get('gpu')}",
        ]
        if env_extra:
            lines.append("- 补充环境：" + "；".join(f"{k}={v}" for k, v in env_extra.items()))
    else:
        lines.append("- 缺失（未记录：`train_report.json` 不存在）")
    if curve:
        lines += [
            f"- loss 曲线（{curve.get('points')} 个记录点）：首 {curve.get('loss_first')} → "
            f"末 {curve.get('loss_last')}，最低 {curve.get('loss_min')}；"
            f"步数 {curve.get('steps')}；平均吞吐 {curve.get('tokens_per_second_mean')} tok/s",
        ]

    lines += render_vram_section(curve, run,
                                 vram_total_gb=(env_extra or {}).get("gpu_total_gb"),
                                 inference=inference)

    lines += ["", "## 四、评测对比（同一套题）", ""]
    if comparison:
        lines += [
            f"- 可比题数 **{comparison.get('questions')}**"
            + (f"（仅基线有：{comparison.get('only_in_baseline')}；"
               f"仅微调有：{comparison.get('only_in_lora')}）"
               if comparison.get("only_in_baseline") or comparison.get("only_in_lora") else ""),
            "",
            "| 指标 | 基线(27B) | 微调(LoRA) | Δ |",
            "|---|---|---|---|",
        ]
        for key in METRIC_KEYS:
            lines.append(f"| {key} | {comparison['baseline'].get(key)} | "
                         f"{comparison['lora'].get(key)} | {comparison['deltas'].get(key):+} |")
        lines += [
            "",
            f"- **转正**（基线不过→微调过）：{comparison.get('turned_green') or '无'}",
            f"- **转负**（基线过→微调不过）：{comparison.get('turned_red') or '无'}",
            f"- P8 相关题：{comparison.get('p8_status')}",
        ]
    else:
        lines.append("- 缺失（未记录：需要 `v5-base-*` 与 `v4-lora` 两份逐题结果）")

    lines += ["", "## 四、判定（预注册标准）", ""]
    if comparison:
        decision = verdict(comparison)
        for check in decision["checks"]:
            lines.append(f"- [{'✅' if check['passed'] else '❌'}] {check['name']} —— {check['detail']}")
        lines += ["", f"**结论：{'达标' if decision['passed'] else '未达标'}**", "",
                  f"> ⚠️ {decision['noise_note']}"]
    else:
        lines.append("- 缺失（无评测对比，无法判定）")

    lines += ["", "---", "",
              "本报告由 `scripts/report_finetune.py` 从**原始留痕**自动生成；"
              "数据来源：`data/sft/stats.json`、`<run>/train_report.json`、`<run>/train_log.jsonl`、"
              "`eval/results/*.jsonl`。"]
    return "\n".join(lines) + "\n"
