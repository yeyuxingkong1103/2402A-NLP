# -*- coding: utf-8 -*-
"""流式问答首字延迟测量脚本（JMeter 只能测整条响应的总耗时，
SSE 的首字到达时间需要逐事件读取，用本脚本单独测）。

测量口径：
    T0 = 请求发出时刻
    T1 = 收到首个 `event: token` 行的时刻      -> 首字延迟 = T1 - T0
    T2 = 收到 `event: message_end` 行的时刻    -> 完整耗时 = T2 - T0
（JMeter 侧 elapsed ≈ 完整耗时，两者可交叉验证。）

用法（在 backend 环境里，requests 已安装；登录与令牌二选一）：
    python scripts/loadtest/first_token_latency.py \
        --host 127.0.0.1 --port 8000 \
        --email you@qq.com --password-env LOADTEST_PASSWORD \
        --n 10 --question "公司解除劳动合同需要提前多久通知？"
    # 或跳过登录，直接复用已有令牌（如从 Postman 登录响应里拿）：
    LOADTEST_TOKEN=<token> python scripts/loadtest/first_token_latency.py \
        --host 127.0.0.1 --port 8000 --token-env LOADTEST_TOKEN --n 10

输出：逐次 (首字延迟, 完整耗时) + 汇总 P50/P95/平均。
密码/令牌只从环境变量读取，不落任何文件、不打印。
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SSE 首字延迟测量")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    auth_group = parser.add_mutually_exclusive_group(required=True)
    auth_group.add_argument("--email", help="已注册账号（配合 --password-env 走登录）")
    auth_group.add_argument("--token-env", help="直接从该环境变量读已有令牌（跳过登录，适合复用 Postman 登录结果）")
    parser.add_argument("--password-env", default="LOADTEST_PASSWORD",
                        help="存放密码的环境变量名（默认 LOADTEST_PASSWORD，配合 --email）")
    parser.add_argument("--n", type=int, default=10, help="压测次数")
    parser.add_argument("--question", default="公司解除劳动合同需要提前多久通知？")
    parser.add_argument("--timeout", type=int, default=180, help="单次完整耗时上限（秒）")
    return parser.parse_args()


def login(base: str, email: str, password: str) -> str:
    resp = requests.post(
        f"{base}/api/v1/auth/login",
        json={"email": email, "password": password},
        timeout=15,
    )
    resp.raise_for_status()
    token = resp.json()["data"]["access_token"]
    return token


def measure_once(base: str, token: str, question: str, timeout: int) -> tuple[float, float]:
    """发起一次流式问答，返回 (首字延迟秒, 完整耗时秒)。"""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    # session_id 每次随机：服务端对未建档会话按首问自动建档，无需提前创建
    payload = {
        "session_id": f"ftl-{os.getpid()}-{time.time_ns()}",
        "character_id": "legal-assistant",
        "message": question,
        "options": {"top_k": 8, "enable_query_rewrite": True, "enable_long_term_memory": False},
    }
    t0 = time.perf_counter()
    first_token: float | None = None
    with requests.post(f"{base}/api/v1/chat/stream", json=payload,
                       headers=headers, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if raw is None:
                continue
            line = raw.strip()
            if first_token is None and line == "event: token":
                first_token = time.perf_counter() - t0
            if line == "event: message_end":
                full = time.perf_counter() - t0
                if first_token is None:
                    raise RuntimeError("流正常结束但未收到任何 token 事件（落零引用档或护栏替换）")
                return first_token, full
    raise RuntimeError("连接结束但未收到 message_end（超时或异常中断）")


def percentile(values: list[float], pct: float) -> float:
    """线性插值百分位（与 numpy.percentile 默认口径一致，不引依赖）。"""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    low, high = int(rank), min(int(rank) + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def main() -> int:
    args = parse_args()
    base = f"http://{args.host}:{args.port}"
    if args.token_env:
        token = os.environ.get(args.token_env, "")
        if not token:
            print(f"[失败] 环境变量 {args.token_env} 未设置", file=sys.stderr)
            return 1
        print(f"[OK] 使用环境变量 {args.token_env} 提供的令牌")
    else:
        password = os.environ.get(args.password_env, "")
        if not password:
            print(f"[失败] 环境变量 {args.password_env} 未设置（密码不落盘，请先 export）", file=sys.stderr)
            return 1
        token = login(base, args.email, password)
    print(f"[OK] 开始测量 {args.n} 次：{base}")

    firsts: list[float] = []
    fulls: list[float] = []
    for i in range(1, args.n + 1):
        try:
            first, full = measure_once(base, token, args.question, args.timeout)
            firsts.append(first)
            fulls.append(full)
            print(f"  #{i:02d}  首字 {first*1000:8.0f} ms   完整 {full*1000:8.0f} ms")
        except Exception as exc:  # 单次失败不中断整轮，最后汇总报告
            print(f"  #{i:02d}  失败：{exc}")

    if not firsts:
        print("[失败] 全部请求失败", file=sys.stderr)
        return 1
    print()
    print(f"成功 {len(firsts)}/{args.n} 次")
    print(f"首字延迟  P50={percentile(firsts,50)*1000:.0f}ms  P95={percentile(firsts,95)*1000:.0f}ms  "
          f"平均={statistics.mean(firsts)*1000:.0f}ms")
    print(f"完整耗时  P50={percentile(fulls,50)*1000:.0f}ms  P95={percentile(fulls,95)*1000:.0f}ms  "
          f"平均={statistics.mean(fulls)*1000:.0f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
