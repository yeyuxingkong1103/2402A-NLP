# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG 项目-金融问答系统部署

容器 HEALTHCHECK 探针：``部署/docker/healthcheck.py``。

为什么不用 curl：``python:3.11-slim`` 不带 curl，装一个只为探活不划算（体积 + 攻击面）。
本脚本**只用标准库**，直接请求 ``/healthz``（存活探针，不碰索引，避免索引加载慢导致容器被误判不健康）。

退出码（Docker HEALTHCHECK 语义）：
    0 = healthy   非 0 = unhealthy（Docker 会标 (unhealthy) 并按 restart 策略处理）

环境变量：
    DEPLOY_HEALTHCHECK_URL      默认 ``http://127.0.0.1:8200/healthz``
    DEPLOY_HEALTHCHECK_TIMEOUT_S 默认 4
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

URL = os.environ.get("DEPLOY_HEALTHCHECK_URL", "http://127.0.0.1:8200/healthz")
TIMEOUT = float(os.environ.get("DEPLOY_HEALTHCHECK_TIMEOUT_S", "4"))


def main() -> int:
    try:
        with urllib.request.urlopen(URL, timeout=TIMEOUT) as resp:
            body = resp.read(4096).decode("utf-8", "replace")
            if resp.status != 200:
                print(f"unhealthy: HTTP {resp.status} {body[:200]}", file=sys.stderr)
                return 1
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                payload = {"raw": body[:120]}
            print(f"healthy: {json.dumps(payload, ensure_ascii=False)}")
            return 0
    except urllib.error.HTTPError as exc:
        print(f"unhealthy: HTTP {exc.code} {URL}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 —— 探针绝不抛栈，只回退出码
        print(f"unhealthy: {type(exc).__name__}: {exc} ({URL})", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
