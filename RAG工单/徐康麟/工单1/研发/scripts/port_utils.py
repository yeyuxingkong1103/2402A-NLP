"""本地服务端口检查工具。

## 为什么需要

`研发/scripts/run_local_llm.py` 与 `run_local_asr.py` 都要先加载模型、
再绑定端口。如果端口已被占用，会**先白等十几秒加载模型，最后才在 bind 处报错**：

    OSError: [Errno 10048] error while attempting to bind on address ('127.0.0.1', 8000)

更好的做法是在加载模型**之前**就检查端口，并且区分三种情况：

1. **空闲** —— 正常启动；
2. **已被本项目同类服务占用** —— 说明服务已经在跑，直接提示并正常退出（退出码 0），
   而不是把它当成错误；
3. **被其它程序占用** —— 明确报错并给出占用进程与两种解决办法（换端口 / 停掉占用者）。

## 用法

    from port_utils import ensure_port_available
    ensure_port_available("127.0.0.1", 8000, service_name="LLM 推理服务",
                          probe_path="/v1/models", force=False)
"""

from __future__ import annotations

import json
import socket
import sys
import urllib.error
import urllib.request


def _listening_pid(port: int) -> int | None:
    """返回监听该端口的进程号（仅 Windows；其它平台返回 None）。"""
    if sys.platform != "win32":
        return None
    import subprocess

    try:
        output = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout
    except Exception:
        return None
    for line in output.splitlines():
        parts = line.split()
        # 形如：TCP  127.0.0.1:8000  0.0.0.0:0  LISTENING  8824
        if len(parts) >= 5 and parts[3] == "LISTENING" and parts[1].endswith(f":{port}"):
            try:
                return int(parts[4])
            except ValueError:
                continue
    return None


def _probe_our_service(host: str, port: int, probe_path: str, timeout: float = 3.0) -> str | None:
    """探测该端口上是否已有**本项目**的同类服务。

    Returns:
        模型名（探测成功时）；否则 ``None``。
    """
    url = f"http://{host}:{port}{probe_path}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            if response.status != 200:
                return None
            payload = json.loads(response.read().decode("utf-8", "ignore"))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None
    # 本项目的服务都返回 OpenAI 风格的 {"object": "list", "data": [{"id": ...}]}
    if isinstance(payload, dict) and payload.get("object") == "list":
        data = payload.get("data") or []
        if data and isinstance(data[0], dict):
            return str(data[0].get("id", "?"))
    return None


def port_is_free(host: str, port: int) -> bool:
    """端口是否可绑定。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def kill_pid(pid: int) -> bool:
    """结束指定进程。

    Windows 上优先用 PowerShell 的 ``Stop-Process``：
    实测在受限环境（DSH 文件沙箱）下 ``taskkill /F`` 会因权限被拒，
    而 ``Stop-Process -Force`` 可以正常结束本项目启动的进程。
    """
    if sys.platform == "win32":
        import subprocess

        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    f"Stop-Process -Id {pid} -Force -ErrorAction Stop",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return True
        except Exception:
            pass
        # 退回到 taskkill（部分环境只允许它）
        try:
            result = subprocess.run(
                ["taskkill", "/PID", str(pid), "/F"],
                capture_output=True,
                text=True,
                timeout=20,
            )
            return result.returncode == 0
        except Exception:
            return False

    import os
    import signal

    try:
        os.kill(pid, signal.SIGTERM)
        return True
    except OSError:
        return False


def ensure_port_available(
    host: str,
    port: int,
    service_name: str,
    probe_path: str = "/v1/models",
    force: bool = False,
    already_running_is_ok: bool = True,
) -> int | None:
    """启动前检查端口；可用时返回 ``None``，需要终止时返回退出码。

    调用方约定::

        code = ensure_port_available(...)
        if code is not None:
            return code      # 已经处理完毕（正常退出或报错退出）

    Args:
        host: 监听地址。
        port: 端口。
        service_name: 用于提示的服务名（中文）。
        probe_path: 用于识别「是否本项目服务」的探测路径。
        force: 端口被占用时是否强制结束占用进程。
        already_running_is_ok: 已存在同类服务时是否视为正常（返回 0 而非报错）。

    Returns:
        ``None`` 表示端口可用、可以继续启动；否则返回建议的进程退出码。
    """
    if port_is_free(host, port):
        return None

    pid = _listening_pid(port)
    owner = f"PID {pid}" if pid else "未知进程"
    model = _probe_our_service(host, port, probe_path)

    # ---- 情况一：本项目同类服务已在运行 ----
    if model is not None:
        if force:
            if pid and kill_pid(pid):
                print(f"[信息] 已结束原有 {service_name}（{owner}），准备用新配置启动。")
                return None
            print(f"[警告] 无法结束 {owner}，请手动处理后重试。", file=sys.stderr)
            return 2
        print()
        print(f"[提示] {service_name} 已经在 http://{host}:{port} 上运行（{owner}，模型 {model}）。")
        print(f"       无需重复启动。如需用新配置重启，请任选其一：")
        print(f"         a) 先停止：  Get-Process python | Where-Object {{ $_.Path -like '*工单1*' }} | Stop-Process -Force")
        print(f"         b) 自动重启：pwsh -File 部署\\run.ps1 {('llm' if 'LLM' in service_name else 'asr')} --force")
        print(f"         c) 换端口：  加 --port 另一个端口，并把 RAG 侧地址指过去")
        if already_running_is_ok:
            print(f"       本次不再重复启动，视为成功。")
            return 0
        return 1

    # ---- 情况二：被其它程序占用 ----
    if force and pid:
        if kill_pid(pid):
            print(f"[信息] 已结束占用 {port} 的进程（{owner}），继续启动。")
            return None
        print(f"[错误] 无法结束 {owner}。", file=sys.stderr)
        return 2

    print(
        f"[错误] 端口 {port} 已被其它程序占用（{owner}），无法启动{service_name}。\n"
        f"       解决办法：\n"
        f"         1) 换端口：  --port 8010，并设置对应的 RAG 环境变量\n"
        f"         2) 结束占用：Stop-Process -Id {pid if pid else '<PID>'} -Force\n"
        f"         3) 自动处理：加 --force 参数",
        file=sys.stderr,
    )
    return 2
