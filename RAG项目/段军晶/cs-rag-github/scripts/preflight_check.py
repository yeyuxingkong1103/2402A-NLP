# -*- coding: utf-8 -*-
"""
后端启动前置体检

职责：
    启动 FastAPI 后端**之前**，检查三个依赖服务是否可达，并在不可达时
    给出具体的处理办法。

为什么需要这一步：
    后端在依赖缺失时**不会启动失败**，只会记录告警并降级运行 ——
    Milvus 不可用则检索不可用，MySQL 不可用则页码溯源不可用，
    Redis 不可用则多轮记忆与缓存降级。这种降级是静默的：服务"看起来
    起来了"，但实际能力已经受损。体检的意义就是**把降级提前暴露给操作者**，
    而不是等他提问时才发现答案没有出处。

用法：
    python -m scripts.preflight_check        # 由 start_backend.bat 调用
    python -m scripts.preflight_check --yes  # 有问题也不询问，直接放行

退出码：
    0 = 可以启动
    1 = 不启动（环境不完整，或用户在发现依赖缺失后选择中止）
"""

from __future__ import annotations

import argparse
import socket
import sys
from typing import List, Optional, Tuple

from backend.config import settings


# ---------------------------------------------------------------------------
# 依赖清单
# ---------------------------------------------------------------------------
# 每项：(显示名, 主机配置属性, 端口配置属性, 用途, 不通时的处理办法, 降级后果)
DEPENDENCIES: List[Tuple[str, str, str, str, str, str]] = [
    (
        "Milvus", "milvus_host", "milvus_port", "向量存储与检索",
        "启动 Docker Desktop，等待其就绪（Milvus 运行在容器中，冷启动约 40 秒）",
        "检索完全不可用",
    ),
    (
        "MySQL", "mysql_host", "mysql_port", "元数据与页码真相源",
        "用【管理员】PowerShell 执行：net start MySQL80\n"
        "             （或 Win+R → services.msc → 右键 MySQL80 → 启动）",
        "答案能返回但没有来源页码",
    ),
    (
        "Redis", "redis_host", "redis_port", "会话记忆与查询缓存",
        "启动 Docker Desktop，等待其就绪（Redis 运行在容器中）",
        "多轮追问失效、重复提问不再命中缓存",
    ),
]


# ---------------------------------------------------------------------------
# 探测
# ---------------------------------------------------------------------------

def _port_open(host: str, port: int, timeout: float = 1.5) -> bool:
    """TCP 连通性探测。

    用 socket 直连而非解析 netstat 输出：中文 Windows 上 netstat 的文案
    是本地化的，解析容易出错。
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            return sock.connect_ex((host, port)) == 0
    except OSError:
        return False


def _pid_on_port(port: int) -> str:
    """
    查出监听指定端口的进程 PID。

    用途：端口被占用时给出可执行的清理指引。**仅解析 LISTENING 行**，
    因为已建立连接的普通行也会含该端口号，容易误判。
    """
    try:
        import subprocess

        output = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True, timeout=10, errors="ignore",
        ).stdout
    except Exception:
        return ""

    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 5 or "LISTENING" not in line.upper():
            continue
        # 形如：TCP  0.0.0.0:8000  0.0.0.0:0  LISTENING  12345
        if parts[1].endswith(":{}".format(port)):
            return parts[-1]
    return ""


def _lan_ip() -> str:
    """取本机在局域网中的地址。

    用 UDP socket 连一个外网地址，让操作系统按路由表选出出口网卡 ——
    并没有真的发包。比解析 ipconfig 可靠：ipconfig 会同时列出 Docker、
    WSL、VMware 的虚拟网卡（形如 172.x.x.x / 192.168.x.1），容易取错。
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return str(sock.getsockname()[0])
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def _line(text: str = "") -> None:
    print(text)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="后端启动前置体检")
    parser.add_argument(
        "--yes", action="store_true",
        help="发现依赖缺失时不再询问，直接放行启动",
    )
    args = parser.parse_args(argv)

    # 让中文在 Windows 控制台正确输出（配合 .bat 里的 chcp 65001）
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    _line("=" * 62)
    _line("  {} —— 启动前体检".format(settings.app_title))
    _line("=" * 62)
    _line()

    # ---- 1. 运行环境
    _line("[1/3] 运行环境")
    python_exe = sys.executable
    _line("      解释器：{}".format(python_exe))
    _line("      版本  ：{}".format(sys.version.split()[0]))
    _line()

    # ---- 2. 依赖服务
    _line("[2/3] 依赖服务连通性")
    failures: List[Tuple[str, str, str]] = []      # (名称, 处理办法, 降级后果)

    for name, host_attr, port_attr, purpose, remedy, impact in DEPENDENCIES:
        host = getattr(settings, host_attr)
        port = int(getattr(settings, port_attr))
        ok = _port_open(host, port)
        mark = "OK  " if ok else "失败"
        _line("      [{}] {}:{}（{}）".format(mark, host, port, purpose))
        if not ok:
            failures.append((name, remedy, impact))

    _line()

    # ---- 3. 端口占用（后端自身）
    backend_busy = _port_open("127.0.0.1", settings.app_port, timeout=0.5)
    if backend_busy:
        pid = _pid_on_port(settings.app_port)
        _line("      注意：端口 {} 已被占用，后端可能已经在运行。".format(settings.app_port))
        _line("            继续启动会因端口冲突而失败。")
        if pid:
            _line("            占用进程 PID = {}".format(pid))
            _line("            若确认需要重启，先结束该进程：")
            _line("                taskkill /PID {} /F".format(pid))
            _line("            提示：若该进程是上次异常退出的残留（监听端口但不响应），")
            _line("                  必须强制结束后才能重新启动。")
        _line()

    # ---- 结论
    if not failures and not backend_busy:
        _line("[3/3] 体检通过，正在启动后端")
        _print_access()
        return 0

    if backend_busy:
        _line("启动已中止：端口 {} 被占用。".format(settings.app_port))
        return 1

    _line("-" * 62)
    _line("发现 {} 个依赖服务不可达：".format(len(failures)))
    _line()
    for name, remedy, impact in failures:
        _line("  ● {} 不可达".format(name))
        _line("      处理办法：{}".format(remedy))
        _line("      若忽略，后果：{}".format(impact))
        _line()

    _line("说明：后端在依赖缺失时【不会启动失败】，只会降级运行 ——")
    _line("      也就是说，即使现在继续启动，服务也会以残缺能力运行。")
    _line()

    if args.yes:
        _line("（--yes 已指定，跳过询问，继续启动）")
    else:
        try:
            answer = input("是否仍要启动？[y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if answer not in ("y", "yes"):
            _line()
            _line("已中止。请先按上面的处理办法修复依赖，再重新运行本脚本。")
            return 1

    _line()
    _line("[3/3] 正在启动后端（降级模式）")
    _print_access()
    return 0


def _print_access() -> None:
    """打印访问地址"""
    _line()
    _line("      本机访问：http://127.0.0.1:{}/".format(settings.app_port))
    lan = _lan_ip()
    if lan:
        _line("      局域网  ：http://{}:{}/".format(lan, settings.app_port))
    _line("      接口文档：http://127.0.0.1:{}/docs".format(settings.app_port))
    _line("      健康检查：http://127.0.0.1:{}/api/health".format(settings.app_port))
    _line()
    _line("      提示：后端启动时会等待 Milvus 就绪，首次可能出现约 40 秒无输出，属正常现象。")
    _line("=" * 62)
    _line()


if __name__ == "__main__":
    sys.exit(main())
