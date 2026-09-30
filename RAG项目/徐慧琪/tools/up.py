"""确保求解链路依赖的两个数据库在线。

存在的理由（2026-09-23 实测）：本机 WSL 2.7.14 的关机机制是"没有 WSL 客户端进程
就关机"——哪怕容器正在跑，只要 Windows 侧没有活的 wsl 进程，VM 仍会在 60~120 秒内
被回收，四个容器随之 Exited(0)。当日集成测因此掉库三次，任务中断。

试过但**无效**的办法：`.wslconfig` 的 `vmIdleTimeout`（-1 与 2147483647 都试了，
VM 照关；同一文件里的 memory=3GB 却生效，证明配置确实被读取）。用户 2026-09-23
选择不留该文件，实测记录改存 `.superpowers/sdd/progress.md`，不再散在机器配置里。

真正有效的是**常驻一个 WSL 客户端进程**。本脚本做两件事：
  1. 四个容器没起就起它们（冷启动要 60~90 秒）
  2. 没有保活进程就起一个（Windows 侧隐藏窗口，`sleep infinity`）

用法：cd D:/xinzg6/fl && python tools/up.py
"""
from __future__ import annotations

import subprocess
import sys

# 本机控制台是 GBK，直接 print 中文会抛 UnicodeEncodeError。
# 显式改 UTF-8 并容忍坏字节：本脚本的职责是把环境拉起来，
# 不该因为一行诊断信息编不出来就整个崩掉
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DISTRO = "Ubuntu"
CONTAINERS = ["milvus-etcd", "milvus-minio", "milvus-standalone", "fl-mysql"]
# 保活进程的识别标志：命令行里含这个串的 sleep 进程即为我们起的那个
KEEPER_MARK = "sleep infinity"


def _wsl(*args: str) -> str:
    """在 WSL 里跑一条命令，返回解码后的输出。

    编码口径是 2026-09-23 实测定的：**从 Python 捕获时 wsl.exe 输出 UTF-8**，
    不是②期笔记里写的 UTF-16LE——那条说的是 bash 管道里的情形（`tr -d '\\0'` 就在那）。
    实测 `wsl ... echo HELLO-中文` 拿到的字节是 b'HELLO-\\xe4\\xb8\\xad\\xe6\\x96\\x87\\n'，
    13 字节的 UTF-8。照搬 UTF-16LE 会把容器名解成乱码，进而让"哪些容器在跑"判错。
    """
    raw = subprocess.run(["wsl", "-d", DISTRO, "-u", "root", *args],
                         capture_output=True, check=False).stdout
    return raw.decode("utf-8", errors="replace")


def running_containers(output: str) -> set[str]:
    """从 `docker ps --format {{.Names}}` 的输出里解析出在跑的容器名。

    独立成纯函数是为了可测：真跑一次 docker 要几秒，而解析逻辑值得单独断言。
    """
    return {line.strip() for line in output.splitlines() if line.strip()}


def start_containers() -> None:
    """启动四个容器。已起的会被 docker 忽略，重复调用是安全的。"""
    _wsl("docker", "start", *CONTAINERS)


def keeper_running() -> bool:
    """WSL 里有没有我们起的保活进程。"""
    # pgrep 找不到时返回非零退出码，用 -c 数个数比看退出码更直白
    return _wsl("pgrep", "-fc", KEEPER_MARK).strip() not in ("", "0")


def start_keeper() -> None:
    """在 Windows 侧起一个隐藏窗口的 WSL 客户端进程当"压舱石"。

    用 PowerShell 的 Start-Process 而不是直接 subprocess：后者起的进程会挂在
    本脚本之下，脚本一退出它就没了——那就正好复现了要解决的问题。
    """
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    "Start-Process -WindowStyle Hidden wsl "
                    f"-ArgumentList '-d','{DISTRO}','-u','root','--','{KEEPER_MARK}'"],
                   check=False)


def ensure(verbose: bool = True) -> int:
    """把环境拉到"可以跑集成测"的状态，返回未就绪的容器数（0 表示就绪）。"""
    before = running_containers(_wsl("docker", "ps", "--format", "{{.Names}}"))
    missing = [name for name in CONTAINERS if name not in before]
    if missing:
        if verbose:
            print(f"未在跑的容器：{missing}，正在启动（冷启动约 60~90 秒）...")
        start_containers()
    if not keeper_running():
        if verbose:
            print("没有保活进程，正在起一个（否则 WSL 空闲回收会把容器一起带走）...")
        start_keeper()
    after = running_containers(_wsl("docker", "ps", "--format", "{{.Names}}"))
    still_missing = [name for name in CONTAINERS if name not in after]
    if verbose:
        print(f"在跑的容器：{sorted(after)}")
        if still_missing:
            print(f"仍未就绪：{still_missing}——Milvus 冷启动较慢，稍等再跑一次")
    return len(still_missing)


def main() -> int:
    pending = ensure()
    return 1 if pending else 0


if __name__ == "__main__":
    sys.exit(main())
