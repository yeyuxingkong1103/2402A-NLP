"""CLI 侧读取 `.env`。**离线管线模块共用的唯一实现。**

## 为什么需要这个模块

`backend/api/config.py` 走 pydantic-settings 读 `.env`，那是**服务进程**的路径。
但离线管线（S2 解析、S5 向量化）是**独立的 CLI 进程**，不经过 `create_app`，
因此从来没有读过 `.env`。

在引入本模块之前，两处模型权重路径是**硬编码在源码里**的：

    backend/parse/__init__.py   DEFAULT_PIPELINE_MODELS / DEFAULT_VLM_MODELS
    backend/embed/__init__.py   MODEL_DIR

而 `.env` 里同时存在同名同义的变量（`MINERU_MODELS_DIR`、`EMBED_MODEL_PATH`），
**没有任何代码读它们** —— 名字看着像配置项，实际是死变量。后果是"改了 .env
不生效"，而这类失效没有任何报错：程序照旧用硬编码值跑得好好的，只有换机器时
才会以"路径不存在"的形式暴露出来，且暴露点与真正的改动点隔了一层。

## 占位符处理

沿用 `api/config.py` 的既有约定：`<YOUR_XXX>` 形状的值视为**未填写**，而不是
一个合法的字符串。`.env` 从 `.env.example` 复制后，这类值会通过"非空"检查，
然后在真正使用时以某个毫不相关的错误失败。

## 失败方向

必需项缺失或路径不存在 → **抛 `EnvConfigError`，不返回默认值**。

constitution 的「密钥与配置」条款要求必需项缺失即失败、MUST NOT 静默降级。
对本模块而言这条尤其要紧：静默降级意味着"用上一台机器的路径继续跑"，而那份
权重可能根本不存在 —— 于是失败会推迟到加载模型时才发生，且错误信息指向模型库
内部，而不是指向那个该改的配置项。

调用方负责把 `EnvConfigError` 翻译成自己的错误类型与退出码。
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "EnvConfigError",
    "load_env",
    "is_placeholder",
    "text",
    "optional_text",
    "require_dir",
]

# 仓库根：backend/envfile.py → parents[0]=backend, [1]=仓库根
REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"

# 占位符的外形。与 `api/config.py` 对 `<YOUR_TOP_K>` 的处理保持同一口径。
_PLACEHOLDER_OPEN = "<"
_PLACEHOLDER_CLOSE = ">"


class EnvConfigError(Exception):
    """`.env` 配置有问题。`message` 面向操作者，指明该改哪一个变量。"""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


_LOADED = False


def load_env() -> None:
    """把仓库根 `.env` 载入环境变量。幂等，可重复调用。

    `override=False`：**已存在的真实环境变量优先于 `.env`**。

    这条不是细节。部署时经常用 `E:\\models\\x=... command` 这种一次性覆盖来
    临时换权重，而 `.env` 里留着常规值。若这里 override=True，临时覆盖会被
    静默忽略 —— 使用者会以为换成功了，实际跑的还是 `.env` 里的旧路径。

    与 pydantic-settings 在服务侧的默认行为一致。
    """
    global _LOADED
    if _LOADED:
        return
    _LOADED = True

    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        # 不静默降级：没有 dotenv 就只能靠真实环境变量，那与"读了 .env"
        # 是两种不同的运行状态，必须让调用方知道。
        raise EnvConfigError(
            "缺少依赖 python-dotenv，无法读取 .env。安装：\n"
            "  D:/zg6_Project/9/med_rag/rag/python.exe -m pip install -r requirements.txt"
        ) from None

    # dotenv 的返回值表示"文件是否读到"，文件不存在不算错误 ——
    # 此时后续的 require_dir 会以"变量未设置"失败，信息更具体。
    load_dotenv(ENV_FILE, override=False, encoding="utf-8")


def is_placeholder(value: str | None) -> bool:
    """判断一个取值是不是 `.env.example` 留下的占位符。

    只认 `<...>` 这种外形。不试图更聪明（比如"看起来像路径"）——
    误判一个有意义的值为占位符，会让一个本来能跑的配置被拒绝，
    而那种失败比"漏判"更难排查。
    """
    if value is None:
        return False
    stripped = value.strip()
    return (
        len(stripped) >= 2
        and stripped.startswith(_PLACEHOLDER_OPEN)
        and stripped.endswith(_PLACEHOLDER_CLOSE)
    )


def optional_text(var_name: str, default: str) -> str:
    """读一个**可选**变量。未设置或仍是占位符时返回 `default`。

    用于真正有合理默认值的项（如 `MINERU_MODEL_SOURCE` 的 `local`）。
    占位符按"未设置"处理 —— 一个写着 `<YOUR_X>` 的变量表达的是"还没填"，
    不是"请把它当字面值用"。
    """
    load_env()
    raw = os.environ.get(var_name)
    if raw is None or is_placeholder(raw) or not raw.strip():
        return default
    return raw.strip()


def text(var_name: str) -> str:
    """读一个**必需**变量。未设置或仍是占位符时报错。"""
    load_env()
    raw = os.environ.get(var_name)
    if raw is None or not raw.strip():
        raise EnvConfigError(
            "缺少必需的配置项 %s。请在 %s 中填写（取值说明见 .env.example）。"
            % (var_name, ENV_FILE)
        )
    if is_placeholder(raw):
        raise EnvConfigError(
            "%s 仍是 .env.example 里的占位符 %r。请把它改成真实取值（见 .env.example）。"
            % (var_name, raw.strip())
        )
    return raw.strip()


def require_dir(var_name: str, what: str) -> Path:
    """读一个**必需的目录路径**，并确认它存在。

    ⚠️ 只报**变量名与路径**，MUST NOT 回显其它环境变量的值
    （constitution 原则 III：日志脱敏）。

    `what` 是给人看的用途说明（如 "MinerU pipeline 权重"），
    放进错误信息里，让操作者不必回头翻文档就知道这个目录是干什么的。
    """
    raw = text(var_name)
    path = Path(raw)

    if not path.is_absolute():
        raise EnvConfigError(
            "%s 必须是绝对路径，当前是 %r。相对路径会随工作目录变化而指向不同位置。"
            % (var_name, raw)
        )
    if not path.is_dir():
        raise EnvConfigError(
            "%s 指向的目录不存在（%s 权重）：%s\n"
            "  请确认该路径已下载完成，或在 .env 中改成正确的位置。"
            % (var_name, what, path)
        )
    return path
