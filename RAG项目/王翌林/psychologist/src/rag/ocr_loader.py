"""文档解析增强：MinerU（文本型 PDF 高质量解析）+ PaddleOCR（扫描件 OCR 回退）

本模块是 src/rag/parser.py 的可选后端，当 PDF 文本层过薄（疑似扫描件）时自动触发。
底层调用 WSL 中的 MinerU / PaddleOCR venv，不影响主流程。

核心函数：
    mineru_extract_pdf(path, pages) -> str     # 文本型 PDF 高质量提取
    paddle_ocr_pdf(path, pages) -> str         # 扫描件 OCR 识别

两者均返回纯文本（parser.py 会走 clean_text/deduplicate_paragraphs 后处理）。
"""
import os
import re
import shlex
import signal
import subprocess
from typing import Optional

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger("rag.ocr_loader")

# _paddle_runner.py 位置（src/rag/_paddle_runner.py）
_HERE = os.path.dirname(os.path.abspath(__file__))
_PADDLE_RUNNER = os.path.join(_HERE, "_paddle_runner.py")


# ---------------- 运行环境检测 ----------------
def _is_wsl() -> bool:
    """检测当前是否运行在 WSL 内部"""
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except Exception:
        return False


def _win_to_wsl_path(path: str) -> str:
    """Windows 盘符路径 -> /mnt/<drive>/...；已是 Linux 路径则原样返回"""
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", path)
    if m:
        drive = m.group(1).lower()
        rest = m.group(2).replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return path


def _run(cmd_parts: list, timeout: int = 1800) -> tuple:
    """执行命令，返回 (returncode, stdout, stderr)

    在 WSL 内直接执行 cmd_parts；在 Windows 上用 wsl 命令包装。
    超时时杀掉整个进程组 —— 只杀 bash 会让 python 子进程变成孤儿继续占用内存。
    """
    # 用 shlex.quote 逐个转义参数：文件路径常含空格/中文/引号等特殊字符，
    # 拼进 `bash -c` 的字符串时若不转义，会被 shell 拆词甚至命令注入，故必须转义。
    shell_cmd = " ".join(shlex.quote(c) for c in cmd_parts)

    # 跨环境桥接：当前已在 WSL 内 → 直接跑 bash；在 Windows 上 → 用 wsl 命令
    # 钻进指定发行版再跑 bash，从而复用 WSL 里装好的 venv 与 GPU 环境。
    if _is_wsl():
        full = ["bash", "-c", shell_cmd]
    else:
        full = ["wsl", "-d", settings.wsl_distribution, "--", "bash", "-c", shell_cmd]

    logger.info("执行: %s", " ".join(full))
    try:
        proc = subprocess.Popen(
            full, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
            start_new_session=True,  # 让子进程自成新"进程组"(成为组长)，便于整组一次性杀死
        )
    except Exception as e:
        logger.error("命令启动失败: %s", e)
        return -1, "", str(e)

    try:
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode, out, err
    except subprocess.TimeoutExpired:
        logger.error("命令超时(%ss)，终止整个进程组: %s", timeout, " ".join(full))
        # ⚠ 必须杀"整个进程组"而非只杀顶层 bash：这里实际执行的是
        #   `bash -c "python _paddle_runner.py ..."`，真正的重活是 bash 派生的
        #   python 子进程。若只 kill 顶层 bash，python 会变成孤儿进程继续偷偷跑
        #   OCR、持续吃内存；多个超时任务叠加就会把机器 OOM 掉。
        #   得益于 start_new_session=True，子进程组与父进程隔离，可整组 killpg 清掉。
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            proc.kill()
        try:
            out, err = proc.communicate(timeout=30)
        except Exception:
            out, err = "", ""
        return -1, out or "", (err or "") + f"\ntimeout after {timeout}s"


# ---------------- MinerU ----------------
def mineru_extract_pdf(
    path: str,
    pages: Optional[str] = None,
    tier: Optional[str] = None,
    timeout: int = 120,
) -> str:
    """用 MinerU 解析 PDF，返回纯文本（markdown 中的文字提取后拼接）

    Args:
        path:   PDF 路径（Windows 或 WSL 均可）
        pages:  页码范围，如 "1-5"、"1,3,5-7"、None=全部
        tier:   MinerU tier，默认读 settings.mineru_tier
        timeout: 单文件超时秒数

    Returns:
        纯文本字符串（已剥离 markdown 标记），失败返回 ""
    """
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        logger.warning("MinerU: 文件不存在 %s", abs_path)
        return ""

    wsl_path = _win_to_wsl_path(abs_path)
    use_tier = tier or settings.mineru_tier
    out_md = f"/tmp/mineru_{os.getpid()}_{os.path.basename(wsl_path)}.md"

    # 默认 pages=all 扫描全部页面；flash tier 只预览前 10 页，需显式指定
    use_pages = pages or "all"

    cmd = [
        settings.mineru_venv_path, "parse", wsl_path,
        "-o", out_md, "-f", "markdown",
        "--tier", use_tier,
        "-p", use_pages,
    ]

    rc, out, err = _run(cmd, timeout=timeout)
    if rc != 0:
        logger.warning("MinerU 解析失败 rc=%s: %s", rc, err[:300])
        return ""

    # 读回 markdown（直接传 shell 命令，不再套一层 bash -c）
    rc2, md_text, err2 = _run(["cat", out_md])
    _run(["rm", "-f", out_md])

    if rc2 != 0:
        logger.warning("MinerU 输出读取失败: %s", err2)
        return ""

    # 剥离 markdown 标记，只留纯文本
    text = _strip_markdown(md_text)
    logger.info("MinerU 解析完成 %s tier=%s -> %d 字符",
                os.path.basename(path), use_tier, len(text))
    return text


# ---------------- PaddleOCR ----------------
def paddle_ocr_pdf(
    path: str,
    pages: str = "all",
    lang: Optional[str] = None,
    timeout: int = 120,
) -> str:
    """用 PaddleOCR 对扫描版 PDF 做 OCR，返回纯文本

    Args:
        path:   PDF 路径（Windows 或 WSL 均可）
        pages:  页码范围，默认 "all"（首次尝试时建议限制页数控制耗时）
        lang:   语言，默认读 settings.paddle_ocr_lang

    Returns:
        纯文本字符串，失败返回 ""
    """
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        logger.warning("PaddleOCR: 文件不存在 %s", abs_path)
        return ""

    wsl_path = _win_to_wsl_path(abs_path)
    runner = _win_to_wsl_path(_PADDLE_RUNNER)
    out_md = f"/tmp/paddle_{os.getpid()}_{os.path.basename(wsl_path)}.md"

    cmd = [
        settings.paddle_ocr_venv_path, runner, wsl_path,
        "--pages", pages,
        "--lang", lang or settings.paddle_ocr_lang,
        "--out", out_md,
    ]

    rc, out, err = _run(cmd, timeout=timeout)
    if rc != 0:
        logger.warning("PaddleOCR 返回码 rc=%s（可能是超时/中断），尝试保留已完成页：%s",
                       rc, err[:200])

    # ⚠ 无论 rc 是否为 0 都读回输出：_paddle_runner 是"逐页增量写盘 + 每页 flush"的，
    #   即使本次因超时被 killpg 杀掉（rc!=0），已完成页的结果仍留在磁盘文件里。
    #   读回它就能把"部分成果"救下来复用，而不是整份重跑（OCR 重跑成本极高）。
    rc2, md_text, err2 = _run(["cat", out_md])
    _run(["rm", "-f", out_md])

    if rc2 != 0:
        logger.warning("PaddleOCR 输出读取失败: %s", err2)
        return ""

    # 剥离 markdown 标记（PaddleOCR 输出自带 ## Page X 标题）
    text = _strip_markdown(md_text)
    if rc == 0:
        logger.info("PaddleOCR 完成 %s pages=%s -> %d 字符",
                    os.path.basename(path), pages, len(text))
    else:
        logger.warning("PaddleOCR 部分完成 %s pages=%s -> %d 字符（未跑完全部页）",
                       os.path.basename(path), pages, len(text))
    return text


# ---------------- 辅助 ----------------
def _strip_markdown(md: str) -> str:
    """把 mineru / paddle_ocr 输出的 markdown 转成纯文本

    - 去掉 HTML 注释（MinerU page marker）
    - 去掉 markdown 标题标记
    - 去掉 ![xxx](xxx) 图片占位
    - 去掉 <u>...</u> 下划线标记
    - 保留文字本身
    """
    if not md:
        return ""
    # ⚠ 剥离顺序有意为之，不可随意调换：
    #   先去 HTML 注释（MinerU 的 <!-- page --> 页标记）→ 去图片占位 → 去 ## 标题
    #   → 先展开 <u>…</u> 再删其余 HTML 标签。若先做通用的 `<[^>]+>` 标签删除，
    #   会连同 <u> 一起删掉而丢失去下划线包裹的正文内容，故必须先单独展开 <u>。
    text = re.sub(r"<!--.*?-->", "", md, flags=re.DOTALL)       # HTML 注释
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)             # ![img](xxx)
    text = re.sub(r"#{1,6}\s*", "", text)                        # ## 标题
    text = re.sub(r"<u>(.*?)</u>", r"\1", text, flags=re.DOTALL) # <u>下划线</u>
    text = re.sub(r"<[^>]+>", "", text)                          # 其他 HTML 标签
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
