# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 工单01 - 基于PDF文档的问答系统
"""
环境自检 —— 施工前的第一道拦截。

【为什么必须先跑这个】
本项目有 5 个「不验就会在后面某处静默翻车」的前提条件，全是实测踩出来的：

1. **宿主 IP 动态发现**：Ollama 跑在 Windows 侧，WSL2 是 NAT 网络，
   WSL 只能用「默认网关」访问宿主。该地址每次 WSL 重启都会变，
   写死在配置里必然失效。这里验证能否自动取到。
2. **think:false 是否真的生效**：qwen3 默认输出思维链。若没关掉，
   单次响应从 0.3s 变成十几秒，工单01「响应 ≤3 秒」直接挂。
   而 OpenAI 兼容端点 /v1 **不保证透传** think 参数 —— 所以要分别验证，
   用实测数据决定走哪个端点。
3. **bge-m3 维度**：Milvus 建表时 FLOAT_VECTOR 的 dim 是**建表即定死、
   事后不能改**的。维度猜错 = 整个知识库推倒重建。
4. **双模型互踢**：qwen3(5.0G) + bge-m3(1.2G) 在 8GB 显存里是紧平衡。
   如果 OLLAMA_MAX_LOADED_MODELS=1，嵌入和聊天会互相卸载，导致
   每次问答都重新冷启动（实测 7.2s）。这里用「嵌入→聊天→嵌入」的
   时序探针实测第二次嵌入的耗时，一眼看出有没有被踢。
5. **Milvus 连通性**：Docker 里的 Milvus 没起来的话，入库阶段会白跑。

任一检查失败即退出码非 0，避免带着坏前提往下施工。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx

# 允许以 `python scripts/preflight.py` 直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import PROJECT_ROOT, detect_host_ip, settings  # noqa: E402

OK = "\033[32m  OK  \033[0m"
FAIL = "\033[31m FAIL \033[0m"
WARN = "\033[33m WARN \033[0m"

_results: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, detail: str = "", *, fatal: bool = True) -> bool:
    """记录一项检查结果。fatal=False 的项失败不阻断整体。"""
    tag = OK if passed else (FAIL if fatal else WARN)
    print(f"[{tag}] {name}" + (f"\n         {detail}" if detail else ""))
    _results.append((name, passed or not fatal, detail))
    return passed


def section(title: str) -> None:
    print(f"\n\033[1m── {title} ──\033[0m")


# ----------------------------------------------------------------------
# 1. 宿主 IP
# ----------------------------------------------------------------------
def check_host_ip() -> str | None:
    section("1. Ollama 宿主地址")
    ip = detect_host_ip()
    if not ip:
        record("自动发现宿主 IP", False, "`ip route show default` 未解析出网关，请检查 WSL 网络")
        return None
    record("自动发现宿主 IP", True, f"网关 = {ip}（每次 WSL 重启可能变化，故运行时动态取）")

    configured = settings.ollama_host.strip()
    if configured and configured != ip:
        print(f"[{WARN}] .env 里写死 {configured}，与当前网关不符，将优先使用动态值")
    return ip


# ----------------------------------------------------------------------
# 2. Ollama 连通 + 模型齐备
# ----------------------------------------------------------------------
def check_ollama(base: str) -> bool:
    section("2. Ollama 连通性与模型")
    try:
        r = httpx.get(f"{base}/api/tags", timeout=10.0)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        record("Ollama 可达", False, f"{base} 无响应：{e}")
        return False
    record("Ollama 可达", True, base)

    names = [m["name"] for m in r.json().get("models", [])]
    for want in (settings.llm_model, settings.embed_model):
        # bge-m3 在 tags 里可能是 "bge-m3:latest"
        hit = any(n == want or n.split(":")[0] == want.split(":")[0] for n in names)
        record(f"模型存在：{want}", hit,
               "" if hit else f"已装：{', '.join(names[:8])}…（请 ollama pull {want}）")
    return True


# ----------------------------------------------------------------------
# 3. think:false 是否生效（原生 /api/chat vs OpenAI 兼容 /v1）
# ----------------------------------------------------------------------
_PROBE = [{"role": "user", "content": "1+1等于几？只回答数字。"}]


def _looks_like_thinking(text: str) -> bool:
    """思维链的典型特征：出现 think 标签或大段自我盘算。"""
    if "think>" in text or "让我" in text or "我们需要" in text:
        return True
    # 正常回答应该很短；超过 200 字基本是在思考
    return len(text) > 200


def check_think(base: str) -> bool:
    section("3. think:false 是否真的生效（决定走哪个端点）")

    # --- 原生 /api/chat ---
    native_ok = native_fast = False
    try:
        t0 = time.perf_counter()
        r = httpx.post(
            f"{base}/api/chat",
            json={
                "model": settings.llm_model,
                "messages": _PROBE,
                "stream": False,
                "think": False,
                "keep_alive": settings.llm_keep_alive,
                "options": {"num_predict": 32, "temperature": 0},
            },
            timeout=180.0,
        )
        native_elapsed = time.perf_counter() - t0
        r.raise_for_status()
        text = r.json().get("message", {}).get("content", "")
        native_ok = not _looks_like_thinking(text)
        native_fast = native_elapsed < 5.0
        record("原生 /api/chat + think:false 关闭思维链", native_ok,
               f"耗时 {native_elapsed:.2f}s｜返回：{text.strip()[:60]!r}")
    except Exception as e:  # noqa: BLE001
        print(f"[{FAIL}] 原生 /api/chat 调用失败：{e}")

    # --- OpenAI 兼容 /v1 ---
    try:
        t0 = time.perf_counter()
        r = httpx.post(
            f"{base}/v1/chat/completions",
            json={
                "model": settings.llm_model,
                "messages": _PROBE,
                "stream": False,
                "max_tokens": 32,
                "temperature": 0,
            },
            timeout=180.0,
        )
        v1_elapsed = time.perf_counter() - t0
        r.raise_for_status()
        v1_text = r.json()["choices"][0]["message"]["content"]
        v1_ok = not _looks_like_thinking(v1_text)
        record("OpenAI 兼容 /v1 是否透传 think", v1_ok,
               f"耗时 {v1_elapsed:.2f}s｜返回：{v1_text.strip()[:60]!r}"
               + ("" if v1_ok else "  ← /v1 未关掉思维链，必须走原生接口"),
               fatal=False)
    except Exception as e:  # noqa: BLE001
        print(f"[{WARN}] /v1 调用失败（不影响，本项目本就走原生接口）：{e}")

    record("结论：原生 /api/chat 满足 ≤3s 口径", native_ok and native_fast,
           f"原生 {native_elapsed:.2f}s > 阈值 5s" if not native_fast else "冷启动已过，后续为热响应")
    return native_ok


# ----------------------------------------------------------------------
# 4. bge-m3 维度
# ----------------------------------------------------------------------
def check_embed_dim(base: str) -> int | None:
    section("4. 向量维度（Milvus FLOAT_VECTOR 建表即定死）")
    try:
        r = httpx.post(
            f"{base}/api/embed",
            json={"model": settings.embed_model, "input": ["测试文本"]},
            timeout=120.0,
        )
        r.raise_for_status()
        vec = r.json()["embeddings"][0]
    except Exception as e:  # noqa: BLE001
        record("bge-m3 向量化", False, f"调用失败：{e}")
        return None

    dim = len(vec)
    match = dim == settings.embed_dim
    record(f"bge-m3 维度 = {settings.embed_dim}", match,
           f"实测 {dim} 维" + ("" if match else f"，与 config.embed_dim={settings.embed_dim} 不符，必须对齐"))
    return dim


# ----------------------------------------------------------------------
# 5. 双模型互踢探针
# ----------------------------------------------------------------------
def check_model_thrash(base: str) -> bool:
    section("5. 双模型互踢探针（嵌入→聊天→嵌入）")
    print("         原理：若 OLLAMA_MAX_LOADED_MODELS=1，第二次嵌入会把 qwen3 卸载，")
    print("         导致下一轮问答重新冷启动 7.2s。观察第二次嵌入耗时即可判断。")

    def embed_once() -> float:
        t0 = time.perf_counter()
        httpx.post(f"{base}/api/embed",
                   json={"model": settings.embed_model, "input": ["探测"]},
                   timeout=120.0).raise_for_status()
        return time.perf_counter() - t0

    try:
        e1 = embed_once()
        httpx.post(
            f"{base}/api/chat",
            json={"model": settings.llm_model, "messages": _PROBE, "stream": False,
                  "think": False, "keep_alive": settings.llm_keep_alive,
                  "options": {"num_predict": 8, "temperature": 0}},
            timeout=180.0,
        ).raise_for_status()
        e2 = embed_once()
    except Exception as e:  # noqa: BLE001
        record("双模型互踢探针", False, f"探测过程出错：{e}")
        return False

    thrash = e2 > e1 * 3 and e2 > 1.0
    record("嵌入模型未被聊天模型踢出", not thrash,
           f"第一次嵌入 {e1 * 1000:.0f}ms → 聊天 → 第二次嵌入 {e2 * 1000:.0f}ms"
           + ("  ← 疑似被踢，请设 OLLAMA_MAX_LOADED_MODELS=2" if thrash else ""))

    # 顺带看看已加载模型数
    try:
        ps = httpx.get(f"{base}/api/ps", timeout=10.0).json().get("models", [])
        print(f"         当前常驻模型：{', '.join(m.get('name', '?') for m in ps) or '(无)'}")
    except Exception:  # noqa: BLE001
        pass
    return not thrash


# ----------------------------------------------------------------------
# 6. Milvus 连通
# ----------------------------------------------------------------------
def check_milvus() -> bool:
    section("6. Milvus 连通性")
    try:
        from pymilvus import MilvusClient
    except ImportError:
        record("pymilvus 已安装", False, "请 pip install pymilvus")
        return False

    uri = f"http://{settings.milvus_host}:{settings.milvus_port}"
    try:
        c = MilvusClient(uri=uri, timeout=10)
        c.list_collections()
        record("Milvus Standalone 可达", True, uri)
        return True
    except Exception as e:  # noqa: BLE001
        record("Milvus Standalone 可达", False,
               f"{uri} 连不上：{str(e)[:160]}\n"
               f"         先执行：docker compose -f docker-compose.milvus.yml up -d")
        return False


# ----------------------------------------------------------------------
# 7. PDF 页数与页码偏移
# ----------------------------------------------------------------------
def check_pdf() -> bool:
    section("7. PDF 解析前提（页数 548 / 页码偏移 −1）")
    pdf = settings.data_path / "raw" / "招股说明书1.pdf"
    if not pdf.exists():
        record("招股说明书1.pdf 存在", False, f"未找到 {pdf}")
        return False

    try:
        import pymupdf
    except ImportError:
        record("pymupdf 已安装", False, "请 pip install pymupdf")
        return False

    doc = pymupdf.open(pdf)
    n = doc.page_count
    record("PDF 页数 = 548", n == 548, f"实测 {n} 页（{pdf.stat().st_size / 1e6:.2f} MB）")

    # 页码偏移：**只看页脚块**（y > 85% 页高），避免把正文里
    # “详见 1-1-21 页” 这类交叉引用误当成页码 —— 这正是之前侦察
    # 得出错误偏移 −1 的原因。
    offsets: list[tuple[int, int]] = []
    for idx in (22, 129, 400):
        if idx >= n:
            continue
        page = doc[idx]
        h = page.rect.height
        for b in page.get_text("dict")["blocks"]:
            if b["bbox"][1] < h * 0.85:  # 非页脚，跳过
                continue
            txt = "".join(s["text"] for l in b.get("lines", []) for s in l["spans"])
            m = re.search(r"1-1-(\d+)", txt)
            if m:
                offsets.append((idx, idx - int(m.group(1))))
                break
    doc.close()

    if not offsets:
        record("页码映射", False, "页脚未找到 1-1-xx 形式的页码，需人工确认", fatal=False)
        return True

    got = {o for _, o in offsets}
    detail = "；".join(f"索引 {i} → 偏移 {o}" for i, o in offsets)
    if len(got) != 1:
        record("页码偏移全页一致", False, f"偏移不统一：{detail}")
    else:
        actual = got.pop()
        want = settings.page_label_offset
        record(f"页码偏移 = {want}（config.page_label_offset）", actual == want,
               f"{detail}" + ("" if actual == want else f"  ← 实测 {actual}，与配置 {want} 不符，"
                              f"必须改 config.page_label_offset，否则引用页码全部错位"))
    return True


# ----------------------------------------------------------------------
# 8. 运行时环境
# ----------------------------------------------------------------------
def check_runtime() -> None:
    section("8. 运行时环境")
    import platform

    record("Python 版本 ≥ 3.10", sys.version_info >= (3, 10),
           f"{platform.python_version()} @ {sys.executable}")

    # 内存：8GB 上限下，Milvus + Ollama 客户端要留够余量
    try:
        mem = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            mem[k] = int(v.strip().split()[0])
        avail_gb = mem["MemAvailable"] / 1024 / 1024
        record("可用内存 ≥ 2GB", avail_gb >= 2.0, f"MemAvailable = {avail_gb:.1f} GB")
    except Exception:  # noqa: BLE001
        pass

    # 数据目录必须在 ext4 上（/mnt/c 是 9p，慢 1-2 个数量级）
    dp = str(settings.data_path)
    record("数据目录不在 /mnt/c（避开 9p）", not dp.startswith("/mnt/c"), dp)

    # 中文字体：生成流程图/截图需要
    try:
        out = subprocess.run(["fc-list", ":lang=zh"], capture_output=True, text=True, timeout=10)
        n_fonts = len([l for l in out.stdout.splitlines() if l.strip()])
        record("中文字体可用", n_fonts > 0, f"fc-list :lang=zh → {n_fonts} 条", fatal=False)
    except Exception:  # noqa: BLE001
        print(f"[{WARN}] 未能检测中文字体（非阻塞）")


# ----------------------------------------------------------------------
def main() -> int:
    print("\033[1m" + "=" * 68)
    print("  工单01 · 环境自检 preflight")
    print("=" * 68 + "\033[0m")
    print(f"项目根目录：{PROJECT_ROOT}")
    print(f"数据目录  ：{settings.data_path}")

    ip = check_host_ip()
    base = f"http://{ip}:{settings.ollama_port}" if ip else settings.ollama_base_url

    if check_ollama(base):
        check_think(base)
        check_embed_dim(base)
        check_model_thrash(base)
    check_milvus()
    check_pdf()
    check_runtime()

    # ---------------- 汇总 ----------------
    section("汇总")
    failed = [(n, d) for n, ok, d in _results if not ok]
    for n, _, _ in _results:
        pass
    print(f"共 {len(_results)} 项检查，通过 {len(_results) - len(failed)} 项，失败 {len(failed)} 项")
    if failed:
        print("\n\033[31m未通过项：\033[0m")
        for n, d in failed:
            print(f"  · {n}\n      {d}")
        print("\n\033[33m提示：Milvus 未就绪不影响先跑 `python scripts/ingest.py --dry-run`\033[0m")
        return 1
    print("\n\033[32m全部通过，可以开始入库。\033[0m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
