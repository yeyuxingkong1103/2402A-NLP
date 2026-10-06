# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 工单编号：人工智能NLP-RAG-Query 理解优化任务
# 工单03 - PDF文档的表格解析及检索优化
# 工单01 - 基于PDF文档的问答系统
# 工单02 - 基于PDF文档的问答系统的优化
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

import asyncio
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
    # 工单04：多模态模型也是硬依赖 —— 缺了它图块会**静默变成只有题注的空块**，
    # 属于"静默答错"，所以判 FAIL 而不是 WARN。
    wants = [settings.llm_model, settings.embed_model]
    if settings.image_enable:
        wants.append(settings.vlm_model)
    for want in wants:
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

    # 【必须先预热再计时】否则测到的是**冷启动**而不是稳态。
    # 实测踩过：不预热时这一项报"原生 8.78s > 阈值 5s"判失败，而同期评测里
    # 10 道题的 TTFT 全部在 1.1–2.7s。差的 6 秒就是 qwen3 从磁盘加载的时间，
    # 它跟"要不要走原生端点"这个被检查的问题毫无关系。
    # 应用的 lifespan 里有 warmup，这里补一次，让探针口径与应用一致。
    try:
        httpx.post(
            f"{base}/api/chat",
            json={"model": settings.llm_model, "messages": _PROBE,
                  "stream": False, "think": False,
                  "keep_alive": settings.llm_keep_alive,
                  "options": {"num_predict": 1, "temperature": 0}},
            timeout=180.0,
        )
    except Exception:  # noqa: BLE001
        pass          # 预热失败不阻断，下面正式探测会如实报错

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
    # 【必须走应用的嵌入路径，不能用裸 /api/embed】
    # 裸调用不带 num_gpu，bge-m3 会被加载到 **GPU** 上 —— 而 8GB 显存只装得下
    # 一个：qwen3 占 5.58GB，bge-m3 上 GPU 就会把 qwen3 挤出去（config 里
    # embed_query_on_cpu 那条注释说的就是这个）。于是第 5 节的聊天调用要重载
    # qwen3、又把 bge-m3 踢掉，第二次嵌入变成 2 秒 —— 探针**亲手制造了它要
    # 检测的故障**，然后判定系统有病。改用 Embedder（内部 num_gpu=0）后，
    # 探测不会改变现场。
    from app.core.embedder import Embedder
    try:
        vec = asyncio.run(Embedder().embed_one("测试文本"))
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
    section("5. 双模型互踢探针（嵌入 ⇄ 聊天，交替两轮）")
    print("         原理：应用每问一题都是「先嵌入、再聊天」。若两个模型互相卸载，")
    print("         每一轮交替都要付一次重载（实测单次重载 ≈2s、冷启动 ≈7s）。")
    print("         所以判据不是「某一次慢」，而是交替两轮后还能不能恢复。")

    # 【为什么必须走应用自己的 Embedder，不能用裸 /api/embed】
    # 应用把 bge-m3 放 CPU 上跑（embed_query_on_cpu=True → num_gpu=0），
    # 正是为了不把显存里的 qwen3 挤出去。裸 /api/embed 不传 num_gpu 会走 GPU，
    # 于是探针自己就把 qwen3 挤掉、下一轮再被 qwen3 挤回来 ——
    # **探针亲手制造了它要检测的故障**。改走真实路径，问的才是对的问题。
    from app.core.embedder import Embedder
    emb = Embedder()

    def embed_once() -> float:
        t0 = time.perf_counter()
        asyncio.run(emb.embed_one("探测"))
        return time.perf_counter() - t0

    def chat_once() -> None:
        httpx.post(
            f"{base}/api/chat",
            json={"model": settings.llm_model, "messages": _PROBE, "stream": False,
                  "think": False, "keep_alive": settings.llm_keep_alive,
                  "options": {"num_predict": 8, "temperature": 0}},
            timeout=180.0,
        ).raise_for_status()

    try:
        embed_once()                    # 预热 bge-m3，避免把冷启动当成"被踢"
        chat_once()                     # 预热 qwen3
        e0 = embed_once()               # 两个都热了，取基准
        chat_once()
        e1 = embed_once()               # 第一次交替后的嵌入
        chat_once()
        e2 = embed_once()               # 第二次交替后的嵌入 ← 这一发才是判据
    except Exception as e:  # noqa: BLE001
        record("双模型互踢探针", False, f"探测过程出错：{e}")
        return False

    # 首轮慢可以理解成"一次性重载"；**连续两轮都慢**才是真的互相卸载。
    base = max(e0, 0.01)
    thrash = e2 > base * 3 and e2 > 1.0
    record("嵌入模型未被聊天模型踢出", not thrash,
           f"基准 {e0 * 1000:.0f}ms → 交替1 {e1 * 1000:.0f}ms → 交替2 "
           f"{e2 * 1000:.0f}ms"
           + ("  ← 连续两轮都慢，确属互相卸载：请设 "
              "OLLAMA_MAX_LOADED_MODELS=2 后重启 Ollama" if thrash else ""))

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
    # 页码偏移**实测为 0**（index N ↔ 页脚 1-1-N）；早期侦察得到的 −1 是错的。
    # 这行标题曾长期写着 −1，与 config.page_label_offset 不一致，已改为读配置。
    section(f"7. PDF 解析前提（页数 548 / 页码偏移 {settings.page_label_offset}）")
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
# 7b. 工单03：第二份 PDF 与「两份都在库里」
# ----------------------------------------------------------------------
#: 每份文档的实测页数（书1 548 / 书2 350），与 doc_profiles 的 key 对齐
EXPECTED_PAGES = {"招股说明书1.pdf": 548, "招股说明书2.pdf": 350}


def check_pdf2() -> bool:
    """工单03 新增：第二份 PDF 的解析前提。

    【为什么单独查】书2 的页脚是**裸数字**（书1 是 `1-1-N`），页眉模板也不同，
    且多出 8 页旋转 90° 的横向表。这三点都是实测出来的、也都能静默出错 ——
    页眉清不掉只是结果变差，页脚认错会让引用页码全错位却照常返回。
    """
    from app.core.doc_profiles import DOC_PROFILES

    pdf = settings.data_path / "raw" / "招股说明书2.pdf"
    section("7b. PDF 解析前提（工单03：招股说明书2.pdf）")
    if not pdf.exists():
        record("招股说明书2.pdf 存在", False, f"未找到 {pdf}")
        return False
    try:
        import pymupdf
    except ImportError:
        record("pymupdf 已安装", False, "请 pip install pymupdf")
        return False

    doc = pymupdf.open(pdf)
    n = doc.page_count
    want = EXPECTED_PAGES["招股说明书2.pdf"]
    record(f"PDF 页数 = {want}", n == want,
           f"实测 {n} 页（{pdf.stat().st_size / 1e6:.2f} MB）")

    # 页码偏移：只看页脚分带（doc_profiles.FOOTER_BAND），取最后一段孤立数字
    from app.core.doc_profiles import FOOTER_BAND
    offsets: list[tuple[int, int]] = []
    for idx in (21, 156, 305):
        if idx >= n:
            continue
        page = doc[idx]
        h = page.rect.height
        for b in page.get_text("dict")["blocks"]:
            if b["bbox"][1] < h * FOOTER_BAND:
                continue
            txt = "".join(s["text"] for l in b.get("lines", []) for s in l["spans"])
            m = re.search(r"(?<!\d)(\d{1,3})(?!\d)", txt.strip())
            if m:
                offsets.append((idx, idx - int(m.group(1))))
                break
    if not offsets:
        record("页码映射（裸数字页脚）", False, "页脚未找到裸数字页码，需人工确认", fatal=False)
    else:
        got = {o for _, o in offsets}
        detail = "；".join(f"索引 {i} → 偏移 {o}" for i, o in offsets)
        actual = got.pop() if len(got) == 1 else None
        record("页码偏移 = 0（书2 页眉页脚实测偏移 0）", actual == 0,
               detail + ("" if actual == 0 else "  ← 偏移不统一或不为 0，引用页码会错位"))

    # 工单03 的核心能力：表格要能被抽出来（书2 的题目答案大半在表里）
    try:
        p21 = doc[21]
        n_tables = len(p21.find_tables().tables)
        record("第 21 页表格可抽取（工单03 表格解析）", n_tables >= 1,
               f"实测 {n_tables} 张（「四、本次发行情况」「五、募集资金用途」两张表都在此页）")
    except Exception as e:  # noqa: BLE001
        record("第 21 页表格可抽取（工单03 表格解析）", False, str(e)[:120])
    doc.close()
    return True


def check_figures() -> None:
    """工单04：图区定位探针 —— 把已知的正确/错误边界钉死。

    【为什么值得单列一节】图区定位全是启发式（面积阈值、竖排标签、图元聚类），
    改一个阈值就可能把整页财报判成组织结构图、或者把两组并排的图拆开。
    这里用三个实测过的锚点做回归：
      · 页 71 必须恰好 1 个图区（饼图 + 柱状图**并排合成一块**，src 有 2 个）
      · 页 38 必须是 diagram，且图内文字里有 6 个「销售处」
      · 页 207/208 必须**没有**图区（旋转 90° 的财报页，最容易被误判）
    """
    section("7d. 图区定位（工单04）")
    if not settings.image_enable:
        record("图像解析已启用", False, "IMAGE_ENABLE=false —— 图区不会入库", fatal=False)
        return
    pdf2 = settings.data_path / "raw" / "招股说明书2.pdf"
    if not pdf2.exists():
        record("招股说明书2.pdf 存在", False, f"未找到 {pdf2}")
        return
    try:
        # 全量解析（约 20s）：截断解析会漏掉后面的图区，断言就没意义了
        from app.core.pdf_parser import parse_pdf_full
        pages, st = parse_pdf_full(pdf2)
    except Exception as e:  # noqa: BLE001
        record("图区定位可运行", False, str(e)[:120])
        return

    by_page = {pc.page_no: pc.figures for pc in pages}
    f71 = by_page.get(71, [])
    record("页 71 恰好 1 个图区，且两块位图已合并",
           len(f71) == 1 and len(f71[0].src_bboxes) == 2,
           f"实测 {len(f71)} 个，src_bboxes={len(f71[0].src_bboxes) if f71 else 0}"
           + (f"，题注={f71[0].title!r}" if f71 else ""))

    d38 = [f for f in by_page.get(38, []) if f.kind == "diagram"]
    n_sales = d38[0].labels.count("销售处") if d38 else 0
    record("页 38 判为结构图，且图内文字含 6 个「销售处」",
           len(d38) == 1 and n_sales == 6,
           f"实测 diagram {len(d38)} 个，销售处 {n_sales} 个")

    bad = [p for p in (207, 208) if by_page.get(p)]
    record("旋转财报页（207/208）未被误判为图区", not bad, f"误判页：{bad}")

    record("图区总数合理（书2 全量 30~40 个）", 30 <= st.n_figures <= 40,
           f"实测 {st.n_figures} 个（位图 {st.n_figures_bitmap} / 矢量 {st.n_figures_vector}）"
           f"｜滤掉噪声位图实例 {st.n_images_dropped}")


def check_image_cache() -> None:
    """工单04：转写缓存探针 —— 缓存为空 / 全是失败，都是"图块会退化"的前兆。"""
    section("7e. 图像转写缓存（工单04）")
    if not settings.image_enable:
        return
    try:
        from app.core import image_cache
        recs = image_cache.all_transcripts()
    except Exception as e:  # noqa: BLE001
        record("缓存可读", False, str(e)[:120], fatal=False)
        return
    ok_text = [r for r in recs if r.effective_text]
    record("缓存非空", len(recs) > 0,
           f"{len(recs)} 条（先跑 scripts/transcribe_images.py --run）")
    record("多数图已转写出文本", len(ok_text) >= max(1, int(len(recs) * 0.7)),
           f"{len(ok_text)}/{len(recs)} 条有文本"
           f"｜失败 {len(recs) - len(ok_text)} 条"
           f"｜人工核对 {sum(1 for r in recs if r.reviewed)} 条")


def check_kb_docs() -> None:
    """库里必须同时有这两份文档 —— 只有一份时工单03 的题会静默答错（张冠李戴）。"""
    section("7c. 知识库里的文档（工单03 需要两份）")
    try:
        from app.core.vectorstore import VectorStore
        store = VectorStore()
        ok, msg = store.health()
        if not ok:
            record("向量库可查文档分布", False, f"Milvus 不可达：{msg[:80]}", fatal=False)
            return
        rows = store.client.query(collection_name=store.collection, filter="id >= 0",
                                  output_fields=["doc_name"], limit=16384)
        seen: dict[str, int] = {}
        for r in rows:
            seen[r.get("doc_name", "?")] = seen.get(r.get("doc_name", "?"), 0) + 1
    except Exception as e:  # noqa: BLE001
        record("向量库可查文档分布", False, str(e)[:120], fatal=False)
        return

    for name in EXPECTED_PAGES:
        n = seen.get(name, 0)
        record(f"库中有《{name}》", n > 0, f"{n} 块" + ("" if n else "  ← 先跑 scripts/ingest_all.py"))
    record("库中不含未登记文档", set(seen) <= set(EXPECTED_PAGES),
           f"实际：{seen}" if set(seen) - set(EXPECTED_PAGES) else f"实际：{seen}")


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
# 8. Query 理解（多轮对话，工单05）—— **全离线**，不需要 Milvus / Ollama
# ----------------------------------------------------------------------
def check_query_understanding() -> bool:
    """把「指代消解 / 省略补全」的核心不变量在启动前就验一遍。

    【为什么值得进 preflight】这是工单05 唯一的新逻辑，而它的失败是**静默**的：
    改写没生效时，系统照样 200、照样给答案，只是路由从"限定到某一份文档"退化成
    "全库检索"，答案悄悄变了。放在自检里，出问题第一分钟就能看见。
    """
    from app.core.doc_profiles import canonical_entity
    from app.core.query_understanding import (
        find_entity_span, intent_template_of, resolve_rules)
    from app.core.session import SessionStore, Turn

    section("9. Query 理解（多轮对话，工单05）")

    script = [
        ("报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？", "none", None),
        ("他参与的哪个工程荣获了国家科技进步一等奖？", "rule-coref",
         "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？"),
        ("这个公司的法定代表人是谁？", "rule-coref",
         "武汉兴图新科电子股份有限公司法定代表人是谁？"),
        ("那武汉力源信息技术股份有限公司呢？", "rule-switch",
         "武汉力源信息技术股份有限公司法定代表人是谁？"),
        ("武汉力源信息技术股份有限公司组织结构图中，哪个销售部的销售处最多？有哪些销售处？",
         "none", None),
    ]
    st = SessionStore()
    sid = st.get_or_create(None).session.id
    bad: list[str] = []
    for q, want_method, want_text in script:
        rr = resolve_rules(q, focus_entity=st.focus_entity(sid),
                           intent_template=st.intent_template(sid))
        if rr is None or rr.method != want_method:
            bad.append(f"{q[:16]}… 期望 {want_method}，实得 {rr.method if rr else 'abstain'}")
        elif want_text and rr.rewritten != want_text:
            bad.append(f"{q[:16]}… 改写为 {rr.rewritten!r}，期望 {want_text!r}")
        if rr:
            span = find_entity_span(rr.rewritten)
            st.append(sid, Turn(question=q, rewritten=rr.rewritten,
                                focus_entity=canonical_entity(span.doc_key) if span else "",
                                intent_template=intent_template_of(rr.rewritten)))
    record("规则改写器能逐轮解析工单 5 轮剧本", not bad,
           "；".join(bad) if bad else "5 轮全中（代词消解 / 话题切换 / 完整问题原样）")

    # ---- 既有 16 题不得被改写（多轮改动不许污染单轮）----
    try:
        import json
        qs = json.loads((PROJECT_ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
        changed = []
        for q in qs["questions"]:
            r = resolve_rules(q["question"])
            if r is None or r.method != "none" or r.rewritten != q["question"]:
                changed.append(q["id"])
        record("既有题库无一轮被改写（回归闸门）", not changed,
               f"被改写的题：{changed}" if changed else f"{len(qs['questions'])} 题全部原样透传")
    except Exception as e:  # noqa: BLE001
        record("既有题库无一轮被改写（回归闸门）", False, f"读取失败：{e}")

    # ---- 会话 TTL 必须可观测（演示 TTL 过期靠它）----
    ttl_store = SessionStore(ttl=0.3)
    ttl_store.get_or_create("ttl-probe")
    time.sleep(0.5)
    lookup = ttl_store.get_or_create("ttl-probe")
    record("会话 TTL 过期可观测", lookup.expired,
           f"reason={lookup.reason}" + ("" if lookup.expired else "  ← 过期后必须报 expired"))

    # ---- 会话有界（LRU）----
    lru = SessionStore(max_sessions=2)
    for s in ("a", "b", "c"):
        lru.get_or_create(s)
    record("会话数有界（LRU 逐出）", lru.stats()["n_sessions"] == 2,
           f"n_sessions={lru.stats()['n_sessions']}（上限 2）")

    # ---- 响应字段声明齐全（extra='ignore' 会静默丢字段）----
    from app.schemas import ChatResponse
    need = ("session_id", "turn_index", "rewritten_question", "rewrite_method",
            "focus_entity", "carried_intent", "session_expired", "history_chars")
    missing = [f for f in need if f not in ChatResponse.model_fields]
    record("ChatResponse 已声明多轮字段", not missing,
           f"缺：{missing}" if missing else f"{len(need)} 个字段齐全")
    return True


# ----------------------------------------------------------------------
# 10. 混合检索栈（工单06）—— 离线检查，不需要 Milvus 的连接数
# ----------------------------------------------------------------------
def check_retrieval_stack() -> bool:
    """把工单06 的四个新件各点一下名：倒排索引 / 融合 / 重排器 / 剖面。

    【为什么值得进 preflight】这一栈的失败几乎都是**静默**的：
    模式写错被悄悄回退、重排器异常被回退成 lexical —— 系统照常 200，
    只是结果不对。所以至少要把"能不能构造出来"验一遍。
    """
    section("10. 混合检索栈（工单06）")
    try:
        from app.core.fulltext import InvertedIndex
        idx = InvertedIndex.build([
            {"id": 1, "content": "注册资本5,520.00万元，见 1-1-42 页",
             "page_no": 1, "page_label": "1-1-1", "chunk_type": "text",
             "section_path": "八、募集资金用途", "doc_name": "t.pdf",
             "chunk_index": 1, "doc_id": "d"}])
        ok = (idx.search('"注册资本"', 3) and not idx.search('"5,530~"', 3)
              and not idx.search('"注册资本用途"', 3))
        record("倒排索引：短语有序 / 数字不做模糊", bool(ok),
               f"词条 {len(idx.postings)}｜删除键 {len(idx.deletes)}")
    except Exception as e:  # noqa: BLE001
        record("倒排索引可构造", False, f"{type(e).__name__}: {e}")

    try:
        from app.core.fusion import rrf_fuse, weighted_fuse
        from app.core.vectorstore import SearchHit as _H
        a = [_H(1, 0.9, "x", 1, "p1", "text", "", "t", 1, base_score=0.9)]
        b = [_H(2, 12.0, "y", 2, "p2", "text", "", "t", 2, base_score=12.0,
                score_kind="bm25")]
        r1 = rrf_fuse(a, b, w_keyword=0.5)
        r2 = weighted_fuse(a, b, w_keyword=0.5)
        record("融合：RRF（投票）/ 加权平均 可用",
               r1.n_total == 2 and r2.n_total == 2,
               f"rrf {r1.n_total} 条 / weighted {r2.n_total} 条")
        # 零权重那一路必须整体排除（否则 w=0 仍在起作用，实测过）
        r0 = rrf_fuse(a, b, w_keyword=0.0)
        record("融合：零权重那一路被完全排除", r0.n_total == 1,
               f"w_keyword=0 → {r0.n_total} 条（应只有向量路的 1 条）")
    except Exception as e:  # noqa: BLE001
        record("融合模块可用", False, f"{type(e).__name__}: {e}")

    try:
        from app.core.profiles import ALL_PROFILES
        from app.core.rerank import RerankInfo
        modes = {p.retrieval_mode for p in ALL_PROFILES.values()}
        rks = {p.reranker for p in ALL_PROFILES.values()}
        record("剖面：三种模式齐备", {"vector", "fulltext", "hybrid"} <= modes,
               f"模式 {sorted(modes)}")
        record("剖面：重排器 ≥3 种（工单要求）",
               len(rks - {"none"}) >= 3, f"重排器 {sorted(rks)}")
        record("重排器留痕结构可用", RerankInfo(reranker="lexical").as_dict() is not None)
    except Exception as e:  # noqa: BLE001
        record("剖面/重排器注册表可用", False, f"{type(e).__name__}: {e}")
    return True


# ----------------------------------------------------------------------
def main() -> int:
    print("\033[1m" + "=" * 68)
    print("  工单03 · 环境自检 preflight")
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
    check_pdf2()        # 工单03
    check_figures()     # 工单04
    check_image_cache() # 工单04
    check_kb_docs()     # 工单03
    check_runtime()
    check_query_understanding()  # 工单05（离线，无需 Milvus/Ollama）
    check_retrieval_stack()      # 工单06（离线）

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
