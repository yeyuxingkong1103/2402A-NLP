"""
图像语义解析模块（工单4 核心 · 多模态大模型路线）
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
工单编号：人工智能NLP-RAG-图像内容解析及检索优化

工单备注硬要求：「PDF 中的图像语义解析使用多模态模型（CLIP 或多模态大模型）实现」。
本模块走**多模态大模型**路线（DashScope 兼容模式的 qwen-vl-plus），
CLIP 路线见 src/clip_encoder.py（负责"以文搜图"的跨模态召回）。

为什么必须用大模型而不是纯 OCR
--------------------------------
两道上机题的答案，光靠"把图里的字抄下来"是拿不到的：

* id=5「销售部有几个部门构成，其中大客户销售部有几个销售处构成？」
  p39 组织图的文字层被逐字打散成「财 / 务 / 部」「销 / 售 / 处」，
  **层级归属关系（6 个销售处挂在"大客户销售部"下面）完全不在文字里**，
  只存在于连接线的拓扑结构中。必须让模型"看图说话"。
* id=6「增长率最快/负增长的是哪个行业？」
  p72/p310 的柱状图里，**负增长那个行业（IC 卡 −2.0%）正文一个字都没提**，
  正文只说"工控 10.5%，位列第二，仅次于汽车电子"。

所以提示词的设计目标是**结构化 + 完整**：层级要逐层展开、数值要逐条列全、
不许用"等"字概括。少列一个子节点，检索就可能差一格。
"""
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import requests

from . import config
from .config import WORK_ORDER_NOS  # noqa: F401  (工单编号，模块标识)

# ------------------------------------------------------------------ 提示词
#
# 提示词里每一条约束都是**针对实测踩到的坑**加的，不要随意删减：
#   * "逐层把每一层都写全，不要省略、不要合并"
#       —— 初版让模型"简要描述"，qwen-vl-plus 输出的销售部子节点里
#          **漏掉了"国际贸易部"**（4 个只写了 3 个），id=5 直接答错。
#   * "数值必须逐项列出，不要只写最大值/最小值"
#       —— 问的是"增长率最快/负增长"，只给极值会让"负增长是哪个"无从判断。
#   * "如果看不清，写（不清），不要编造"
#       —— 组织图缩略后个别小字会糊，宁可漏也不要编，编了会污染检索。
_SYSTEM = (
    "你是一个严谨的招股说明书图像解析器。你的输出会被直接写入检索库供问答系统使用，"
    "因此必须**完整、准确、不遗漏、不编造**。"
)

_USER_TMPL = """请仔细阅读这张从《{doc_name}》第 {page} 页裁出的图片，输出结构化描述。

严格按下面的格式输出，不要添加任何额外的开场白或解释：

【图类型】组织结构图/业务流程图/饼图/柱状图/折线图/示意图/表格/证照照片/其他
【图题】图片的标题（没有就写"无"）
【图内文字】把图中出现的所有文字**逐条完整列出**，用"；"分隔；是表格的按行列出
【结构关系】仅组织结构图/流程图填写。用「父节点 → 子节点1、子节点2、…」的形式，
**逐层把每一层都写全**，一个字都不要省略、不要把多个节点合并成一个
【数据】仅图表填写。把每个类别及其数值/占比**逐条列全**（不要只写最大/最小值），
形如「类别：数值（占比）」
【可回答的问题】这张图能回答的 2~4 个具体问题

硬性要求：
1. 图内所有文字都要抄全，禁止用"等""等等"概括；
2. 组织结构的层级关系必须完整，每个父节点下有几个子节点就要写几个；
3. 数值逐项列出，包含负值；
4. 看不清的写"（不清）"，**绝对不要编造**；
5. 用中文回答。{extra}"""

# 图题作为额外上下文一并喂进去（图题已从正文剔除，只存在于图像块里）
_EXTRA_TMPL = "\n\n已知这张图的图题/资料来源是：{caption}\n请以此校验你读到的标题。"

# ------------------------------------------------------------------ 第二遍：按图类型校验
#
# **为什么必须做第二遍**：同一个 prompt、temperature=0，qwen-vl-plus 的
# 层级/数值对应关系**不稳定**。p39 组织图实测：
#   第 1 次 → 「电话及网络销售部 → 珠海销售处、深圳销售处」「大客户销售部 → 北京、武汉、广州、成都」
#             （6 个销售处被劈成两半，id=5 会答成 4 个）
#   第 2 次 → 「大客户销售部 → 珠海、深圳、北京、武汉、广州、成都」（正确）
# 验收要求"id 1/2/3/4 检索结果准确无误"，靠一次采样赌运气不行。
# 第二遍**只针对容易出错的两类图**（层级图 / 图表），且问法完全不同于第一遍
# —— 让它把注意力集中在一个维度上，实测能稳定拿对：
#   dpi=220 + 结构校验 → 销售部 4 个、大客户销售部 6 个，全对。
#   （dpi=300 反而更差：图被内部缩放后连接线更糊，销售部数成 3 个。）
_VERIFY_STRUCT = """再看一次这张组织结构图/流程图，**只做一件事**：重新核对层级归属。

图中的连接方式是：每个方框底部向下引出一条竖线，竖线接一条**横向母线**，
横向母线上再垂下若干条竖线连到下一层的方框。
**一条横向母线只归属与它竖线相连的那一个父方框**，不要按横向母线的左右跨度去分配子方框。

请逐个父方框核对其直接下属，特别检查"一个父节点下挂多个子节点"的层级，
确认子节点没有被错拆到别的父节点下面。

只输出两行，不要任何解释：
第一行：父方框 → 子方框1、子方框2、…（每个父方框一行，用换行分隔）
第二行：从"数量汇总："开始，逐个写「父方框：N」"""

_VERIFY_DATA = """再看一次这张图表，**只做一件事**：逐项核对每个部分的标签与数值的对应关系。

注意饼图要看清每个扇区引出的引线指向哪个标签；柱状图要看清每根柱子对应的行标签；
折线图要看清图例与线的对应。**不要按照数值大小去猜顺序，严格按图上的位置读。**

只输出两行，不要任何解释：
第一行：从"结构数据："开始，把每个类别及其数值/占比逐条列全，形如「类别：数值（占比）」
第二行：从"负值项："开始，列出所有小于 0 的类别及其数值；没有就写"无"."""


def _post(image_path: Path, prompt: str, model: str) -> tuple[str, str, int]:
    """发一次多模态请求，返回 (文本, 错误, 耗时ms)。"""
    if not config.LLM_API_KEY:
        return "", "未配置 LLM_API_KEY", 0
    try:
        b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    except Exception as exc:  # noqa: BLE001
        return "", f"读图失败: {exc}", 0

    payload = {
        "model": model,
        "messages": [{"role": "system", "content": _SYSTEM},
                     {"role": "user", "content": [
                         {"type": "image_url",
                          "image_url": {"url": f"data:image/png;base64,{b64}"}},
                         {"type": "text", "text": prompt}]}],
        "temperature": 0.0,
        "max_tokens": config.VL_MAX_TOKENS,
    }
    url = config.LLM_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {config.LLM_API_KEY}",
               "Content-Type": "application/json"}
    # 本机环境变量里有 http_proxy，会把发往 DashScope 的请求导向本地代理，实测直接 502
    proxies = {"http": None, "https": None}

    last = ""
    for attempt in range(1, config.VL_MAX_RETRY + 1):
        t0 = time.time()
        try:
            r = requests.post(url, headers=headers, json=payload,
                              timeout=config.VL_TIMEOUT_SECONDS, proxies=proxies)
            ms = int((time.time() - t0) * 1000)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"], "", ms
            last = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as exc:  # noqa: BLE001
            ms = int((time.time() - t0) * 1000)
            last = f"{type(exc).__name__}: {exc}"
        if attempt < config.VL_MAX_RETRY:
            time.sleep(1.5 * attempt)
    return "", last, 0


# 图类型 → 需要哪一路校验
_STRUCT_TYPES = ("组织结构", "流程", "结构图", "架构", "关系图")
_CHART_TYPES = ("饼图", "柱状", "条形", "折线", "曲线", "图表", "统计图")


def verify_kind(fig_type: str) -> str:
    """判定该图需要哪一路第二遍校验：struct / data / ''（不需要）。"""
    t = fig_type or ""
    if any(k in t for k in _STRUCT_TYPES):
        return "struct"
    if any(k in t for k in _CHART_TYPES):
        return "data"
    return ""


@dataclass
class ImageSemantics:
    """一张图的语义解析结果。"""

    doc_key: str
    page: int
    index: int
    caption: str = ""
    raw: str = ""            # 模型原始输出（第一遍）
    fig_type: str = ""       # 【图类型】
    title: str = ""          # 【图题】
    inner_text: str = ""     # 【图内文字】
    structure: str = ""      # 【结构关系】（第二遍校验后会被覆盖）
    data_block: str = ""     # 【数据】（第二遍校验后会被追加/覆盖）
    questions: str = ""      # 【可回答的问题】
    verify_kind: str = ""    # struct / data / ''
    verify_raw: str = ""     # 第二遍原始输出
    verify_ms: int = 0
    verify_error: str = ""
    model: str = ""
    elapsed_ms: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.raw.strip())

    def label(self) -> str:
        """人类可读的定位标签："武汉力源信息技术股份有限公司 第 39 页 第 1 张图"。"""
        doc = config.DOCS_BY_KEY.get(self.doc_key, {})
        name = doc.get("company") or doc.get("name") or self.doc_key
        return f"{name} 第 {self.page} 页 第 {self.index} 张图"

    def as_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ 结构化切分
_SECTIONS = ["图类型", "图题", "图内文字", "结构关系", "数据", "可回答的问题"]


def split_sections(raw: str) -> dict[str, str]:
    """把模型的【xx】分节输出切成字典。

    模型偶尔会把标题写成 ``【图类型】`` 之外的形式（全角/半角括号、缺右括号），
    所以用宽松正则匹配，匹配不到就整段塞进 raw 兜底。
    """
    import re

    out: dict[str, str] = {}
    idxs: list[tuple[int, str, int]] = []
    for name in _SECTIONS:
        m = re.search(r"[【\[]\s*" + re.escape(name) + r"\s*[】\]]", raw)
        if m:
            idxs.append((m.start(), name, m.end()))
    idxs.sort()
    for i, (_, name, end) in enumerate(idxs):
        stop = idxs[i + 1][0] if i + 1 < len(idxs) else len(raw)
        out[name] = raw[end:stop].strip(" \n：:")
    return out


# ------------------------------------------------------------------ 调用
def describe_image(image_path: Path, page: int, doc_name: str, caption: str = "",
                   model: str | None = None, doc_key: str = "", index: int = 0,
                   verify: bool = True) -> ImageSemantics:
    """调多模态大模型解析一张图。

    两遍：
      第一遍 **结构化描述**（图类型/图题/图内文字/结构关系/数据/可回答问题）；
      第二遍 **按图类型校验**（层级图查层级、图表查标签与数值的对应），
      只在第一遍判定为「组织结构图/流程图」或「图表」时执行。
      校验结果**覆盖** structure / data_block 字段 —— 两遍不一致时以第二遍为准，
      因为第二遍只让它盯一个维度，实测更稳。
    """
    model = model or config.VL_MODEL
    res = ImageSemantics(doc_key=doc_key, page=page, index=index, caption=caption, model=model)

    extra = _EXTRA_TMPL.format(caption=caption) if caption else ""
    user = _USER_TMPL.format(doc_name=doc_name, page=page, extra=extra)

    t0 = time.time()
    raw, err, ms = _post(image_path, user, model)
    res.raw, res.error, res.elapsed_ms = raw, err, ms
    if err:
        return res

    secs = split_sections(raw)
    res.fig_type = secs.get("图类型", "")
    res.title = secs.get("图题", "")
    res.inner_text = secs.get("图内文字", "")
    res.structure = secs.get("结构关系", "")
    res.data_block = secs.get("数据", "")
    res.questions = secs.get("可回答的问题", "")

    if verify:
        kind = verify_kind(res.fig_type)
        res.verify_kind = kind
        if kind:
            prompt = _VERIFY_STRUCT if kind == "struct" else _VERIFY_DATA
            vraw, verr, vms = _post(image_path, prompt, model)
            res.verify_raw, res.verify_error, res.verify_ms = vraw, verr, vms
            if not verr and vraw.strip():
                # 校验结果作为**主证据**放在前面，第一遍的原文留在后面兜底
                if kind == "struct":
                    res.structure = vraw.strip()
                else:
                    res.data_block = vraw.strip()
            res.elapsed_ms = int((time.time() - t0) * 1000)
    return res


def render_description(sem: ImageSemantics, fig: dict | None = None) -> str:
    """把语义结果渲染成**入索引的文本**。

    这个字符串就是检索命中的载体，所以：
      * 每一节都带中文标签（"结构关系："、"数据："），保证问题里的
        "组织结构""增长率""销售部"等词能对上；
      * 用自然语句包裹层级（"销售部 → 电话及网络销售部、渠道销售部、大客户销售部、国际贸易部"），
        而不是只堆节点名 —— BM25 需要"销售部"和"大客户销售部"出现在同一句里；
      * 数据类问题（id=6 问"增长率最快/负增长"）必须能把**所有类别**都摆出来，
        只给极值会让"负增长是哪个"无从判断，所以 data 字段整段保留。

    **层级以几何拓扑为准**（fig["topology"]["usable"] 为真时）：
    多模态模型在层级归属上不稳定（p39 三次跑出三种结果，见 diagram_topology 模块注释），
    而"方框 + 连接线"的父子关系由矢量图元的相接关系唯一确定。
    模型读出的层级仍会附在后面并标注来源，便于人工复核时对照。
    """
    parts = [
        f"【图片】{sem.label()}",
        f"图类型：{sem.fig_type or '未知'}",
    ]
    if sem.title and sem.title not in ("无", "（无）"):
        parts.append(f"图题：{sem.title}")
    if sem.caption:
        parts.append(f"图中标注/资料来源：{sem.caption}")

    topo = (fig or {}).get("topology") or {}
    topo_used = bool(topo.get("usable")) and bool((topo.get("render") or "").strip())
    if topo_used:
        parts.append(f"结构关系（按 PDF 矢量连接线几何还原，权威）：\n{topo['render'].strip()}")
        if sem.structure and sem.structure not in ("无", "（无）", ""):
            parts.append(f"（参考）多模态模型读出的层级：{sem.structure}")
    elif sem.structure and sem.structure not in ("无", "（无）"):
        parts.append(f"结构关系：{sem.structure}")

    if sem.data_block and sem.data_block not in ("无", "（无）"):
        tag = "数据（已二次核对）" if sem.verify_kind == "data" and sem.verify_raw.strip() \
            else "数据"
        parts.append(f"{tag}：{sem.data_block}")
    if sem.inner_text and sem.inner_text not in ("无", "（无）"):
        parts.append(f"图内文字：{sem.inner_text}")
    if sem.questions and sem.questions not in ("无", "（无）"):
        parts.append(f"可回答的问题：{sem.questions}")
    text = "\n".join(parts)
    limit = config.IMAGE_DESC_MAX_CHARS
    return text if len(text) <= limit else text[:limit].rstrip()


def parse_figures(figures: list[dict], image_root: Path | None = None,
                  only_pages: set[int] | None = None,
                  only_doc: str = "", verbose: bool = True) -> list[dict]:
    """批量解析图清单，回填 desc 字段。**已解析过的图会被跳过**（支持断点续跑）。"""
    root = image_root or config.ROOT_DIR
    out: list[dict] = []
    total = len(figures)
    t0 = time.time()

    for i, fig in enumerate(figures, 1):
        if only_doc and fig["doc_key"] != only_doc:
            out.append(fig)
            continue
        if only_pages and fig["page"] not in only_pages:
            out.append(fig)
            continue
        if fig.get("desc") and not fig.get("error"):
            out.append(fig)
            continue

        img = root / fig["image"] if fig.get("image") else None
        if not img or not img.exists():
            fig["error"] = f"图片不存在: {fig.get('image')}"
            out.append(fig)
            continue

        doc = config.DOCS_BY_KEY.get(fig["doc_key"], {})
        sem = describe_image(img, fig["page"], doc.get("name", fig["doc_key"]),
                             caption=fig.get("caption", ""), doc_key=fig["doc_key"],
                             index=fig["index"])
        fig["semantics"] = sem.as_dict()
        fig["error"] = sem.error
        fig["desc"] = render_description(sem, fig) if sem.ok else ""
        fig["elapsed_ms"] = sem.elapsed_ms
        out.append(fig)

        if verbose:
            flag = "OK " if sem.ok else "ERR"
            print(f"  [{i:>3}/{total}] {flag} {fig['doc_key']} p{fig['page']}#{fig['index']} "
                  f"{sem.elapsed_ms}ms {sem.fig_type[:12]}")
            if not sem.ok:
                print(f"        {sem.error[:140]}")

    if verbose:
        print(f"  合计 {total} 张，耗时 {time.time() - t0:.1f}s")
    return out


def save_figures(figures: list[dict], path: Path | None = None) -> Path:
    p = path or config.FIGURES_JSON
    p.write_text(json.dumps(figures, ensure_ascii=False, indent=1), encoding="utf-8")
    return p
