# -*- coding: utf-8 -*-
"""
图像语义解析模块（多模态模型：CLIP 粗分类 + 多模态大模型精解析）
工单编号：人工智能NLP-RAG-图像内容解析及检索优化

工单备注明确要求「PDF 中的图像语义解析使用多模态模型（CLIP 或多模态大模型）实现」，
本模块把两者**级联**起来用（详见 docs/技术文档.md 的选型对比）：

    CLIP（粗筛）              多模态大模型（精读）
    判定图属于哪一类   ──▶   按类型选专用 prompt，把图「读」成可检索文本
    · 组织结构图/柱状图/…      · 组织结构图 → 层级树（父子隶属关系）
    · 置信度                   · 图表 → 结构化数据（extract_chart_data）

解析产物：
    results/image_descriptions.json   逐图的结构化解析结果（含检索用文本）
    results/image_descriptions.md     人读版（演示/文档用）

其中 retrieval_text 是真正写入向量库的文本，由四段拼成：
    头信息（文档/页码/图类型） + 自然语言描述 + 结构化数据 + 层级关系展开
这样「组织结构图里销售部有几个下属部门」这类问题既能被向量命中，
也能被 BM25 按节点名（销售部/大客户销售部/销售处）精确命中。

用法：
    python src/image_semantic_parse.py                  # 解析全部图页（默认）
    python src/image_semantic_parse.py --pages 39,72    # 只解析指定页
    python src/image_semantic_parse.py --all            # 连产品照片一起解析
    python src/image_semantic_parse.py --no-cache       # 忽略缓存重新调用模型
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from image_extractor import RESULTS_DIR, WORKDIR, load_inventory  # noqa: E402
from rag_core import config  # noqa: E402
from rag_core.image_parse import VLMImageParser  # noqa: E402

DESC_JSON = RESULTS_DIR / "image_descriptions.json"
DESC_MD = RESULTS_DIR / "image_descriptions.md"
CACHE_JSON = RESULTS_DIR / "cache" / "semantic_cache.json"

# 低检索价值的图类（默认跳过，节省多模态算力）
LOW_VALUE_LABELS = {"产品图", "地图"}


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _abs_path(rec: dict) -> Path:
    """清单里的 file 是相对工单目录的路径，这里还原为绝对路径。"""
    p = Path(rec.get("abs_file") or "")
    if p.exists():
        return p
    return (WORKDIR / rec["file"]).resolve()


def _load_cache() -> dict:
    if CACHE_JSON.exists():
        try:
            return json.loads(CACHE_JSON.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_JSON.parent.mkdir(parents=True, exist_ok=True)
    CACHE_JSON.write_text(json.dumps(cache, ensure_ascii=False, indent=1),
                          encoding="utf-8")


def _cache_key(rec: dict, kind: str) -> str:
    """缓存键：图像文件 + 修改时间 + 解析类型（图变了就重解析）。"""
    p = _abs_path(rec)
    mtime = p.stat().st_mtime if p.exists() else 0
    raw = f"{rec['image_id']}|{mtime}|{kind}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 1. 图类型判定（CLIP 结果 → prompt 类型）
# ---------------------------------------------------------------------------
def decide_kind(rec: dict) -> str:
    """
    决定用哪种 prompt 解析：
      org   —— 组织结构图/股权结构图，走 _ORG_PROMPT（输出层级树）
      auto  —— 其余图表，走 _CHART_PROMPT + extract_chart_data
    """
    caption = rec.get("caption") or ""
    if re.search(r"(组织结构图|股权结构图|结构图)", caption):
        return "org"
    clip = rec.get("clip_labels") or []
    if clip:
        top = clip[0]["label"]
        if top in ("组织结构图", "股权结构图"):
            return "org"
    return "auto"


# ---------------------------------------------------------------------------
# 2. 组织结构图：层级树 → 隶属关系事实
# ---------------------------------------------------------------------------
_INDENT_PAT = re.compile(r"^(\s*)[-*+]?\s*(.+?)\s*$")


def tree_to_facts(markdown: str) -> list[str]:
    """
    把 VLM 输出的 Markdown 层级树展开成「父节点 → 子节点」事实句。

    为什么需要：用户问的是「销售部有几个部门构成」，而层级树只是缩进文本，
    向量检索与 LLM 都不容易稳定地数对。这里把树结构显式展开为
       「销售部 的直接下级（4 个）：渠道销售部、电话及网络销售部、…」
    数量由树结构本身统计得出，不做任何人工设定。

    缩进层级按「空格/制表符宽度」推断；解析失败时返回空列表（不影响主流程）。
    """
    stack: list[tuple[int, str]] = []          # [(缩进宽度, 节点名)]
    children: dict[str, list[str]] = {}
    order: list[str] = []

    for raw in (markdown or "").split("\n"):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        m = _INDENT_PAT.match(raw.replace("\t", "    "))
        if not m:
            continue
        indent, name = len(m.group(1)), m.group(2).strip()
        # 去掉「- 」已在上面的正则处理；剔除纯符号行
        if not name or set(name) <= set("-*+ "):
            continue
        # 去掉行内的加粗/代码标记与「1.」「（1）」式编号
        name = name.strip("`*_ ").strip()
        name = re.sub(r"^(\d{1,2}[、.．)）]|（\d{1,2}）)\s*", "", name).strip()
        if not name:
            continue
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if stack:
            parent = stack[-1][1]
            children.setdefault(parent, [])
            if name not in children[parent]:
                children[parent].append(name)
        stack.append((indent, name))
        if name not in order:
            order.append(name)

    facts: list[str] = []
    for parent in order:
        kids = children.get(parent)
        if not kids:
            continue
        facts.append(f"{parent} 的直接下级（{len(kids)} 个）：{'、'.join(kids)}")
    return facts


# ---------------------------------------------------------------------------
# 3. 结构化数据 → 可检索文本
# ---------------------------------------------------------------------------
def chart_data_to_text(data: dict) -> str:
    """把 extract_chart_data 的 JSON 展平成中文行，便于 BM25/向量检索。"""
    if not data or data.get("_error"):
        return ""
    lines: list[str] = []
    if data.get("title"):
        lines.append(f"图标题：{data['title']}")
    if data.get("type"):
        lines.append(f"图表类型：{data['type']}")
    axes = []
    if data.get("x_axis"):
        axes.append(f"横轴：{data['x_axis']}")
    if data.get("y_axis"):
        axes.append(f"纵轴：{data['y_axis']}")
    if axes:
        lines.append("；".join(axes))
    for s in data.get("series", []) or []:
        name = s.get("name") or "数值"
        items = s.get("data") or {}
        if not items:
            continue
        pairs = "，".join(
            f"{k}：{v if v is not None else '未读出'}" for k, v in items.items())
        lines.append(f"{name} —— {pairs}")
        # 最值提示（帮助回答「增长最快/负增长」类问题）
        nums = [(k, v) for k, v in items.items()
                if isinstance(v, (int, float))]
        if len(nums) >= 2:
            hi = max(nums, key=lambda kv: kv[1])
            lo = min(nums, key=lambda kv: kv[1])
            lines.append(f"{name} 中的最大值：{hi[0]}（{hi[1]}）；"
                         f"最小值：{lo[0]}（{lo[1]}）")
            neg = [k for k, v in nums if v < 0]
            if neg:
                lines.append(f"{name} 中为负值的类别：{'、'.join(neg)}")
    if data.get("notes"):
        lines.append(f"图注：{data['notes']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. 单图解析主流程
# ---------------------------------------------------------------------------
def parse_image(rec: dict, vlm: VLMImageParser, kind: str,
                use_cache: bool = True, cache: dict | None = None) -> dict:
    """对一张图做「CLIP 分类 + VLM 描述（+图表结构化）」，返回解析记录。"""
    cache = cache if cache is not None else {}
    key = _cache_key(rec, kind)
    if use_cache and key in cache:
        out = dict(cache[key])
        out["cached"] = True
        return out

    img_path = _abs_path(rec)
    result: dict = {
        "image_id": rec["image_id"],
        "doc": rec["doc"],
        "page": rec["page"],
        "source": rec["source"],
        "file": rec["file"],
        "caption": rec.get("caption") or "",
        "kind": kind,
        "clip_labels": rec.get("clip_labels") or [],
        "chart_type": (rec.get("clip_labels") or [{}])[0].get("label", "")
                      if rec.get("clip_labels") else "",
        "vlm_description": "",
        "chart_data": {},
        "hierarchy_facts": [],
        "error": "",
        "cached": False,
    }

    if not img_path.exists():
        result["error"] = f"图像文件缺失：{img_path}"
        return result

    t0 = time.perf_counter()
    try:
        # --- 多模态大模型：自然语言/层级描述 ---
        desc = vlm.describe(img_path, kind=kind)
        result["vlm_description"] = desc or ""
        if not desc:
            result["error"] = ("多模态模型未返回内容：请检查 RAG_VLM_MODEL / "
                               "Ollama 视觉模型是否就绪")

        # --- 图表：额外抽结构化数据（数值类问题的关键）---
        if kind == "auto":
            data = vlm.extract_chart_data(img_path)
            result["chart_data"] = data or {}
            if data and data.get("type"):
                result["chart_type"] = data["type"]
        else:
            # 组织结构图：把层级树展开成隶属关系事实
            result["hierarchy_facts"] = tree_to_facts(result["vlm_description"])
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    result["parse_seconds"] = round(time.perf_counter() - t0, 2)
    cache[key] = result
    return result


def build_retrieval_text(r: dict) -> str:
    """
    组装最终写入知识库的「图像语义文本」。

    结构固定为四段，保证：
      · 头信息让模型知道信息来自哪份文档的第几页、是什么图（可溯源）；
      · 描述段覆盖自然语言提问；
      · 数据段覆盖数值提问；
      · 层级事实段覆盖「A 有几个下级」这类计数提问。
    """
    head = f"[图像] 《{r['doc']}》第{r['page']}页｜类型：{r.get('chart_type') or '图表'}"
    if r.get("caption"):
        head += f"｜标题：{r['caption']}"
    if r.get("source") == "page_render":
        head += "｜来源：矢量图整页渲染"

    parts = [head]
    if r.get("vlm_description"):
        parts.append("【图像语义描述】\n" + r["vlm_description"])
    data_txt = chart_data_to_text(r.get("chart_data") or {})
    if data_txt:
        parts.append("【图表结构化数据】\n" + data_txt)
    if r.get("hierarchy_facts"):
        parts.append("【层级隶属关系（由层级树自动展开）】\n"
                     + "\n".join(r["hierarchy_facts"]))
    if r.get("error"):
        parts.append(f"（解析备注：{r['error']}）")
    return "\n".join(parts).strip()


# ---------------------------------------------------------------------------
# 5. 批量入口
# ---------------------------------------------------------------------------
def select_targets(records: list[dict], pages: list[int] | None,
                   all_images: bool) -> list[dict]:
    """挑选待解析的图：默认只解析「图页」上的图（跳过产品照片等低价值图）。"""
    out = []
    for r in records:
        if pages and r["page"] not in pages:
            continue
        if not all_images:
            if not r.get("figure_page"):
                continue
            top = (r.get("clip_labels") or [{}])
            if top and top[0].get("label") in LOW_VALUE_LABELS:
                continue
        out.append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="图像语义解析（CLIP + 多模态大模型）")
    ap.add_argument("--pages", default="", help="只解析指定页，逗号分隔（如 39,72,310）")
    ap.add_argument("--doc", default="", help="只解析指定文档（如 招股说明书2）")
    ap.add_argument("--all", action="store_true", help="解析全部图（含产品照片）")
    ap.add_argument("--no-cache", action="store_true", help="忽略缓存，强制重解析")
    ap.add_argument("--max-images", type=int, default=0, help="最多解析张数（0=不限）")
    args = ap.parse_args()

    pages = [int(x) for x in args.pages.split(",") if x.strip()] or None
    inv = load_inventory()
    records = inv["images"]
    if args.doc:
        records = [r for r in records if r["doc"] == args.doc]
    targets = select_targets(records, pages, args.all)
    if args.max_images:
        targets = targets[: args.max_images]

    if not targets:
        print("[warn] 没有待解析的图，请检查 --pages/--doc 或先运行 image_extractor.py")
        return

    print(f"[解析] 待处理 {len(targets)} 张图"
          f"（来自清单 {len(records)} 张，已按图页/低价值图筛选）")
    vlm = VLMImageParser()
    cache = _load_cache()
    results: list[dict] = []
    t0 = time.perf_counter()

    for i, rec in enumerate(targets, 1):
        kind = decide_kind(rec)
        print(f"  [{i}/{len(targets)}] 《{rec['doc']}》第{rec['page']}页 "
              f"{rec['source']} → prompt={kind} …", end="", flush=True)
        r = parse_image(rec, vlm, kind, use_cache=not args.no_cache, cache=cache)
        r["retrieval_text"] = build_retrieval_text(r)
        results.append(r)
        flag = "缓存" if r.get("cached") else f"{r.get('parse_seconds', 0)}s"
        print(f" {'OK' if r['retrieval_text'] else '空'} ({flag})"
              + (f" [{r['error'][:60]}]" if r.get("error") else ""))
        _save_cache(cache)                       # 逐张落盘，中断可续

    save_descriptions(results, inv)
    ok = sum(1 for r in results if r.get("vlm_description"))
    print(f"\n[完成] {ok}/{len(results)} 张解析成功，"
          f"耗时 {time.perf_counter() - t0:.1f}s")
    print(f"        {DESC_JSON}")
    print(f"        {DESC_MD}")


def save_descriptions(results: list[dict], inv: dict) -> Path:
    """保存解析结果（JSON + Markdown）。"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "workorder": "人工智能NLP-RAG-图像内容解析及检索优化",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "extractor": "CLIP(零样本分类) + 多模态大模型(语义描述/结构化抽取)",
        "vlm_backend": _describe_vlm_backend(),
        "n_images": len(results),
        "n_ok": sum(1 for r in results if r.get("vlm_description")),
        "n_page_render": sum(1 for r in results if r.get("source") == "page_render"),
        "images": results,
    }
    DESC_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    md = [
        "# 图像语义解析结果",
        "",
        "工单编号：人工智能NLP-RAG-图像内容解析及检索优化",
        "",
        f"- 生成时间：{payload['generated_at']}",
        f"- 解析器：{payload['extractor']}",
        f"- VLM 后端：{payload['vlm_backend']}",
        f"- 成功解析：{payload['n_ok']}/{payload['n_images']}"
        f"（其中矢量图整页渲染 {payload['n_page_render']} 张）",
        "",
    ]
    for r in results:
        md += [
            f"## {r['image_id']}（第 {r['page']} 页｜{r['source']}）",
            "",
            f"- 图类型：{r.get('chart_type') or '—'}"
            + (f"｜题注：{r['caption']}" if r.get("caption") else ""),
            f"- 图像文件：`{r['file']}`",
            f"- CLIP 分类："
            + ("、".join(f"{c['label']}({c['score']})"
                         for c in (r.get("clip_labels") or [])) or "—"),
            "",
        ]
        if r.get("vlm_description"):
            md += ["**多模态语义描述**", "", r["vlm_description"], ""]
        if r.get("hierarchy_facts"):
            md += ["**层级隶属关系**", ""]
            md += [f"- {x}" for x in r["hierarchy_facts"]]
            md.append("")
        if r.get("chart_data"):
            md += ["**图表结构化数据**", "",
                   "```json",
                   json.dumps(r["chart_data"], ensure_ascii=False, indent=2),
                   "```", ""]
        if r.get("error"):
            md += [f"> 解析备注：{r['error']}", ""]
        md += ["---", ""]
    DESC_MD.write_text("\n".join(md), encoding="utf-8")
    return DESC_JSON


def _describe_vlm_backend() -> str:
    """记录实际使用的 VLM 后端，便于复现实验。"""
    import os
    if os.environ.get("RAG_VLM_MODEL"):
        return f"OpenAI 兼容多模态接口（{os.environ['RAG_VLM_MODEL']}）"
    try:
        from rag_core.image_parse import _pick_ollama_vlm
        return f"Ollama 本地视觉模型（{_pick_ollama_vlm()}）"
    except Exception:
        return "未探测到可用后端"


def load_descriptions() -> dict:
    """供建索引脚本读取解析结果。"""
    if not DESC_JSON.exists():
        raise FileNotFoundError(
            f"未找到解析结果 {DESC_JSON}，请先运行：python src/image_semantic_parse.py")
    return json.loads(DESC_JSON.read_text(encoding="utf-8"))


def text_by_page(doc: str | None = None) -> dict[tuple[str, int], list[str]]:
    """
    返回 {(文档名, 页码): [检索文本, ...]}，供 build_multimodal_index 回填。
    同一页多张图时按清单顺序拼接成多个块（各自成 chunk）。
    """
    data = load_descriptions()
    out: dict[tuple[str, int], list[str]] = {}
    for r in data["images"]:
        if doc and r["doc"] != doc:
            continue
        txt = r.get("retrieval_text") or ""
        if txt.strip():
            out.setdefault((r["doc"], r["page"]), []).append(txt)
    return out


if __name__ == "__main__":
    main()
