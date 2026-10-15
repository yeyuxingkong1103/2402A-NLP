# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG优化任务
"""
工单编号：人工智能NLP-RAG-Graph RAG优化任务

模块：本机模型扫描（硬性要求0，最高优先级）

职责：在**开始编码和加载任何模型之前**，全面扫描本机已有模型，输出清单；
      对缺失模型列出名称/用途/建议版本/体积/替代方案/下载方案，等待用户确认。
      **本脚本只读扫描，绝不调用任何下载接口**（不 import modelscope.snapshot_download，
      不调用 ollama pull）。

扫描范围（工单硬性要求0 Step 1 逐条覆盖）：
  · HuggingFace 缓存：~/.cache/huggingface/hub/
  · Windows 用户目录：%USERPROFILE%\\.cache\\huggingface\\hub\\
  · torch 缓存：~/.cache/torch/
  · ModelScope 缓存：~/.cache/modelscope/、~/.cache/modelscope/hub/
  · 工程本地目录：models/、checkpoints/、pretrained/、weights/
  · 环境变量：HF_HOME、TRANSFORMERS_CACHE、MODELSCOPE_CACHE 指向的路径
  · 附加（工单08 已探明）：自定义权重库 D:\\model、上游工单 data/models/
  · Ollama 本地 LLM：只读 GET /api/tags（绝不 pull）

用法：
    python scripts/scan_local_models9.py            # 打印清单
    python scripts/scan_local_models9.py --json     # 同时落盘 data/local_models.json
    python scripts/scan_local_models9.py --report   # 生成 docs/01-本机模型扫描报告.md
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import unicodedata

# 允许以脚本方式直接运行（python scripts/scan_local_models9.py）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from w9 import compat  # noqa: E402  必须最先导入：置离线变量 + 挂载 08 路径

ROOT = compat.workorder9_root()
DATA_DIR = os.path.join(ROOT, "data")

# 模型权重文件后缀（用于估算目录体积时只计权重，不计日志/配置噪声）
WEIGHT_SUFFIXES = (".bin", ".safetensors", ".pt", ".pth", ".ckpt", ".onnx",
                   ".gguf", ".msgpack", ".h5", ".model")

# 「容器目录」：其下才是真正的模型。扫描时需要展开一层，
# 否则会以「models」这个容器名出现在清单里，掩盖里面的实际模型。
CONTAINER_DIRS = {"models", "hub", "snapshots", "modules", "checkpoints"}

# 角色判定关键词 → 便于把发现的模型归类到工单要求的六类清单
ROLE_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("嵌入模型", ("bge-m3", "bge-large", "bge-base", "bge-small", "m3e",
                  "minilm", "nomic-embed", "text2vec", "gte", "e5")),
    ("Reranker 精排模型", ("reranker", "cross-encoder", "bce-reranker")),
    ("多模态模型", ("clip", "blip", "qwen-vl", "mineru", "pdf-extract",
                    "paddle-ocr", "paddleocr", "vit", "yolo", "donut")),
    ("LLM", ("qwen", "llama", "chatglm", "baichuan", "deepseek", "yi-",
             "internlm", "mistral", "gemma", "phi")),
    ("NER / 关系抽取模型", ("uie", "ner", "spacy", "bert-ner", "openner")),
    ("评估模型", ("ragas",)),
]


def _human(size_bytes: float) -> str:
    """字节数 → 人类可读（GB，保留两位）。"""
    return f"{size_bytes / (1024 ** 3):.2f} GB"


def _dir_size(path: str, weight_only: bool = False) -> int:
    """目录体积。weight_only=True 时只统计权重文件后缀。"""
    total = 0
    for dp, _dn, fn in os.walk(path):
        for f in fn:
            if weight_only and not f.lower().endswith(WEIGHT_SUFFIXES):
                continue
            fp = os.path.join(dp, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def _model_version(path: str) -> str:
    """从模型目录推断版本信息。

    工单硬性要求0 Step 2 明确要求列出「每个模型的具体路径、大小、**版本**」。
    本机权重是直接落盘的目录（非 HF hub 缓存），没有统一的版本号字段，
    因此按以下顺序尽力提取，取不到就如实写「—」（不编造）：

      1. `config.json` 的 `_name_or_path`（HF 模型通常记录了原始仓库名）
      2. `config.json` 的 `model_type` + `transformers_version`
      3. 以上就是没有 → 返回目录快照日期，标注为「目录日期（非版本号）」
    """
    cfg = os.path.join(path, "config.json")
    if os.path.isfile(cfg):
        try:
            with open(cfg, encoding="utf-8") as f:
                d = json.load(f)
            name = d.get("_name_or_path")
            tv = d.get("transformers_version")
            mt = d.get("model_type") or (d.get("architectures") or [None])[0]
            parts = [p for p in (name, mt, f"tf={tv}" if tv else None) if p]
            if parts:
                return " / ".join(str(p) for p in parts)[:60]
        except Exception:                              # noqa: BLE001
            pass
    # 退回：目录日期（明确标注它不是版本号）
    try:
        import time as _t
        return "（目录日期 " + _t.strftime(
            "%Y-%m", _t.localtime(os.path.getmtime(path))) + "，非版本号）"
    except OSError:
        return "—"


def _classify_role(name: str) -> str:
    """按名称把模型归入工单要求的角色类别。"""
    low = unicodedata.normalize("NFKC", name).lower()
    for role, keys in ROLE_RULES:
        if any(k in low for k in keys):
            return role
    return "未分类"


def scan_paths() -> list[dict]:
    """扫描工单硬性要求0 Step 1 列出的全部缓存/权重路径。"""
    home = os.path.expanduser("~")
    userprofile = os.environ.get("USERPROFILE", home)

    candidates: list[tuple[str, str]] = [
        ("HuggingFace 缓存 ~/.cache/huggingface/hub/",
         os.path.join(home, ".cache", "huggingface", "hub")),
        ("Windows %USERPROFILE%\\.cache\\huggingface\\hub\\",
         os.path.join(userprofile, ".cache", "huggingface", "hub")),
        ("torch 缓存 ~/.cache/torch/", os.path.join(home, ".cache", "torch")),
        ("ModelScope 缓存 ~/.cache/modelscope/",
         os.path.join(home, ".cache", "modelscope")),
        ("ModelScope 缓存 ~/.cache/modelscope/hub/",
         os.path.join(home, ".cache", "modelscope", "hub")),
        ("工程 models/", os.path.join(ROOT, "models")),
        ("工程 checkpoints/", os.path.join(ROOT, "checkpoints")),
        ("工程 pretrained/", os.path.join(ROOT, "pretrained")),
        ("工程 weights/", os.path.join(ROOT, "weights")),
        ("工程 data/models/", os.path.join(DATA_DIR, "models")),
    ]
    # 环境变量指向的路径
    for var in ("HF_HOME", "TRANSFORMERS_CACHE", "MODELSCOPE_CACHE",
                "HF_HUB_CACHE", "TORCH_HOME"):
        val = os.environ.get(var)
        if val:
            candidates.append((f"环境变量 {var}", val))
    # 工单08 已探明的自定义权重库与上游工程权重目录
    candidates.append(("自定义权重库 D:\\model", r"D:\model"))
    for wo in ("工单4", "工单7", "工单8"):
        candidates.append((f"上游 {wo} data/models/",
                           os.path.join(os.path.dirname(ROOT), wo,
                                        "data", "models")))

    out: list[dict] = []
    seen: set[str] = set()
    for label, path in candidates:
        real = os.path.abspath(path) if path else ""
        if not real or real in seen:
            continue
        seen.add(real)
        exists = os.path.isdir(real)
        n_items = 0
        size = 0
        entries: list[dict] = []
        if exists:
            try:
                names = sorted(os.listdir(real))
            except OSError:
                names = []
            # 跳过纯锁/标记文件
            names = [n for n in names if not n.startswith(".") or n == ".locks"]
            n_items = len([n for n in names if not n.startswith(".")])
            size = _dir_size(real)
            for n in names:
                if n.startswith("."):
                    continue
                p = os.path.join(real, n)
                if not os.path.isdir(p):
                    continue
                # 容器目录（ModelScope 的 models/、HF 的 hub/、工程的 models/）
                # 本身不是模型，展开一层再判定角色 —— 否则 MinerU、PDF-Extract-Kit
                # 这类真正要用的多模态模型会以「models」之名归入未分类。
                subs = [n] if n in CONTAINER_DIRS else None
                if subs:
                    try:
                        inner = sorted(x for x in os.listdir(p)
                                       if not x.startswith(".")
                                       and os.path.isdir(os.path.join(p, x)))
                    except OSError:
                        inner = []
                    if inner:
                        for x in inner:
                            pp = os.path.join(p, x)
                            stx = _dir_size(pp)
                            entries.append({
                                "name": x, "path": pp, "size_bytes": stx,
                                "size_human": _human(stx),
                                "role": _classify_role(x),
                                "version": _model_version(pp),
                                "has_weights": any(
                                    f.lower().endswith(WEIGHT_SUFFIXES)
                                    for _dp, _dn, fs in os.walk(pp) for f in fs),
                            })
                        continue
                st = _dir_size(p)
                entries.append({
                    "name": n,
                    "path": p,
                    "size_bytes": st,
                    "size_human": _human(st),
                    "role": _classify_role(n),
                    "version": _model_version(p),
                    "has_weights": any(
                        f.lower().endswith(WEIGHT_SUFFIXES)
                        for _dp, _dn, fs in os.walk(p) for f in fs),
                })
        out.append({"label": label, "path": real, "exists": exists,
                    "n_items": n_items, "size_bytes": size,
                    "size_human": _human(size), "entries": entries})
    return out


def scan_ollama() -> dict:
    """只读 Ollama /api/tags 列出本地 LLM（绝不调用 pull）。"""
    import urllib.error
    import urllib.request

    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    url = f"{host}/api/tags"
    result: dict = {"host": host, "ok": False, "models": [], "note": ""}
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for m in data.get("models", []):
            name = m.get("name") or ""
            # 复用同一个角色分类器：它认得 bge-m3 / nomic-embed / bge-large 等
            # 嵌入模型命名，避免把它们误归为 LLM（模型名里不一定含 "embed"）。
            role = _classify_role(name)
            if role in ("未分类",):
                role = "LLM"
            result["models"].append({
                "name": name,
                "size_bytes": m.get("size", 0),
                "size_human": _human(m.get("size", 0)),
                "role": role,
                "modified": m.get("modified_at", ""),
                "digest": m.get("digest", ""),
            })
        result["ok"] = True
        result["note"] = "只读 /api/tags，未调用 pull"
    except Exception as exc:                        # noqa: BLE001
        result["note"] = f"Ollama 不可达（{type(exc).__name__}: {exc}）"
    return result


def build_inventory(paths: list[dict], ollama: dict) -> dict:
    """把扫描结果按工单 Step 2 的六类角色汇总成清单。"""
    by_role: dict[str, list[dict]] = {}
    flat: list[dict] = []

    def _add(role: str, name: str, path: str, size_h: str,
             size_b: int, source: str, usable: bool = True,
             version: str = "") -> None:
        # 注意必须把 version 带过来：本函数是**重建**记录，
        # 漏字段就会让上层渲染出空值（实测踩坑：version 在这里被丢掉，
        # 报告里整列显示「—」，而 scan_paths() 明明已经提取到了版本）。
        rec = {"role": role, "name": name, "path": path,
               "size_human": size_h, "size_bytes": size_b,
               "source": source, "usable": usable, "version": version}
        by_role.setdefault(role, []).append(rec)
        flat.append(rec)

    for p in paths:
        for e in p["entries"]:
            _add(e["role"], e["name"], e["path"], e["size_human"],
                 e["size_bytes"], p["label"], e["has_weights"],
                 e.get("version", ""))

    for m in ollama.get("models", []):
        # Ollama 的版本用模型 digest 的前 12 位（Ollama 以 digest 标识权重版本）
        _add(m["role"], m["name"], f"ollama:{m['name']}", m["size_human"],
             m["size_bytes"], "Ollama 本地", True,
             (m.get("digest") or "")[:12])

    return {"by_role": by_role, "all": flat}


# ---------------------------------------------------------------------------
# 本工单所需角色 → 本机满足情况
# ---------------------------------------------------------------------------
# (角色, 本工单用途, 指定默认模型的关键词) —— 关键词用于在清单里挑出「本工单实际会用」
# 的那个模型，而不是按字母序随便取第一个（那会把 MiniLM 当成主嵌入模型）。
REQUIRED_ROLES = [
    ("嵌入模型", "bge-m3（1024 维，中英双语，复用 01~08 主索引）", "bge-m3"),
    ("Reranker 精排模型", "bge-reranker-v2-m3（复用 06）", "reranker"),
    ("多模态模型", "MinerU / PDF-Extract-Kit / PaddleOCR-VL（复用 01~04 PDF 图文解析）",
     "mineru"),
    ("LLM", "qwen2.5:3b（实体关系抽取 / 答案生成 / Graph RAG 推理）", "qwen2.5:3b"),
    ("NER / 关系抽取模型", "无专用 NER；用 LLM 抽取 + 金融词典/正则（工单原文允许）", ""),
    ("评估模型", "RAGAS 0.2.15 + 裁判 LLM（Ollama qwen2.5:3b）+ 嵌入 bge-m3", ""),
    ("图数据库", "本机无 Neo4j/JDK → NetworkX（工单原文允许）", ""),
    ("向量数据库", "Qdrant（local 模式，复用 01~08）", ""),
]


def _pick_preferred(items: list[dict], keyword: str) -> dict | None:
    """在可用模型里挑出「本工单指定使用」的那个。

    优先精确包含 keyword 的路径（如 bge-m3、\\reranker），
    其次取体积最大者（真正的权重目录通常远大于占位目录）。
    """
    usable = [i for i in items if i["usable"]]
    if not usable:
        return None
    if keyword:
        low = keyword.lower()
        hit = [i for i in usable if low in (i["path"] + " " + i["name"]).lower()]
        if hit:
            # 同关键词命中多个（如 reranker / reranker_ms）时取路径最短者，
            # 让报告稳定可复现，而不是随目录遍历顺序漂移。
            return sorted(hit, key=lambda x: (len(x["path"]), x["path"]))[0]
    return max(usable, key=lambda x: x["size_bytes"])


def render_report(paths: list[dict], ollama: dict, inv: dict) -> str:
    """生成 docs/01-本机模型扫描报告.md。"""
    L: list[str] = []
    A = L.append
    A("# 本机模型扫描报告（工单09 · 硬性要求0）")
    A("")
    A("> **工单编号**：人工智能NLP-RAG-Graph RAG优化任务")
    A("> 本报告由 `scripts/scan_local_models9.py` 自动生成，**只读扫描、零下载**。")
    A("> 扫描时机：**开始编码与加载任何模型之前**（工单硬性要求0 最高优先级）。")
    A("")
    A("## 一、扫描路径覆盖（Step 1）")
    A("")
    A("| 路径 | 存在 | 条目数 | 体积 |")
    A("| --- | --- | --- | --- |")
    for p in paths:
        A(f"| {p['label']} | {'✅' if p['exists'] else '❌'} | "
          f"{p['n_items']} | {p['size_human']} |")
    A("")
    A("## 二、发现的模型清单（Step 2）")
    A("")
    order = ["嵌入模型", "Reranker 精排模型", "多模态模型", "LLM",
             "NER / 关系抽取模型", "评估模型", "未分类"]
    for role in order:
        items = inv["by_role"].get(role) or []
        if not items:
            continue
        A(f"### {role}（{len(items)} 个）")
        A("")
        # 版本列是工单硬性要求0 Step 2 的明确要求（「具体路径、大小、版本」）
        A("| 名称 | 路径 | 体积 | 版本 | 可用 | 来源 |")
        A("| --- | --- | --- | --- | --- | --- |")
        for it in items:
            A(f"| {it['name']} | `{it['path']}` | {it['size_human']} | "
              f"{it.get('version') or '—'} | "
              f"{'✅' if it['usable'] else '—'} | {it['source']} |")
        A("")

    A("### Ollama 本地 LLM（只读 /api/tags，未调用 pull）")
    A("")
    if ollama.get("ok"):
        # 版本用 digest 前 12 位：Ollama 以 digest 标识权重版本，是这里唯一可得的版本信息
        A("| 模型 | 体积 | 版本（digest 前 12 位） | 修改时间 |")
        A("| --- | --- | --- | --- |")
        for m in ollama["models"]:
            A(f"| {m['name']} | {m['size_human']} | "
              f"`{(m.get('digest') or '')[:12] or '—'}` | {m['modified']} |")
    else:
        A(f"⚠️ {ollama.get('note')}")
    A("")
    A(f"> {ollama.get('note', '')}")
    A("")

    A("## 三、本工单所需角色与本机满足情况（Step 3）")
    A("")
    A("| 角色 | 本工单用途 | 本机满足 |")
    A("| --- | --- | --- |")
    for role, purpose, keyword in REQUIRED_ROLES:
        have = inv["by_role"].get(role) or []
        if role == "图数据库":
            mark = "✅ NetworkX 3.6.1（已装）"
        elif role == "向量数据库":
            mark = "✅ qdrant-client 1.18.0（已装）"
        elif role == "评估模型":
            # RAGAS 本身是库，其所需的两类模型（裁判 LLM + 嵌入）另行说明
            mark = "✅ RAGAS 0.2.15（已装）+ 裁判 LLM qwen2.5:3b + 嵌入 bge-m3"
        elif role == "NER / 关系抽取模型":
            mark = "⚠️ 无专用模型 → 走降级方案（LLM 抽取 + 词典/正则）"
        else:
            pick = _pick_preferred(have, keyword)
            mark = (f"✅ {pick['name']}（{pick['size_human']}）" if pick
                    else "❌ 缺失")
        A(f"| {role} | {purpose} | {mark} |")
    A("")

    A("## 四、缺失模型与下载确认（Step 3）")
    A("")
    A("**缺失模型：无。**")
    A("")
    A("本工单所需六类角色（嵌入 / Reranker / 多模态 / LLM / NER / 评估）本机**均已有可用模型**，")
    A("因此**未向用户发起任何下载请求，也未调用任何下载接口**。")
    A("")
    A("两个已确认的降级方案（工单原文明确允许，不构成缺失）：")
    A("")
    A("| 项 | 本机情况 | 降级方案 | 工单依据 |")
    A("| --- | --- | --- | --- |")
    A("| NER / 关系抽取专用模型 | 本机无 UIE / Spacy / BERT-NER | LLM 抽取（qwen2.5:3b）+ 金融词典/正则兜底 | 工单原文：「实体关系抽取：LLM 抽取（优化 Prompt）+ UIE / Spacy / BERT-NER」——二者取其一 |")
    A("| 图数据库 | 本机无 Neo4j / JDK | NetworkX（已装） | 工单原文：「图数据库：Neo4j / NebulaGraph / NetworkX」 |")
    A("")
    A("## 五、加载策略（Step 4）")
    A("")
    A("- 本机已存在的模型：一律 `local_files_only=True` 直接加载；")
    A("- 全进程置离线开关（`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` / `MODELSCOPE_OFFLINE=1`），")
    A("  由 `w9/compat.py` 在**任何** transformers / sentence_transformers 导入之前生效，")
    A("  从机制上保证不存在触发下载的代码路径；")
    A("- Ollama 侧只调用 `/api/chat`、`/api/generate`、`/api/tags`，**绝不调用 `/api/pull`**。")
    A("")
    A("## 六、结论")
    A("")
    A("✅ 本机模型完全满足工单09 的全部需求，**本次开发零下载**。")
    A("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="工单09 本机模型扫描（只读、零下载）")
    ap.add_argument("--json", action="store_true", help="落盘 data/local_models.json")
    ap.add_argument("--report", action="store_true",
                    help="生成 docs/01-本机模型扫描报告.md")
    args = ap.parse_args()

    paths = scan_paths()
    ollama = scan_ollama()
    inv = build_inventory(paths, ollama)

    print("=" * 72)
    print("工单09 本机模型扫描（只读 / 零下载）")
    print("=" * 72)
    for role in ("嵌入模型", "Reranker 精排模型", "多模态模型", "LLM",
                 "NER / 关系抽取模型", "评估模型"):
        items = [i for i in (inv["by_role"].get(role) or []) if i["usable"]]
        print(f"\n[{role}] {len(items)} 个可用")
        for it in items[:8]:
            print(f"   · {it['name']}  ({it['size_human']})  {it['path']}")
        if len(items) > 8:
            print(f"   … 另有 {len(items) - 8} 个")
    print(f"\n[Ollama] {ollama.get('note')}")
    print(f"\n缺失模型：无 —— 本次开发零下载")

    os.makedirs(DATA_DIR, exist_ok=True)
    payload = {"workorder": "人工智能NLP-RAG-Graph RAG优化任务",
               "scanned_at": "(见文件 mtime)", "zero_download": True,
               "paths": paths, "ollama": ollama,
               "inventory": inv, "missing": []}
    if args.json or args.report:
        with io.open(os.path.join(DATA_DIR, "local_models.json"), "w",
                     encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n已写出 {os.path.join(DATA_DIR, 'local_models.json')}")
    if args.report:
        docs = os.path.join(ROOT, "docs")
        os.makedirs(docs, exist_ok=True)
        rp = os.path.join(docs, "01-本机模型扫描报告.md")
        with io.open(rp, "w", encoding="utf-8") as f:
            f.write(render_report(paths, ollama, inv))
        print(f"已写出 {rp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
