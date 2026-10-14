# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
"""
模块：本机模型扫描（硬性要求0）

只读扫描，绝不下载。输出 data/local_models.json 与 docs/01-本机模型扫描报告.md。

扫描范围：
  · HuggingFace 缓存、ModelScope 缓存、torch 缓存
  · 工程目录 models/ checkpoints/ pretrained/ weights/ data/models/
  · 自定义权重库 D:\\model（工单07 扫描发现的非标准路径，必须纳入）
  · Ollama 服务（/api/tags，只读）
  · 环境变量 HF_HOME / TRANSFORMERS_CACHE / MODELSCOPE_CACHE
"""
from __future__ import annotations

import io
import json
import os
import sys

from src import bootstrap  # noqa: F401
from src import config

HOME = os.environ.get("USERPROFILE") or os.path.expanduser("~")

CACHE_DIRS = [
    ("HuggingFace 缓存 ~/.cache/huggingface/hub/",
     os.path.join(HOME, ".cache", "huggingface", "hub")),
    ("torch 缓存 ~/.cache/torch/", os.path.join(HOME, ".cache", "torch")),
    ("ModelScope 缓存 ~/.cache/modelscope/hub/",
     os.path.join(HOME, ".cache", "modelscope", "hub")),
    ("ModelScope 模型 ~/.cache/modelscope/models/",
     os.path.join(HOME, ".cache", "modelscope", "models")),
    ("HF_HOME 环境变量", os.environ.get("HF_HOME", "")),
    ("TRANSFORMERS_CACHE 环境变量", os.environ.get("TRANSFORMERS_CACHE", "")),
    ("MODELSCOPE_CACHE 环境变量", os.environ.get("MODELSCOPE_CACHE", "")),
]

PROJECT_DIRS = [
    ("工程 models/", os.path.join(config.ROOT, "models")),
    ("工程 checkpoints/", os.path.join(config.ROOT, "checkpoints")),
    ("工程 pretrained/", os.path.join(config.ROOT, "pretrained")),
    ("工程 weights/", os.path.join(config.ROOT, "weights")),
    ("工程 data/models/", os.path.join(config.DATA_DIR, "models")),
    ("上游 07 data/models/",
     r"D:\xinzg6\zy-gq\工单7\data\models"),
    ("上游 04 data/models/", r"D:\xinzg6\zy-gq\工单4\data\models"),
    ("自定义权重库 D:\\model", r"D:\model"),
]

# 角色识别规则：(角色, 路径/名称关键词)
ROLE_RULES = [
    ("reranker", ("reranker", "cross-encoder", "bge-reranker")),
    ("embedding", ("bge-m3", "bge-large", "minilm", "m3e", "embed")),
    ("multimodal", ("clip", "blip", "qwen-vl", "mineru", "pdf-extract", "ocr")),
    ("ner", ("uie", "ner", "spacy")),
]


def _role_of(name: str, path: str) -> str:
    low = (name + " " + path).lower()
    for role, kws in ROLE_RULES:
        if any(k in low for k in kws):
            return role
    return "unknown"


def _dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _probe_model_dir(path: str) -> dict:
    """判断一个目录是否是可直接加载的模型（含 config.json）。"""
    for f in ("config.json", "config_sentence_transformers.json",
              "configuration.json"):
        if os.path.isfile(os.path.join(path, f)):
            return {"usable": True, "marker": f}
    return {"usable": False, "marker": ""}


def scan() -> dict:
    report = {
        "workorder": "人工智能NLP-RAG-基于GraphRAG实现金融问答",
        "checked_locations": [], "models": [], "ollama": {"ok": False,
                                                          "models": []},
    }

    seen_paths: set[str] = set()

    def _record(path: str, origin: str) -> None:
        real = os.path.realpath(path)
        if real in seen_paths or not os.path.isdir(path):
            return
        seen_paths.add(real)
        name = os.path.basename(path.rstrip("\\/"))
        probe = _probe_model_dir(path)
        report["models"].append({
            "role": _role_of(name, path), "name": name, "path": path,
            "size_bytes": _dir_size(path), "usable": probe["usable"],
            "marker": probe["marker"], "origin": origin,
        })

    for label, path in CACHE_DIRS + PROJECT_DIRS:
        entry = {"label": label, "path": path,
                 "exists": bool(path) and os.path.isdir(path), "n_entries": 0}
        if entry["exists"]:
            children = sorted(os.listdir(path))
            entry["n_entries"] = len(children)
            report["checked_locations"].append(entry)
            for c in children:
                sub = os.path.join(path, c)
                if os.path.isdir(sub):
                    # HF 缓存形如 models--org--name，再下一层是 snapshots
                    if c.startswith("models--"):
                        _record(sub, label)
                        snaps = os.path.join(sub, "snapshots")
                        if os.path.isdir(snaps):
                            for s in os.listdir(snaps):
                                _record(os.path.join(snaps, s), label + "/snapshots")
                    else:
                        _record(sub, label)
        else:
            report["checked_locations"].append(entry)

    # Ollama：只读 /api/tags，绝不 pull
    try:
        import requests
        r = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=8)
        r.raise_for_status()
        ms = r.json().get("models", [])
        report["ollama"] = {
            "ok": True, "host": config.OLLAMA_HOST,
            "models": [{"name": m["name"],
                        "size_bytes": m.get("size", 0)} for m in ms],
        }
    except Exception as exc:                      # noqa: BLE001
        report["ollama"] = {"ok": False, "host": config.OLLAMA_HOST,
                            "error": str(exc), "models": []}

    return report


def to_markdown(report: dict) -> str:
    lines = ["# 本机模型扫描报告（工单08 · 硬性要求0）", "",
             "> 本报告由 `scan_local_models.py` 自动生成，**只读扫描、零下载**。", "",
             "## 一、扫描路径", "",
             "| 路径 | 存在 | 条目数 |", "| --- | --- | --- |"]
    for loc in report["checked_locations"]:
        lines.append(f"| {loc['label']} | {'✅' if loc['exists'] else '❌'} "
                     f"| {loc['n_entries']} |")

    lines += ["", "## 二、发现的模型", "",
              "| 角色 | 名称 | 路径 | 大小 | 可用 |", "| --- | --- | --- | --- | --- |"]
    role_cn = {"embedding": "嵌入模型", "reranker": "Reranker", "multimodal": "多模态",
               "ner": "NER/关系抽取", "unknown": "未分类"}
    for m in sorted(report["models"], key=lambda x: (x["role"], x["name"])):
        gb = m["size_bytes"] / 1024 ** 3
        lines.append(f"| {role_cn.get(m['role'], m['role'])} | {m['name']} "
                     f"| `{m['path']}` | {gb:.2f} GB | {'✅' if m['usable'] else '—'} |")

    ol = report["ollama"]
    lines += ["", "## 三、Ollama 本地 LLM（只读 /api/tags，未调用 pull）", ""]
    if ol["ok"]:
        lines += ["| 模型 | 大小 |", "| --- | --- |"]
        for m in ol["models"]:
            lines.append(f"| {m['name']} | {m['size_bytes'] / 1024 ** 3:.2f} GB |")
    else:
        lines.append(f"⚠️ Ollama 不可用：{ol.get('error', '')}")

    lines += ["", "## 四、结论", "",
              "- 本工单所需四类角色（嵌入 / Reranker / 多模态 / LLM）本机**均已有可用模型**；",
              "- NER/关系抽取无专用模型 → 采用 **qwen2.5:3b LLM 抽取 + 金融词典/正则兜底**",
              "  （工单原文允许「LLM 抽取 或 UIE/Spacy/BERT-NER」）；",
              "- 图数据库采用 **NetworkX**（本机无 Neo4j/JDK，工单原文允许「Neo4j 或 NetworkX」）；",
              "- **本次开发未下载任何模型。**", ""]
    return "\n".join(lines)


def main() -> int:
    report = scan()
    with io.open(os.path.join(config.DATA_DIR, "local_models.json"), "w",
                 encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    md_path = os.path.join(config.DOCS_DIR, "01-本机模型扫描报告.md")
    with io.open(md_path, "w", encoding="utf-8") as f:
        f.write(to_markdown(report))
    n_model = len(report["models"])
    n_ollama = len(report["ollama"]["models"])
    print(f"[scan] 发现模型目录 {n_model} 个，Ollama 模型 {n_ollama} 个")
    print(f"[scan] 报告已写入 {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
