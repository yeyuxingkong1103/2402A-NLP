# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
第一步产物：扫描本机已下载模型清单

硬约束：禁止下载任何模型。本脚本**只读**扫描本机，不做任何网络请求。

扫描位置：
  1. HuggingFace 缓存      ~/.cache/huggingface/hub/
  2. ModelScope 缓存       ~/.cache/modelscope/hub/
  3. Ollama 模型           ollama list + $OLLAMA_MODELS/manifests
  4. 常见自定义目录        ./models/、D:/model、/models、/data/models ...
  5. MinerU 模型目录       ~/.cache/mineru/
  6. PaddleOCR 模型目录    ~/.paddleocr/、~/.paddlex/official_models/

输出：模型类型(embedding/llm/ocr/reranker/未知) | 名称 | 本地路径 | 文件大小

用法：
    python scan_local_models.py            # 打印清单
    python scan_local_models.py --json     # 额外输出 JSON，便于程序消费
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

HOME = os.path.expanduser("~")

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def dir_size(path: str) -> int:
    """递归统计目录/文件字节数；忽略无权限项。"""
    if os.path.isfile(path):
        try:
            return os.path.getsize(path)
        except OSError:
            return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def human(num: int) -> str:
    """字节数转可读字符串。"""
    val = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if val < 1024 or unit == "TB":
            return f"{val:.1f}{unit}"
        val /= 1024
    return f"{val:.1f}TB"


# ---------------------------------------------------------------------------
# 模型类型判定规则：按目录/模型名关键词推断
# ---------------------------------------------------------------------------
RULES = [
    # (类型, 命中关键词, 说明)
    ("reranker", ("reranker", "rerank", "bge-reranker"), "重排序模型"),
    ("ocr",      ("paddleocr", "doclayout", "ocr", "mineru", "pdf-extract-kit",
                  "layout", "table-rec", "formula-rec"), "OCR / 版面分析"),
    ("embedding", ("bge-m3", "bge-large", "bge-small", "bge-base", "m3e",
                   "text2vec", "minilm", "gte-", "embed", "stella", "conan"), "向量化模型"),
    ("llm",      ("qwen", "chatglm", "glm", "baichuan", "llama", "yi-",
                  "mistral", "deepseek", "internlm", "phi", "gemma"), "大语言模型"),
]


def classify(name: str, extra_hint: str = "") -> tuple[str, str]:
    """根据名称推断模型类型，返回 (类型, 说明)。"""
    blob = f"{name} {extra_hint}".lower()
    for kind, keys, note in RULES:
        if any(k in blob for k in keys):
            return kind, note
    return "未知", "未能按名称判定"


def is_probably_model_dir(path: str) -> bool:
    """判断目录内是否含模型权重文件。"""
    markers = {
        "config.json", "model.safetensors", "pytorch_model.bin",
        "inference.pdiparams", "inference.json", "inference.yml",
        "tokenizer.json", "sentencepiece.bpe.model", "modules.json",
        "model.onnx", "GGUF",
    }
    try:
        entries = set(os.listdir(path))
    except OSError:
        return False
    if entries & markers:
        return True
    # 允许再往下一层（例如 official_models/PaddleOCR-VL）
    for e in list(entries)[:40]:
        sub = os.path.join(path, e)
        if os.path.isdir(sub):
            try:
                if set(os.listdir(sub)) & markers:
                    return True
            except OSError:
                pass
    return False


# ---------------------------------------------------------------------------
# 各来源扫描
# ---------------------------------------------------------------------------
found: list[dict] = []


def add(kind: str, name: str, path: str, size: int, note: str, origin: str) -> None:
    found.append({
        "type": kind, "name": name, "path": path,
        "size": size, "size_human": human(size), "note": note, "origin": origin,
    })


def scan_hf_cache() -> None:
    """扫描 HuggingFace hub 缓存：解析 models--org--name 结构。"""
    hub = os.path.join(HOME, ".cache", "huggingface", "hub")
    if not os.path.isdir(hub):
        return
    for entry in sorted(os.listdir(hub)):
        if not entry.startswith("models--"):
            continue
        full = os.path.join(hub, entry)
        model_id = entry[len("models--"):].replace("--", "/")
        size = dir_size(full)
        # 只有 refs/snapshots 里的真实权重才算；空壳（仅 refs）要标记出来
        size = dir_size(full)
        kind, note = classify(model_id)
        if size < 1024 * 1024:  # <1MB 视为空壳/未真正下载
            note += "（疑似空壳，未真正下载权重）"
        add(kind, model_id, full, size, note, "HuggingFace缓存")


def scan_modelscope_cache() -> None:
    """扫描 ModelScope 缓存。"""
    for hub in (os.path.join(HOME, ".cache", "modelscope", "hub"),
                os.path.join(HOME, ".modelscope", "hub")):
        if not os.path.isdir(hub):
            continue
        for entry in sorted(os.listdir(hub)):
            full = os.path.join(hub, entry)
            if not os.path.isdir(full):
                continue
            model_id = entry.replace("--", "/")
            kind, note = classify(model_id)
            add(kind, model_id, full, dir_size(full), note, "ModelScope缓存")


def scan_ollama() -> None:
    """扫描 Ollama 已 pull 的模型：优先读 manifests，其次调 ollama list。"""
    seen: set[str] = set()

    models_root = os.environ.get("OLLAMA_MODELS", os.path.join(HOME, ".ollama", "models"))
    manifests = os.path.join(models_root, "manifests")
    if os.path.isdir(manifests):
        # 结构: manifests/registry.ollama.ai/<namespace>/<model>/<tag>
        for reg in os.listdir(manifests):
            reg_path = os.path.join(manifests, reg)
            if not os.path.isdir(reg_path):
                continue
            for ns in os.listdir(reg_path):
                ns_path = os.path.join(reg_path, ns)
                if not os.path.isdir(ns_path):
                    continue
                for model in os.listdir(ns_path):
                    m_path = os.path.join(ns_path, model)
                    if not os.path.isdir(m_path):
                        continue
                    for tag in os.listdir(m_path):
                        name = f"{ns}/{model}:{tag}" if ns != "library" else f"{model}:{tag}"
                        if name in seen:
                            continue
                        seen.add(name)
                        # manifest JSON 内 layer digest -> blobs 实际大小
                        size = 0
                        try:
                            with open(os.path.join(m_path, tag), encoding="utf-8") as fh:
                                man = json.load(fh)
                            for layer in man.get("layers", []):
                                digest = layer.get("digest", "")
                                if ":" in digest:
                                    blob = os.path.join(models_root, "blobs",
                                                       digest.replace(":", "-"))
                                    if os.path.isfile(blob):
                                        size += os.path.getsize(blob)
                        except Exception:
                            pass
                        kind, note = classify(name)
                        extra = ""
                        if kind == "embedding":
                            extra = "（Ollama 嵌入模型）"
                        elif kind == "llm":
                            extra = "（Ollama 生成模型，本地已 pull）"
                        add(kind, name, os.path.join(m_path, tag), size,
                            note + extra, "Ollama")

    if seen:
        return
    # 回退：调用 ollama list（模型已被 pull，不触发下载）
    try:
        out = subprocess.run(["ollama", "list"], capture_output=True,
                             text=True, timeout=20, encoding="utf-8", errors="ignore")
        for line in (out.stdout or "").splitlines()[1:]:
            parts = line.split()
            if not parts:
                continue
            name = parts[0]
            kind, note = classify(name)
            add(kind, name, f"(ollama store: {models_root})", 0,
                note + "（Ollama 已 pull）", "Ollama")
    except Exception:
        pass


def scan_custom_dirs() -> None:
    """扫描常见自定义模型目录。"""
    candidates = [
        os.path.join(os.getcwd(), "models"),
        r"D:\model", r"D:\models", r"C:\models", r"D:\data\models",
        r"D:\ai\models", r"E:\models", r"D:\LLM", r"D:\models\hub",
    ]
    for base in candidates:
        if not os.path.isdir(base):
            continue
        for entry in sorted(os.listdir(base)):
            full = os.path.join(base, entry)
            if not os.path.isdir(full):
                continue
            if not is_probably_model_dir(full):
                continue
            kind, note = classify(entry)
            add(kind, entry, full, dir_size(full), note, f"自定义目录({base})")


def scan_mineru() -> None:
    """扫描 MinerU 模型目录。"""
    for p in (os.path.join(HOME, ".cache", "mineru"),
              os.path.join(HOME, "mineru", "models")):
        if os.path.isdir(p):
            for e in sorted(os.listdir(p)):
                full = os.path.join(p, e)
                if os.path.isdir(full):
                    add("ocr", e, full, dir_size(full), "MinerU 配套模型", "MinerU")


def scan_paddleocr() -> None:
    """扫描 PaddleOCR / PaddleX 模型目录。"""
    roots = [
        os.path.join(HOME, ".paddleocr"),
        os.path.join(HOME, ".paddlex", "official_models"),
        r"D:\model\Paddle-OCR\official_models",
        r"D:\model\Paddle-OCR",
    ]
    for root in roots:
        if not os.path.isdir(root):
            continue
        for e in sorted(os.listdir(root)):
            full = os.path.join(root, e)
            if not os.path.isdir(full):
                continue
            if full.rstrip("\\/").lower().endswith("official_models"):
                continue
            kind, note = classify(e, "ocr")
            if "ocr" not in (kind, "ocr"):
                kind = "ocr" if "ocr" in e.lower() or "layout" in e.lower() else kind
            # venv/wheels 等非模型目录跳过
            if e.lower() in {"venv", "wheels", "temp", "locks", "func_ret"}:
                continue
            if not is_probably_model_dir(full):
                continue
            add("ocr", e, full, dir_size(full), "PaddleOCR / PaddleX 模型", "PaddleOCR")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="扫描本机已下载模型（只读、不联网）")
    parser.add_argument("--json", action="store_true", help="同时输出 JSON")
    args = parser.parse_args()

    for fn in (scan_hf_cache, scan_modelscope_cache, scan_ollama,
               scan_custom_dirs, scan_mineru, scan_paddleocr):
        try:
            fn()
        except Exception as exc:  # 单个来源失败不影响整体
            print(f"[警告] {fn.__name__} 扫描失败: {exc}", file=sys.stderr)

    # 去重（同一路径只保留一次）
    uniq: dict[str, dict] = {}
    for item in found:
        uniq.setdefault(item["path"], item)
    items = sorted(uniq.values(), key=lambda x: (x["type"], -x["size"]))

    order = {"embedding": 0, "llm": 1, "reranker": 2, "ocr": 3, "未知": 4}
    items.sort(key=lambda x: (order.get(x["type"], 9), -x["size"]))

    print("=" * 100)
    print("本机已下载模型清单（只读扫描，未做任何下载）")
    print("=" * 100)
    print(f"{'类型':<10} {'名称':<44} {'大小':>10}  说明")
    print("-" * 100)
    for it in items:
        note = it["note"]
        if it["size"] == 0:
            note += " [大小未知]"
        print(f"{it['type']:<10} {it['name'][:44]:<44} {it['size_human']:>10}  {note}")
    print("-" * 100)
    print(f"合计 {len(items)} 个条目")
    for kind in ("embedding", "llm", "reranker", "ocr", "未知"):
        names = [i["name"] for i in items if i["type"] == kind]
        if names:
            print(f"  [{kind}] {len(names)} 个: {', '.join(names[:8])}"
                  + (" ..." if len(names) > 8 else ""))
    print("\n完整路径：")
    for it in items:
        print(f"  - [{it['type']}] {it['path']}")

    if args.json:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "data", "local_models.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(items, fh, ensure_ascii=False, indent=2)
        print(f"\nJSON 已写入: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
