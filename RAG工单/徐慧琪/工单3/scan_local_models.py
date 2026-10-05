# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
模块：第一步产物 —— 本机已下载模型扫描（只读，全程不联网）

硬约束：禁止下载任何模型。本脚本仅扫描本地文件系统（并读取 ollama 本地清单），
        不发起任何网络请求，不调用 from_pretrained / snapshot_download。

扫描位置：
  1. HuggingFace 缓存        ~/.cache/huggingface/hub/
  2. ModelScope 缓存         ~/.cache/modelscope/models/（本机 MinerU 权重在此）
  3. Ollama                  $OLLAMA_MODELS/manifests（等价于 ollama list）
  4. 自定义目录              ./models/、D:/model、/models/、/data/models/ 等
  5. MinerU 缓存             ~/.cache/mineru/、~/mineru/models/
  6. PaddleOCR / PaddleX     ~/.paddleocr/、~/.paddlex/official_models/

输出：模型类型（embedding / llm / ocr / reranker / 未知）| 名称 | 本地路径 | 大小

用法：
    python scan_local_models.py            # 打印清单
    python scan_local_models.py --json     # 额外写出 data/local_models.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HOME = os.path.expanduser("~")
TASK_ID = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"

# 这些目录名不是模型本体，扫描时跳过
SKIP_NAMES = {
    "venv", "wheels", "temp", "locks", "func_ret", ".cache", ".git",
    "assets", "imgs", "onnx", ".eval_results", ".msc", ".mv", "logs",
    "finalshell", "node_modules", "templates", "sample_data",
}


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
# 模型类型判定规则（按名称关键词推断，顺序即优先级）
# ---------------------------------------------------------------------------
RULES = [
    ("reranker", ("reranker", "rerank", "cross-encoder"), "重排序模型（cross-encoder）"),
    ("ocr", ("paddleocr", "pp-doclayout", "doclayout", "paddlex", "mineru",
             "pdf-extract-kit", "unimernet", "structeqtable", "tablemaster",
             "layout", "ocr", "formula"), "OCR / 版面分析 / 文档解析"),
    ("embedding", ("bge-m3", "bge-large", "bge-small", "bge-base", "m3e",
                   "text2vec", "minilm", "gte-", "embed", "stella", "conan"),
     "向量化（Embedding）模型"),
    ("llm", ("qwen", "chatglm", "glm", "baichuan", "llama", "yi-", "mistral",
             "deepseek", "internlm", "phi", "gemma"), "大语言模型（LLM）"),
]


def classify(name: str, extra_hint: str = "") -> tuple:
    """根据名称推断模型类型，返回 (类型, 说明)。"""
    blob = f"{name} {extra_hint}".lower()
    for kind, keys, note in RULES:
        if any(k in blob for k in keys):
            return kind, note
    return "未知", "未能按名称判定"


MODEL_MARKERS = {
    "config.json", "model.safetensors", "pytorch_model.bin", "model.onnx",
    "inference.pdiparams", "inference.json", "inference.yml",
    "tokenizer.json", "sentencepiece.bpe.model", "modules.json",
    "vocab.txt", "generation_config.json", "configuration.json",
}


def looks_like_model(path: str) -> bool:
    """判断目录/文件本身是否含模型权重标识。"""
    if os.path.isfile(path):
        low = path.lower()
        return low.endswith((".safetensors", ".gguf", ".pt", ".pth", ".bin", ".onnx"))
    try:
        entries = set(os.listdir(path))
    except OSError:
        return False
    if entries & MODEL_MARKERS:
        return True
    return any(e.lower().endswith(".gguf") for e in entries)


# ---------------------------------------------------------------------------
# 结果收集
# ---------------------------------------------------------------------------
found: list = []


def add(kind: str, name: str, path: str, size: int, note: str, origin: str) -> None:
    found.append({
        "type": kind, "name": name, "path": path,
        "size": size, "size_human": human(size), "note": note, "origin": origin,
    })


# ---------------------------------------------------------------------------
# 各来源扫描
# ---------------------------------------------------------------------------
def scan_hf_cache() -> None:
    """扫描 HuggingFace hub 缓存：models--org--name 结构。"""
    hub = os.path.join(HOME, ".cache", "huggingface", "hub")
    if not os.path.isdir(hub):
        return
    for entry in sorted(os.listdir(hub)):
        if not entry.startswith("models--"):
            continue
        full = os.path.join(hub, entry)
        model_id = entry[len("models--"):].replace("--", "/")
        snapshots = os.path.join(full, "snapshots")
        has_snapshot = os.path.isdir(snapshots) and bool(os.listdir(snapshots))
        if not has_snapshot:
            add("未知", model_id, full, dir_size(full),
                "HF 缓存空壳（仅 refs，无实际权重）", "HuggingFace")
            continue
        kind, note = classify(model_id)
        add(kind, model_id, full, dir_size(full), note, "HuggingFace")


def scan_modelscope_cache() -> None:
    """扫描 ModelScope 缓存：目录名形如 Org--Name。"""
    hubs = [
        os.path.join(HOME, ".cache", "modelscope", "models"),
        os.path.join(HOME, ".cache", "modelscope", "hub"),
    ]
    for hub in hubs:
        if not os.path.isdir(hub):
            continue
        for entry in sorted(os.listdir(hub)):
            full = os.path.join(hub, entry)
            if not os.path.isdir(full) or entry in SKIP_NAMES:
                continue
            model_id = entry.replace("--", "/")
            kind, note = classify(entry)
            add(kind, model_id, full, dir_size(full),
                note + "（ModelScope 缓存）", "ModelScope")


def ollama_roots() -> list:
    """Ollama 模型仓库根目录候选。"""
    roots = []
    env = os.environ.get("OLLAMA_MODELS")
    if env:
        roots.append(env)
    roots += [os.path.join(HOME, ".ollama", "models"), r"D:\Ollama\OLLAMA_MODELS"]
    return [r for r in roots if os.path.isdir(r)]


def scan_ollama() -> None:
    """扫描 Ollama manifest，解析出已 pull 的模型与真实占用。"""
    for root in ollama_roots():
        manifests = os.path.join(root, "manifests")
        if not os.path.isdir(manifests):
            continue
        for dirpath, _dirs, files in os.walk(manifests):
            for fn in files:
                mpath = os.path.join(dirpath, fn)
                rel = os.path.relpath(mpath, manifests).replace("\\", "/")
                parts = rel.split("/")
                if len(parts) < 3:
                    continue
                namespace, model, tag = parts[-3], parts[-2], parts[-1]
                name = (f"{model}:{tag}" if namespace == "library"
                        else f"{namespace}/{model}:{tag}")
                size = 0
                try:
                    with open(mpath, "r", encoding="utf-8") as fh:
                        manifest = json.load(fh)
                    for layer in manifest.get("layers", []):
                        size += int(layer.get("size", 0) or 0)
                except Exception:
                    pass
                kind, note = classify(name)
                add(kind, name, f"ollama://{name} (store: {root})", size,
                    note + "（Ollama 已 pull，离线可用）", "Ollama")


def scan_custom_dirs() -> None:
    """扫描常见自定义模型目录（只取自身即含权重的目录）。"""
    candidates = [
        os.path.join(os.getcwd(), "models"),
        os.path.join(os.getcwd(), "model"),
        r"D:\model", r"D:\models", r"C:\models", r"E:\models",
        r"D:\data\models", r"D:\ai\models", r"D:\LLM", r"D:\models\hub",
    ]
    for base in candidates:
        if not os.path.isdir(base):
            continue
        for entry in sorted(os.listdir(base)):
            full = os.path.join(base, entry)
            if not os.path.isdir(full) or entry in SKIP_NAMES:
                continue
            # PaddleOCR 目录交给 scan_paddleocr 细粒度处理，避免把 venv 算进来
            if entry.lower().startswith("paddle-ocr") or entry.lower() == "paddleocr":
                continue
            if not looks_like_model(full):
                continue
            kind, note = classify(entry)
            add(kind, entry, full, dir_size(full), note, f"自定义目录({base})")


def scan_mineru() -> None:
    """扫描 MinerU 独立模型缓存目录。"""
    for p in (os.path.join(HOME, ".cache", "mineru"),
              os.path.join(HOME, "mineru", "models")):
        if not os.path.isdir(p):
            continue
        for e in sorted(os.listdir(p)):
            full = os.path.join(p, e)
            if not os.path.isdir(full) or e in SKIP_NAMES:
                continue
            add("ocr", e, full, dir_size(full), "MinerU 配套模型", "MinerU")


def scan_paddleocr() -> None:
    """扫描 PaddleOCR / PaddleX 模型目录（含官方便携目录）。"""
    roots = [
        os.path.join(HOME, ".paddleocr"),
        os.path.join(HOME, ".paddlex", "official_models"),
        r"D:\model\Paddle-OCR\official_models",
    ]
    for root in roots:
        if not os.path.isdir(root):
            continue
        for e in sorted(os.listdir(root)):
            full = os.path.join(root, e)
            if not os.path.isdir(full) or e in SKIP_NAMES:
                continue
            if not looks_like_model(full):
                continue
            kind, note = classify(e, "ocr paddleocr")
            if kind != "ocr":
                kind = "ocr"
            add("ocr", e, full, dir_size(full), note + "（PaddleOCR / PaddleX）", "PaddleOCR")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
ORDER = {"embedding": 0, "llm": 1, "reranker": 2, "ocr": 3, "未知": 4}


def main() -> int:
    parser = argparse.ArgumentParser(description="扫描本机已下载模型（只读、不联网）")
    parser.add_argument("--json", action="store_true", help="同时输出 JSON 清单")
    args = parser.parse_args()

    print(f"工单：{TASK_ID}")
    print("正在扫描本机模型（只读，不发起任何网络请求）...\n", file=sys.stderr)

    for fn in (scan_hf_cache, scan_modelscope_cache, scan_ollama,
               scan_custom_dirs, scan_mineru, scan_paddleocr):
        try:
            fn()
        except Exception as exc:  # 单个来源失败不影响整体
            print(f"[警告] {fn.__name__} 扫描失败: {exc}", file=sys.stderr)

    # 去重（同一路径只保留一次），并按类型/大小排序
    uniq = {}
    for item in found:
        uniq.setdefault(item["path"], item)
    items = sorted(uniq.values(), key=lambda x: (ORDER.get(x["type"], 9), -x["size"]))

    width = 108
    print("=" * width)
    print(f"本机已下载模型清单（只读扫描，未做任何下载）   {TASK_ID}")
    print("=" * width)
    print(f"{'类型':<10} {'名称':<42} {'大小':>10}  说明")
    print("-" * width)
    for it in items:
        name = it["name"]
        name = name if len(name) <= 42 else name[:39] + "..."
        note = it["note"] + (" [大小未知]" if it["size"] == 0 else "")
        print(f"{it['type']:<10} {name:<42} {it['size_human']:>10}  {note}")
    print("-" * width)
    print(f"合计 {len(items)} 个条目")
    for kind in ("embedding", "llm", "reranker", "ocr", "未知"):
        names = [i["name"] for i in items if i["type"] == kind]
        if names:
            print(f"  [{kind}] {len(names)} 个: {', '.join(names[:8])}"
                  + (" ..." if len(names) > 8 else ""))

    print("\n完整本地路径：")
    for it in items:
        print(f"  - [{it['type']:<9}] {it['path']}")

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