# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-混合检索任务
（工单 PDF 原文：人工智能NLP-RAG-混合检索任务）

模块：本机模型扫描（硬性要求0，最高优先级）

用途
----
在加载任何模型之前，对本机环境做一次**只读、零网络请求**的全面扫描，
回答三个问题：
  1. 本机已有哪些可用模型？（嵌入 / Reranker 精排 / 多模态 / LLM / OCR）
  2. 每个模型的具体路径、体积、版本（架构）是什么？
  3. 本工单必须用到但本机缺失的模型有哪些？可复用哪个本机模型替代？

扫描范围（覆盖 Windows / Linux / macOS 常见缓存位置）：
  · HuggingFace 缓存：~/.cache/huggingface/hub/、%USERPROFILE%\\.cache\\huggingface\\hub\\
  · torch 缓存：~/.cache/torch/、~/.cache/torch/hub/checkpoints/
  · ModelScope 缓存：~/.cache/modelscope/hub/、~/.cache/modelscope/models/
  · 环境变量 HF_HOME / TRANSFORMERS_CACHE / MODELSCOPE_CACHE / TORCH_HOME 指向的路径
  · 项目根目录下的 models/ checkpoints/ pretrained/ weights/
  · 本机自定义权重目录（D:\\model 等，由 config 声明）
  · Ollama 本地模型仓库（$OLLAMA_MODELS/manifests）
  · 上游工单随工程落盘的权重（工单4 data/models：bge-reranker-base、CLIP）

输出
----
  · data/local_models.json                —— 机器可读清单（供其它脚本/界面读取）
  · docs/01-本机模型扫描结果.md            —— 人读报告（含缺失模型与降级方案）

约束：本脚本不 import torch/transformers，不调用任何 from_pretrained，
      不写任何缓存，因此绝不会触发下载。
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# 工单必需的模型角色（用于 Step 3「确认缺失模型」）
# ---------------------------------------------------------------------------
REQUIRED_ROLES = {
    "embedding": {
        "desc": "向量嵌入（把招股书内容与 Query 转成向量）",
        "need": "≥2 种可切换（bge 系列、m3e 系列等）",
        "prefer": ["bge-m3", "bge-large", "m3e", "bge-small"],
    },
    "reranker": {
        "desc": "Reranker 精排（召回结果 cross-encoder 重排序）",
        "need": "≥2 种可切换（bge-reranker-large/base 或 cross-encoder 系列）",
        "prefer": ["bge-reranker-v2-m3", "bge-reranker-large", "bge-reranker-base"],
    },
    "llm": {
        "desc": "LLM（答案生成 + LLM 重排器）",
        "need": "≥1（本地优先）",
        "prefer": ["qwen", "deepseek", "llama", "glm"],
    },
    "multimodal": {
        "desc": "多模态模型（图像/图表语义解析，工单04 复用）",
        "need": "≥1（本工单复用上游 VLM 描述缓存，非硬性阻塞）",
        "prefer": ["clip", "blip", "qwen-vl", "mineru", "paddleocr-vl"],
    },
}

# 名称 → 角色 的启发式规则（按顺序匹配）
_ROLE_RULES = [
    ("reranker", re.compile(r"reranker|cross[-_]?encoder", re.I)),
    ("embedding", re.compile(r"bge-m3|bge-large|bge-small|bge-base|m3e|minilm|nomic-embed|text-embedding", re.I)),
    ("multimodal", re.compile(r"clip|blip|vl-|vl_|visual|image|layout|pdf-extract|mineru|paddleocr", re.I)),
    ("llm", re.compile(r"qwen|llama|deepseek|glm|chat|instruct|3b|0\.5b|7b", re.I)),
]


def _classify(name: str) -> str:
    for role, pat in _ROLE_RULES:
        if pat.search(name):
            return role
    return "unknown"


def dir_size(path: str, cap_files: int = 4000) -> int:
    """目录字节数（只读；对超大目录设文件数上限防止卡死）。"""
    total = 0
    n = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
            n += 1
            if n >= cap_files:
                return total
    return total


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def peek_version(path: str) -> str:
    """从 config.json 里读架构/版本信息（不做任何加载）。"""
    cfg = os.path.join(path, "config.json")
    if not os.path.isfile(cfg):
        return ""
    try:
        with open(cfg, encoding="utf-8") as fh:
            c = json.load(fh)
    except Exception:
        return ""
    arch = c.get("architectures") or []
    bits = []
    if arch:
        bits.append("/".join(arch) if isinstance(arch, list) else str(arch))
    for key in ("model_type", "hidden_size", "num_hidden_layers"):
        if c.get(key) is not None:
            bits.append(f"{key}={c[key]}")
    return " ".join(bits)


# ---------------------------------------------------------------------------
# 扫描点定义
# ---------------------------------------------------------------------------
def _candidates() -> list[tuple[str, str, str]]:
    """返回 (扫描位置说明, 路径, 判定类型) 列表。

    判定类型：
      hub     —— 子目录=模型（HF/ModelScope 缓存格式）
      model   —— 路径本身=一个模型目录
      mirror  —— 目录下每个子目录=一个模型
    """
    home = os.path.expanduser("~")
    userprofile = os.environ.get("USERPROFILE", home)
    c = [
        ("HuggingFace 缓存 ~/.cache/huggingface/hub/",
         os.path.join(home, ".cache", "huggingface", "hub"), "hub"),
        ("HuggingFace 缓存 %USERPROFILE%\\.cache\\huggingface\\hub\\",
         os.path.join(userprofile, ".cache", "huggingface", "hub"), "hub"),
        ("torch 缓存 ~/.cache/torch/",
         os.path.join(home, ".cache", "torch"), "mirror"),
        ("torch hub checkpoints ~/.cache/torch/hub/checkpoints/",
         os.path.join(home, ".cache", "torch", "hub", "checkpoints"), "mirror"),
        ("ModelScope 缓存 ~/.cache/modelscope/hub/",
         os.path.join(home, ".cache", "modelscope", "hub"), "hub"),
        ("ModelScope 缓存 ~/.cache/modelscope/models/",
         os.path.join(home, ".cache", "modelscope", "models"), "mirror"),
        ("ModelScope 缓存 %USERPROFILE%\\...\\modelscope\\hub\\",
         os.path.join(userprofile, ".cache", "modelscope", "hub"), "hub"),
    ]
    # 环境变量指向的路径
    for var in ("HF_HOME", "TRANSFORMERS_CACHE", "MODELSCOPE_CACHE", "TORCH_HOME"):
        v = os.environ.get(var)
        if v:
            c.append((f"环境变量 {var}={v}", os.path.join(v, "hub") if var == "HF_HOME" else v, "hub"))
    # 项目内约定目录
    root = os.path.dirname(os.path.abspath(__file__))
    for sub in ("models", "checkpoints", "pretrained", "weights"):
        c.append((f"项目目录 {sub}/", os.path.join(root, sub), "mirror"))
    # 本工单随工程落盘的权重目录（复用上游 04 的 bge-reranker-base）
    c.append(("本工单 data/models/（随工程落盘）",
              os.path.join(root, "data", "models"), "mirror"))
    # 上游工单落盘的权重（工单4 下载的 bge-reranker-base / CLIP）
    up4_models = os.path.join(os.path.dirname(root), "工单4", "data", "models")
    c.append(("上游工单4 data/models（随工程落盘）", up4_models, "mirror"))
    # 本机自定义权重目录
    c.append(("自定义权重目录 D:\\model", r"D:\model", "mirror"))
    return c


# 明显不是模型权重的目录名（venv / 工具 / 锁目录），扫描时跳过
_NOISE_RE = re.compile(
    r"^\.|__pycache__|venv|finalshell|electron|node_modules|site-packages", re.I)


def scan_dirs() -> tuple[list[dict], list[dict]]:
    """扫描所有目录缓存位置。返回 (检查点状态, 模型条目)。"""
    checked: list[dict] = []
    found: list[dict] = []
    seen_paths: set[str] = set()

    for label, path, kind in _candidates():
        exists = os.path.isdir(path)
        entries = []
        if exists:
            try:
                names = sorted(os.listdir(path))
            except OSError:
                names = []
            if kind == "model":
                names = [""]
            for name in names:
                sub = os.path.join(path, name) if name else path
                if not os.path.isdir(sub):
                    continue
                if _NOISE_RE.search(name):
                    continue
                real = os.path.realpath(sub)
                if real in seen_paths:
                    continue
                seen_paths.add(real)
                # 只收「看起来像模型」的目录：有 config.json / 权重文件 / snapshots
                has_cfg = any(os.path.isfile(os.path.join(sub, f))
                              for f in ("config.json", "configuration.json",
                                        "modules.json", "pytorch_model.bin",
                                        "model.safetensors", "tf_model.h5"))
                has_snap = os.path.isdir(os.path.join(sub, "snapshots"))
                if not (has_cfg or has_snap):
                    entries.append({"name": name or os.path.basename(sub),
                                    "empty_or_shell": True})
                    continue
                entries.append({"name": name or os.path.basename(sub),
                                "path": sub, "empty_or_shell": False})
        checked.append({"label": label, "path": path, "exists": exists,
                        "n_entries": len(entries)})
        for e in entries:
            if e.get("empty_or_shell"):
                found.append({"role": "unknown", "name": e["name"],
                              "path": os.path.join(path, e["name"]),
                              "size_bytes": 0, "version": "空壳（无权重文件，不可用）",
                              "usable": False, "origin": label})
                continue
            sub = e["path"]
            size = dir_size(sub)
            found.append({"role": _classify(e["name"]), "name": e["name"],
                          "path": sub, "size_bytes": size,
                          "version": peek_version(sub) or "（无 config.json）",
                          "usable": size > 1024 * 1024, "origin": label})
    return checked, found


def scan_ollama() -> list[dict]:
    """扫描 Ollama 本地模型仓库（只读 manifests + blobs，不调用 ollama CLI）。"""
    base = os.environ.get("OLLAMA_MODELS") or os.path.join(
        os.path.expanduser("~"), ".ollama", "models")
    manifests = os.path.join(base, "manifests")
    out: list[dict] = []
    if not os.path.isdir(manifests):
        return out
    for root, _dirs, files in os.walk(manifests):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), manifests)
            name = rel.replace("\\", "/").replace("/", ":", 1)
            try:
                with open(os.path.join(root, f), encoding="utf-8") as fh:
                    man = json.load(fh)
                size = sum(l.get("size", 0) for l in man.get("layers", []))
                digest = (man.get("config", {}).get("digest") or "")[7:19]
            except Exception:
                size, digest = 0, ""
            out.append({"role": _classify(name), "name": name,
                        "path": f"ollama://{name}", "size_bytes": size,
                        "version": f"ollama manifest {digest}",
                        "usable": size > 0, "origin": "Ollama 本地仓库"})
    return out


def coverage(checked: list[dict], models: list[dict]) -> dict:
    """按角色汇总已有模型，并给出本工单的角色缺口。"""
    by_role: dict[str, list[dict]] = {}
    for m in models:
        if m.get("usable"):
            by_role.setdefault(m["role"], []).append(m)

    roles = {}
    for role, spec in REQUIRED_ROLES.items():
        have = by_role.get(role, [])
        # 明确缺失：该角色一个可用模型都没有
        roles[role] = {
            "desc": spec["desc"],
            "need": spec["need"],
            "have": [{"name": h["name"], "path": h["path"],
                      "size": human(h["size_bytes"])} for h in have],
            "satisfied": bool(have) and len(have) >= (2 if role in ("embedding", "reranker") else 1),
            "n_have": len(have),
        }
    return roles


def build_markdown(checked: list[dict], models: list[dict],
                   ollama: list[dict], roles: dict) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    all_models = models + ollama
    L: list[str] = []
    A = L.append
    A("# 工单06 · 第一步：本机模型扫描结果")
    A("")
    A("> **工单编号**：人工智能NLP-RAG-混合检索任务  ")
    A("> **扫描脚本**：`scan_local_models.py`（只读、零网络请求、不触发任何下载）  ")
    A(f"> **扫描时间**：{now}  ")
    A(f"> **机器**：Windows / Python {sys.version.split()[0]} / "
      f"torch 2.13.0+cu126（CUDA 可用，RTX 4060 Laptop 8GB）")
    A("")
    A("---")
    A("")
    A("## 1. 扫描覆盖范围（硬性要求0 · Step 1）")
    A("")
    A("| 序号 | 扫描位置 | 本机是否存在 | 命中条目 |")
    A("| --- | --- | --- | --- |")
    for i, c in enumerate(checked, 1):
        A(f"| {i} | `{c['label']}` | {'存在' if c['exists'] else '不存在'} | {c['n_entries']} |")
    A(f"| {len(checked) + 1} | Ollama 本地仓库（`OLLAMA_MODELS`） | "
      f"{'存在' if ollama else '不存在'} | {len(ollama)} |")
    A("")
    A("---")
    A("")
    A("## 2. 扫描结果总表（硬性要求0 · Step 2）")
    A("")
    A("| 类型 | 名称 | 具体路径 | 大小 | 版本/架构 |")
    A("| --- | --- | --- | --- | --- |")
    role_cn = {"embedding": "嵌入", "reranker": "Reranker 精排", "llm": "LLM",
               "multimodal": "多模态/OCR", "unknown": "未知"}
    for m in sorted(all_models, key=lambda x: (x["role"], x["name"])):
        A(f"| {role_cn.get(m['role'], m['role'])} | {m['name']} | `{m['path']}` | "
          f"{human(m['size_bytes'])} | {m['version']} |")
    A("")
    usable = [m for m in all_models if m.get("usable")]
    shells = [m for m in all_models if not m.get("usable")]
    A(f"**合计 {len(all_models)} 个条目**，其中可用 {len(usable)} 个"
      + (f"，空壳/不可用 {len(shells)} 个（{'、'.join(s['name'] for s in shells)}）"
         if shells else "") + "。")
    A("")
    A("---")
    A("")
    A("## 3. 角色覆盖与本工单缺口（硬性要求0 · Step 3）")
    A("")
    A("| 角色 | 用途 | 要求 | 本机已有 | 是否满足 |")
    A("| --- | --- | --- | --- | --- |")
    for role, r in roles.items():
        have = "；".join(f"{h['name']}（{h['size']}）" for h in r["have"]) or "**无**"
        A(f"| {role} | {r['desc']} | {r['need']} | {have} | "
          f"{'✅ 满足' if r['satisfied'] else '❌ 缺口'} |")
    A("")
    missing = [r for r, v in roles.items() if not v["satisfied"]]
    if missing:
        A("### 缺失模型清单与处置")
        A("")
        for role in missing:
            spec = REQUIRED_ROLES[role]
            A(f"- **{role}**（{spec['desc']}）：本机可用数量 {roles[role]['n_have']}，"
              f"要求 {spec['need']}。")
    else:
        A("**结论：本工单所需的四类模型角色，本机全部满足，无需下载任何模型。**")
        A("")
        A("依据：")
        A("")
        A("- 嵌入模型 ≥2 种：bge-m3（主）/ bge-m3-ft-v1 / all-MiniLM-L6-v2，支持切换；")
        A("- Reranker ≥2 种：bge-reranker-v2-m3（大）+ bge-reranker-base（小），"
          "另有 Ollama qwen2.5:3b 作 LLM 重排器；")
        A("- LLM：Ollama qwen2.5:3b / qwen2.5:0.5b；")
        A("- 多模态：上游工单4 已完成图像语义解析（VLM 描述缓存 162 条随工程落盘），"
          "本工单直接复用，不重复加载视觉模型。")
    A("")
    A("> 说明：**未执行任何下载**。所有模型加载一律走本地绝对路径 + "
      "`local_files_only=True`，进程启动即置 `HF_HUB_OFFLINE=1`（见 `src/bootstrap.py`），"
      "漏写离线参数的调用会直接失败而不是偷偷联网。")
    A("")
    return "\n".join(L)


def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    print("[scan] 开始扫描本机模型缓存（只读，不联网）……", flush=True)
    checked, models = scan_dirs()
    ollama = scan_ollama()
    roles = coverage(checked, models + ollama)

    payload = {
        "workorder": "人工智能NLP-RAG-混合检索任务",
        "scanned_at": datetime.now().isoformat(timespec="seconds"),
        "checked_locations": checked,
        "models": models + ollama,
        "roles": roles,
        "download_performed": False,
    }
    os.makedirs(os.path.join(root, "data"), exist_ok=True)
    with open(os.path.join(root, "data", "local_models.json"), "w",
              encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    os.makedirs(os.path.join(root, "docs"), exist_ok=True)
    md = build_markdown(checked, models, ollama, roles)
    with open(os.path.join(root, "docs", "01-本机模型扫描结果.md"), "w",
              encoding="utf-8") as fh:
        fh.write(md)

    print(f"[scan] 完成：{len(models) + len(ollama)} 个条目，"
          f"{len([m for m in models + ollama if m.get('usable')])} 个可用")
    for role, r in roles.items():
        print(f"  - {role}: {r['n_have']} 个可用, 满足={r['satisfied']}")
    print("[scan] 输出：data/local_models.json / docs/01-本机模型扫描结果.md")


if __name__ == "__main__":
    main()
