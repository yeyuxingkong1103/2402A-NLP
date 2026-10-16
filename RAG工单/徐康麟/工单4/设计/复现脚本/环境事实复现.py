# -*- coding: utf-8 -*-
"""工单：人工智能NLP-RAG-图像内容解析及检索优化
用途：环境事实【一键复现】脚本。设计/环境事实.md 中每一条结论都由本脚本实测产出，
      任何成员怀疑结论失效时，跑本脚本即可重新取证，不得凭记忆或推测修改事实文档。

运行（工作目录 = E:\\gao6gongdan\\工单4）：
    pwsh -NoProfile -File run_py.ps1 设计/复现脚本/环境事实复现.py            # 全量（约 40 秒）
    pwsh -NoProfile -File run_py.ps1 设计/复现脚本/环境事实复现.py --skip-scan # 跳过 898 页图像扫描（约 20 秒）
    pwsh -NoProfile -File run_py.ps1 设计/复现脚本/环境事实复现.py --skip-net  # 跳过联网探测

设计约束（工单4 硬性纪律）：
  1. 本文件所有注释均含工单编号「人工智能NLP-RAG-图像内容解析及检索优化」，注释一律中文。
  2. 每个函数入口/出口都记录结构化日志（函数名、输入摘要、输出摘要、耗时、异常堆栈）。
  3. 禁止静默失败：捕获异常必须记日志后显式降级或传播，绝不 pass。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time
import traceback
from typing import Any, Callable

# 工单编号常量：所有日志与注释均以它标记归属，便于全仓库检索归属
GONGDAN = "人工智能NLP-RAG-图像内容解析及检索优化"

# 工单4 仓库根目录（本文件位于 设计/复现脚本/ 下，向上两级即根目录）
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PDF1 = os.path.join(ROOT, "研发", "data", "raw", "招股说明书1.pdf")
PDF2 = os.path.join(ROOT, "研发", "data", "raw", "招股说明书2.pdf")

# 依赖清单：名称 -> 用途。缺失项必须如实报告，不得编造替代能力。
DEPS = [
    ("pymupdf", "PDF 文本/图像/绘图/表格解析"),
    ("numpy", "向量计算"),
    ("jieba", "中文分词（BM25）"),
    ("faiss", "向量索引（提示词指定）"),
    ("torch", "张量后端（CPU 版）"),
    ("transformers", "CLIP / 本地嵌入"),
    ("sentence_transformers", "本地嵌入降级"),
    ("PIL", "图像处理"),
    ("scipy", "数值工具"),
    ("sklearn", "工具"),
    ("fastapi", "HTTP 服务"),
    ("uvicorn", "ASGI 服务器"),
    ("pytest", "测试框架"),
    ("requests", "HTTP 客户端"),
    ("openai", "兼容 vLLM/SGLang 后端"),
    ("huggingface_hub", "权重下载"),
    ("cv2", "图像处理（缺失）"),
    ("pytesseract", "OCR（缺失）"),
    ("paddleocr", "OCR（缺失）"),
    ("easyocr", "OCR（缺失）"),
    ("onnxruntime", "推理加速（缺失）"),
    ("gradio", "Web 界面（缺失）"),
    ("streamlit", "Web 界面（缺失）"),
    ("pdfplumber", "表格解析（缺失）"),
    ("pandas", "数据处理（缺失）"),
    ("loguru", "日志（缺失）"),
    ("matplotlib", "绘图（缺失）"),
    ("open_clip", "多模态（缺失）"),
    ("clip", "多模态（缺失）"),
    ("timm", "视觉骨干（缺失）"),
    ("accelerate", "模型加载加速（缺失）"),
]

# 第 5 题载体（PDF2 物理页 39）组织结构图应出现的 29 个节点
节点39 = [
    "股东大会", "董事会", "总经理", "监事会", "董事会秘书", "证券部", "上海分公司",
    "薪酬与考核委员会", "提名委员会", "战略委员会", "审计委员会", "内部审计部",
    "财务部", "市场开发部", "信息系统部", "销售部", "行政人事部", "物流部",
    "研发中心", "电话及网络销售部", "渠道销售部", "大客户销售部", "国际贸易部",
    "珠海销售处", "深圳销售处", "北京销售处", "武汉销售处", "广州销售处", "成都销售处",
]

_RESULT: dict[str, Any] = {}


def _log(事件: str, **字段: Any) -> None:
    """工单4 结构化日志：单行 JSON，含工单编号，禁止只写「开始/结束」。"""
    payload = {"工单": GONGDAN, "事件": 事件}
    payload.update(字段)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def 步骤(名称: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """装饰器：记录函数入口/出口的结构化日志（函数名、输入摘要、输出摘要、耗时、异常堆栈）。"""

    def 包装(fn: Callable[..., Any]) -> Callable[..., Any]:
        def 内层(*args: Any, **kwargs: Any) -> Any:
            输入摘要 = {
                "args": [str(a)[:120] for a in args],
                "kwargs": {k: str(v)[:120] for k, v in kwargs.items()},
            }
            _log("函数入口", 函数=fn.__qualname__, 步骤=名称, 输入摘要=输入摘要)
            t0 = time.perf_counter()
            try:
                out = fn(*args, **kwargs)
            except Exception as exc:  # 禁止静默失败：记日志后继续传播，由上层统一降级
                _log(
                    "函数异常",
                    函数=fn.__qualname__,
                    步骤=名称,
                    耗时ms=round((time.perf_counter() - t0) * 1000, 2),
                    异常类型=type(exc).__name__,
                    异常信息=str(exc)[:400],
                    异常堆栈=traceback.format_exc()[:1200],
                )
                raise
            _log(
                "函数出口",
                函数=fn.__qualname__,
                步骤=名称,
                耗时ms=round((time.perf_counter() - t0) * 1000, 2),
                输出摘要=str(out)[:200],
            )
            return out

        return 内层

    return 包装


@步骤("§2 解释器与依赖来源")
def 探测解释器() -> dict[str, Any]:
    """探测解释器身份：venv 前缀、base 前缀、pip 落点、每个包的物理来源。

    关键动因：本机 venv 是 `--system-site-packages` 型 venv，绝大多数包其实来自
    Anaconda 基础环境，pip install 的落点又在工单1 目录内 —— 不实测就会踩红线。
    """
    import importlib
    import sysconfig

    包来源: dict[str, str] = {}
    for 包名, 用途 in DEPS:
        try:
            m = importlib.import_module(包名)
            包来源[包名] = f"{getattr(m, '__version__', '未知')} <- {getattr(m, '__file__', '未知')}"
        except Exception as exc:
            # 缺失是【事实】而非错误：如实登记异常类型与原信息
            包来源[包名] = f"缺失 {type(exc).__name__}: {str(exc)[:80]}（{用途}）"
    info = {
        "sys_executable": sys.executable,
        "python_version": platform.python_version(),
        "sys_prefix": sys.prefix,
        "sys_base_prefix": sys.base_prefix,
        "sys_path": sys.path,
        "pip安装落点_purelib": sysconfig.get_paths()["purelib"],
        "pip安装落点_scripts": sysconfig.get_paths()["scripts"],
        "包来源": 包来源,
        "platform": platform.platform(),
        "cpu_logical": os.cpu_count(),
        "环境变量": {
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "PYTHONUTF8": os.environ.get("PYTHONUTF8", ""),
            "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE", ""),
            "PYTHONIOENCODING": os.environ.get("PYTHONIOENCODING", ""),
            "HF_ENDPOINT": os.environ.get("HF_ENDPOINT", ""),
        },
    }
    _RESULT["§2_解释器与依赖"] = info
    return info


@步骤("§3 算力与多模态类")
def 探测算力() -> dict[str, Any]:
    """探测 torch 是否可用 CUDA、transformers 是否自带 CLIP 类（决定图像编码实现路径）。"""
    import torch

    info = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_build": torch.version.cuda,
    }
    try:
        import transformers

        info["transformers_version"] = transformers.__version__
        info["has_CLIPModel"] = hasattr(transformers, "CLIPModel")
        info["has_ChineseCLIPModel"] = hasattr(transformers, "ChineseCLIPModel")
    except Exception as exc:
        # 显式降级：CLIP 能力结论标为「无法判定」，不写入任何猜测值
        info["transformers_error"] = f"{type(exc).__name__}: {exc}"
        _log("显式降级", 步骤="§3 算力与多模态类", 原因="transformers 探测失败", 详情=str(exc)[:200])
    _RESULT["§3_算力"] = info
    return info


@步骤("§4 Ollama 冷热延迟")
def 探测ollama() -> dict[str, Any]:
    """探测 Ollama 版本、模型清单、bge-m3 嵌入维度、qwen2.5:3b 冷启动与预热后首字延迟。"""
    import requests

    base = "http://127.0.0.1:11434"
    info: dict[str, Any] = {"base": base}
    info["version"] = requests.get(base + "/api/version", timeout=5).json().get("version")
    info["models"] = [m["name"] for m in requests.get(base + "/api/tags", timeout=5).json().get("models", [])]
    r = requests.post(base + "/api/embed", json={"model": "bge-m3:latest", "input": ["测试"]}, timeout=60)
    emb = r.json().get("embeddings", [[]])[0]
    info["bge_m3_dims"] = len(emb)
    info["bge_m3_prefix"] = [round(float(x), 4) for x in emb[:4]]

    轮次: list[dict[str, Any]] = []
    for i in range(4):
        t0 = time.perf_counter()
        首字ms = None
        文本: list[str] = []
        with requests.post(
            base + "/api/generate",
            json={
                "model": "qwen2.5:3b",
                "prompt": "武汉力源信息技术股份有限公司的主营业务是什么？请一句话回答。",
                "stream": True,
                "options": {"num_predict": 120},
            },
            stream=True,
            timeout=180,
        ) as resp:
            for line in resp.iter_lines():
                if not line:
                    continue
                obj = json.loads(line.decode("utf-8"))
                if 首字ms is None:
                    首字ms = round((time.perf_counter() - t0) * 1000, 1)
                文本.append(obj.get("response", ""))
                if obj.get("done"):
                    break
        轮次.append(
            {
                "轮次": i + 1,
                "首字ms": 首字ms,
                "总耗时ms": round((time.perf_counter() - t0) * 1000, 1),
                "样例输出": "".join(文本)[:80],
            }
        )
    info["生成实测"] = 轮次
    预热 = [x["首字ms"] for x in 轮次[1:] if x["首字ms"] is not None]
    if 预热:
        info["预热后首字_ms_最小"] = min(预热)
        info["预热后首字_ms_最大"] = max(预热)
    info["说明"] = "第 1 轮含模型加载（冷启动），第 2 轮起为预热口径；验收延迟必须用预热口径并独占 Ollama"
    _RESULT["§4_Ollama"] = info
    return info


@步骤("§5 网络与提权")
def 探测网络() -> dict[str, Any]:
    """探测外网可达性、HF 镜像上 CLIP 权重可达性与实测下载速率。

    结论将直接决定 T2 走「下载 CLIP 权重」还是「如实记录不可用」，
    因此必须给出响应码与实测速率，不得只写「有网 / 没网」。
    """
    import requests

    out: dict[str, Any] = {}
    for 名称, 地址 in (
        ("baidu", "https://www.baidu.com"),
        ("pypi清华镜像", "https://pypi.tuna.tsinghua.edu.cn/simple/"),
        ("hf-mirror首页", "https://hf-mirror.com"),
    ):
        try:
            r = requests.get(地址, timeout=15)
            out[名称] = f"HTTP {r.status_code}（{len(r.content)} 字节）"
        except Exception as exc:
            out[名称] = f"{type(exc).__name__}: {str(exc)[:200]}"
    # 中文 CLIP 权重可达性：先取 config.json（小），再 HEAD 权重（不下载）
    仓库 = "https://hf-mirror.com/OFA-Sys/chinese-clip-vit-base-patch16/resolve/main/"
    try:
        r = requests.get(仓库 + "config.json", timeout=20)
        out["CLIP_config.json"] = f"HTTP {r.status_code} {len(r.content)} 字节"
        out["CLIP_architectures"] = r.json().get("architectures") if r.status_code == 200 else None
    except Exception as exc:
        out["CLIP_config.json"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    try:
        r = requests.head(仓库 + "pytorch_model.bin", timeout=20, allow_redirects=True)
        out["CLIP_pytorch_model.bin"] = f"HTTP {r.status_code} Content-Length={r.headers.get('Content-Length')}"
        out["CLIP权重字节"] = int(r.headers.get("Content-Length") or 0)
    except Exception as exc:
        out["CLIP_pytorch_model.bin"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    try:
        r = requests.head(仓库 + "model.safetensors", timeout=20, allow_redirects=True)
        out["CLIP_model.safetensors"] = f"HTTP {r.status_code}（404 表示仓库只有 .bin）"
    except Exception as exc:
        out["CLIP_model.safetensors"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    # 实测速率：只拉 32MB 片段，避免真的把 718MB 拉完
    try:
        t0 = time.time()
        字节 = 0
        with requests.get(仓库 + "pytorch_model.bin", stream=True, timeout=60,
                          headers={"Range": "bytes=0-33554431"}) as r:
            for chunk in r.iter_content(1 << 20):
                字节 += len(chunk)
                if 字节 >= (1 << 25) or time.time() - t0 > 30:
                    break
        dt = time.time() - t0
        速率 = 字节 / 1048576 / dt if dt > 0 else 0.0
        out["实测下载速率_MBps"] = round(速率, 2)
        out["718MB预计耗时秒"] = round(753177983 / 1048576 / 速率, 1) if 速率 > 0 else None
    except Exception as exc:
        out["实测下载速率_MBps"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    _RESULT["§5_网络"] = out
    return out


@步骤("§6 页码口径")
def 探测页码口径() -> dict[str, Any]:
    """实测「页脚印刷页码」与「物理页码」的关系，防止引用页码整篇错位。

    方法：取每页 y 坐标最大的文本 span（页脚一定在最底部），
    而不是取 get_text() 的最后一行 —— 后者在表格/跨页页面上不是页脚。
    """
    import pymupdf

    out: dict[str, Any] = {"方法": "取该页 y 最大的文本 span 作为页脚；同时全页正则搜 1-1-N"}
    for 标签, 路径, 页集 in (
        ("PDF1", PDF1, [1, 2, 51, 129, 130, 131, 401, 548]),
        ("PDF2", PDF2, [1, 22, 39, 40, 72, 113, 157, 340, 350]),
    ):
        doc = pymupdf.open(路径)
        rows = []
        for 物理页 in 页集:
            if 物理页 > doc.page_count:
                continue
            page = doc[物理页 - 1]
            spans = []
            for b in page.get_text("dict")["blocks"]:
                for line in b.get("lines", []):
                    for s in line.get("spans", []):
                        if s["text"].strip():
                            spans.append((float(s["bbox"][3]), s["text"].strip()))
            spans.sort()
            页脚 = spans[-1][1] if spans else ""
            匹配 = re.findall(r"1-1-(\d+)", page.get_text())
            数字 = int(页脚) if 页脚.isdigit() else None
            正则值 = int(匹配[0]) if 匹配 else None
            # 印刷页码取「页脚为纯数字」或「页脚形如 1-1-N」两条取值路径中实际命中的那条
            印刷 = 数字 if 数字 is not None else 正则值
            rows.append(
                {
                    "物理页码": 物理页,
                    "0based索引": 物理页 - 1,
                    "页脚span": 页脚,
                    "页脚纯数字": 数字,
                    "1-1-N正则值": 正则值,
                    "印刷页码": 印刷,
                    "印刷等于物理减1": (印刷 == 物理页 - 1) if 印刷 is not None else None,
                }
            )
        doc.close()
        out[标签] = rows
    out["结论口径"] = "引用一律用 1-based 物理页码（pymupdf 页序 d[i] → 物理 i+1）；页脚仅用于人工核对"
    _RESULT["§6_页码口径"] = out
    return out


@步骤("§7 语料与图像题载体")
def 探测语料() -> dict[str, Any]:
    """探测两份 PDF 的大小/哈希/页数，并复核第 5、6 题的证据页与图像归属。"""
    import pymupdf

    out: dict[str, Any] = {}
    for 标签, 路径 in (("招股说明书1.pdf", PDF1), ("招股说明书2.pdf", PDF2)):
        h = hashlib.sha256()
        with open(路径, "rb") as f:
            for 块 in iter(lambda: f.read(1 << 20), b""):
                h.update(块)
        doc = pymupdf.open(路径)
        out[标签] = {
            "size": os.path.getsize(路径),
            "sha256": h.hexdigest(),
            "sha256_16": h.hexdigest()[:16].upper(),
            "页数": doc.page_count,
            "producer": (doc.metadata or {}).get("producer"),
            "加密": bool(doc.is_encrypted),
            "需要口令": bool(doc.needs_pass),
        }
        doc.close()

    doc = pymupdf.open(PDF2)
    # ---- 第 5 题：物理页 39 组织结构图（矢量框图 + 文本层，且文本层逐字成 span）----
    p39 = doc[38]
    raw = p39.get_text()
    norm = re.sub(r"\s+", "", raw)
    span_items = []
    for b in p39.get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            for s in line.get("spans", []):
                if s["text"].strip():
                    span_items.append(
                        {
                            "y": round(float(s["bbox"][1]), 1),
                            "x": round(float(s["bbox"][0]), 1),
                            "text": s["text"].strip(),
                        }
                    )
    关键y = {}
    for 词 in ("销售部", "财务部", "物流部", "大客户销售部", "渠道销售部", "国际贸易部",
               "电话及网络销售部", "珠海销售处", "成都销售处", "上海分公司", "研发中心"):
        ys = [it["y"] for it in span_items if it["text"] == 词]
        关键y[词] = ys[:6]
    # 行聚类复原（反面示例）：把 span 按 y 容差 6pt 归行、行内按 x 拼接 —— 实测会串行，不能用于本图
    行列表: list[dict[str, Any]] = []
    for it in sorted(span_items, key=lambda z: (z["y"], z["x"])):
        if it["y"] > 700:  # 跳过页脚
            continue
        if not 行列表 or abs(it["y"] - 行列表[-1]["y"]) > 6:
            行列表.append({"y": it["y"], "x起": it["x"], "片段": [it]})
        else:
            行列表[-1]["片段"].append(it)
    复原行 = []
    for 行 in 行列表:
        行["片段"].sort(key=lambda z: z["x"])
        复原行.append(
            {
                "y": round(行["y"], 1),
                "x起": round(行["x起"], 1),
                "文本": "".join(z["text"] for z in 行["片段"])[:60],
            }
        )

    # 列聚类复原（正确算法）：x 相近(±6pt) 且 y 递增连续(间隔<14pt) 归为同一竖排文本块，块内按 y 拼接
    列列表: list[dict[str, Any]] = []
    for it in sorted(span_items, key=lambda z: (z["x"], z["y"])):
        if it["y"] > 700:
            continue
        已归入 = False
        for 列 in 列列表:
            if abs(列["x"] - it["x"]) <= 6 and it["y"] - 列["y末"] < 14 and it["y"] > 列["y末"]:
                列["片段"].append(it)
                列["y末"] = it["y"]
                已归入 = True
                break
        if not 已归入:
            列列表.append({"x": it["x"], "y起": it["y"], "y末": it["y"], "片段": [it]})
    复原列 = [
        {
            "x": 列["x"],
            "y起": 列["y起"],
            "y末": 列["y末"],
            "文本": "".join(z["text"] for z in 列["片段"]),
        }
        for 列 in 列列表
    ]
    复原列.sort(key=lambda z: (z["y起"], z["x"]))
    复原文本集合 = {z["文本"] for z in 复原列}
    out["第5题_物理页39"] = {
        "drawings数": len(p39.get_drawings()),
        "images数": len(p39.get_images(full=True)),
        "原始文本长度": len(raw),
        "归一化文本长度": len(norm),
        "原始文本是否含「销售部」子串": "销售部" in raw,
        "归一化后节点命中数": sum(1 for k in 节点39 if k in norm),
        "归一化后缺失节点": [k for k in 节点39 if k not in norm],
        "归一化文本": norm,
        "span总数": len(span_items),
        "关键节点y": 关键y,
        "列聚类复原结果": 复原列,
        "列聚类是否复原全部29节点": all(any(k == z["文本"] for z in 复原列) for k in 节点39),
        "列聚类未复原节点": [k for k in 节点39 if not any(k == z["文本"] for z in 复原列)],
        "行聚类复原结果_反面示例": 复原行,
        "取数要点": "框图下半部分（财务部…销售处）是【竖排文字，每字一个 span】；按行聚合同 y 字符会把 6 个方框串成一行（见反面示例），"
                    "必须按 x 列聚类、列内按 y 升序拼接，才能得到 财务部/市场开发部/销售部/大客户销售部/珠海销售处 等正确节点名",
    }
    # 第 5 题交叉验证页
    cross = {}
    for 页码 in (40, 113, 239):
        t = doc[页码 - 1].get_text()
        cross[f"物理{页码}"] = {
            "销售部下设": "销售部下设" in t,
            "渠道销售部": "渠道销售部" in t,
            "大客户销售部": "大客户销售部" in t,
            "销售处": "销售处" in t,
        }
    out["第5题_交叉验证"] = {
        "物理40": "销售部下设渠道销售部、电话及网络销售部、大客户销售部和国际贸易部（4 个部门）",
        "物理113": "上海 1 家分公司；北京、广州、成都、深圳、武汉、珠海各 1 家销售处（共 6 处）",
        "命中明细": cross,
    }
    # ---- 第 6 题：物理页 72 两张位图并列（不是重叠）----
    p72 = doc[71]
    imgs72 = []
    for it in p72.get_image_info(xrefs=True):
        imgs72.append(
            {
                "xref": it.get("xref"),
                "宽": it.get("width"),
                "高": it.get("height"),
                "bbox": [round(float(v), 1) for v in it.get("bbox", ())],
            }
        )
    图候选 = [x for x in imgs72 if x["xref"] not in (2338,)]
    左 = min(图候选, key=lambda x: x["bbox"][0]) if 图候选 else None
    右 = max(图候选, key=lambda x: x["bbox"][0]) if 图候选 else None
    out["第6题_物理页72"] = {
        "images数": len(p72.get_images(full=True)),
        "drawings数": len(p72.get_drawings()),
        "image_info": imgs72,
        "图表候选": 图候选,
        "左侧图": 左,
        "右侧图": 右,
        "左侧图内容": "xref=1693 638x479 = 3D 饼图「2008 年中国IC 市场应用结构(亿元)」"
                      "（计算机 2515/42%、消费 1541/26%、网络通信 1189/20%、工控 448/7%、其他 137/2%、汽车 102/2%、IC卡 48/1%）",
        "右侧图内容": "xref=1692 525x473 = 横向条形图「增长率」"
                      "（汽车 14.0%、工控 10.5%、计算机 7.9%、其他 6.1%、消费 5.1%、网络通信 2.6%、IC卡 -2.0%）",
        "两图关系": "左右并列（bbox 左 89.9~318.2 / 右 317.3~505.1，仅 0.9pt 相接），不是重叠绘制",
        "标题命中": [k for k in ("2008", "IC", "应用结构与增长", "赛迪顾问") if k in p72.get_text()],
        "正文交叉证据_仅次于汽车电子": "仅次于汽车电子" in p72.get_text(),
        "正文交叉句": "工业控制领域所占的比例虽然仅为7%，但其增长率达10.5%，在所有应用行业中位列第二，仅次于汽车电子。",
        "图内容来源": "本文件作者用 read_image 视觉核对 xref1692/1693 导出的原图后录入（见 设计/复现脚本/证据图/）",
    }
    doc.close()
    _RESULT["§7_语料"] = out
    return out


@步骤("§8 图像资产全量扫描")
def 探测图像资产() -> dict[str, Any]:
    """逐页扫描 548+350 页：统计内嵌图片、矢量绘图元素与重复图（水印）分布。

    目的：图像解析方案必须先知道「图是什么形态」（位图 / 矢量 / 水印），
    否则去重与候选筛选策略会全错。
    """
    import pymupdf

    out: dict[str, Any] = {}
    for 标签, 路径 in (("招股说明书1.pdf", PDF1), ("招股说明书2.pdf", PDF2)):
        doc = pymupdf.open(路径)
        t0 = time.perf_counter()
        xref计数: dict[int, int] = {}
        图片总数 = 绘图总数 = 0
        含图页 = 含绘图页 = 0
        绘图页分布: dict[str, int] = {"0": 0, "1-9": 0, "10-49": 0, "50-199": 0, ">=200": 0}
        色彩空间: dict[str, int] = {}
        for i in range(doc.page_count):
            page = doc[i]
            imgs = page.get_images(full=True)
            drs = page.get_drawings()
            图片总数 += len(imgs)
            绘图总数 += len(drs)
            含图页 += 1 if imgs else 0
            含绘图页 += 1 if drs else 0
            n = len(drs)
            if n == 0:
                绘图页分布["0"] += 1
            elif n < 10:
                绘图页分布["1-9"] += 1
            elif n < 50:
                绘图页分布["10-49"] += 1
            elif n < 200:
                绘图页分布["50-199"] += 1
            else:
                绘图页分布[">=200"] += 1
            for img in imgs:
                xref计数[img[0]] = xref计数.get(img[0], 0) + 1
                色彩空间[f"cs={img[5]} bpc={img[4]}"] = 色彩空间.get(f"cs={img[5]} bpc={img[4]}", 0) + 1
        排行 = sorted(xref计数.items(), key=lambda x: -x[1])[:5]
        out[标签] = {
            "页数": doc.page_count,
            "图片总数": 图片总数,
            "绘图元素总数": 绘图总数,
            "含图页数": 含图页,
            "含绘图页数": 含绘图页,
            "绘图页分布": 绘图页分布,
            "色彩空间分布": 色彩空间,
            "重复度最高xref": [
                {
                    "xref": x,
                    "出现次数": c,
                    "占比": f"{c / 图片总数:.1%}",
                    "说明": "跨页水印图（八维教育），去重时必须按 xref 过滤"
                    if c >= doc.page_count * 0.5
                    else "个别页复用的普通图，属正常重复",
                }
                for x, c in 排行
            ],
            "独立xref数": len(xref计数),
            "扫描耗时秒": round(time.perf_counter() - t0, 2),
        }
        doc.close()
    _RESULT["§8_图像资产"] = out
    return out


def main() -> int:
    """入口：顺序执行全部探测步骤；单步失败只降级不中断，保证一次运行取全证据。"""
    ap = argparse.ArgumentParser(description="工单4 环境事实复现")
    ap.add_argument("--skip-scan", action="store_true", help="跳过 898 页全量图像扫描")
    ap.add_argument("--skip-net", action="store_true", help="跳过联网探测（离线复核用）")
    args = ap.parse_args()
    _log("脚本入口", 脚本="环境事实复现.py", 参数=vars(args), 根目录=ROOT, 解释器=sys.executable)

    计划: list[tuple[str, Callable[[], Any]]] = [
        ("§2 解释器与依赖来源", 探测解释器),
        ("§3 算力与多模态类", 探测算力),
        ("§6 页码口径", 探测页码口径),
        ("§7 语料与图像题载体", 探测语料),
        ("§4 Ollama 冷热延迟", 探测ollama),
    ]
    if not args.skip_net:
        计划.append(("§5 网络与提权", 探测网络))
    if not args.skip_scan:
        计划.append(("§8 图像资产全量扫描", 探测图像资产))

    for 名称, fn in 计划:
        try:
            fn()
        except Exception as exc:
            # 显式降级而非中断：单步失败不影响其它结论取证，但必须留痕
            _log("步骤降级", 步骤=名称, 原因=f"{type(exc).__name__}: {exc}", 处理="记日志后继续后续步骤")
            _RESULT.setdefault("失败步骤", {})[名称] = f"{type(exc).__name__}: {str(exc)[:300]}"

    输出 = os.path.join(ROOT, "设计", "复现脚本", "环境事实复现结果.json")
    with open(输出, "w", encoding="utf-8") as f:
        json.dump(_RESULT, f, ensure_ascii=False, indent=2)
    _log("脚本出口", 脚本="环境事实复现.py", 落盘=输出, 失败步骤数=len(_RESULT.get("失败步骤", {})))
    print("\n========== 汇总（完整结果见 %s）==========" % 输出, flush=True)
    print(json.dumps(_RESULT, ensure_ascii=False, indent=2)[:20000], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
