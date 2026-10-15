# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-18-实现文档质量评估Skill并集成至智能体工作流
# 模块职责：遍历目标文件夹完成格式分布统计；逐文件探测（MD5、文本长度、PDF 页面类型、
#           SimHash、敏感信息）并落盘中间态，支持进度反馈与中断恢复。

import os                                                          # 路径与遍历
import re                                                          # 后缀匹配
import json                                                        # 状态与统计落盘
import time                                                        # 耗时统计
import hashlib                                                     # MD5 精确去重
import zipfile                                                     # docx 纯标准库解析
import xml.etree.ElementTree as ET                                 # docx 文本抽取

from . import pdf_probe                                            # PDF 页面类型识别
from . import sensitive                                            # 敏感信息检测
from . import simhash as simhash_mod                               # 近似重复指纹

_WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"  # docx 命名空间
_SAMPLE_CHARS = 1200                                               # 每条记录保留的文本样例长度


def discover_files(target, extensions, recursive=True):
    """遍历文件夹或文件列表，返回受支持文件的绝对路径列表（已排序）。"""
    exts = set(e.lower() for e in extensions)                      # 统一小写比较
    found = []                                                     # 结果
    if isinstance(target, (list, tuple)):                          # 直接给定文件列表
        for item in target:
            p = os.path.abspath(item)                              # 绝对化
            if os.path.isfile(p) and os.path.splitext(p)[1].lower() in exts:
                found.append(p)                                    # 后缀受支持才收
            elif os.path.isdir(p):
                found.extend(_walk(p, exts, recursive))            # 列表里混了目录也支持
    elif os.path.isfile(target):                                   # 单个文件
        if os.path.splitext(target)[1].lower() in exts:
            found.append(os.path.abspath(target))
    elif os.path.isdir(target):                                    # 单个目录
        found.extend(_walk(target, exts, recursive))
    else:
        raise IOError("目标路径不存在或不可访问：%s" % target)
    return sorted(set(found))                                      # 去重排序，保证可复现


def _walk(root, exts, recursive):
    """递归/非递归遍历目录，收集受支持文件。"""
    out = []                                                       # 结果
    if recursive:                                                  # 递归模式
        for cur, _dirs, names in os.walk(root):
            for n in names:
                if os.path.splitext(n)[1].lower() in exts:
                    out.append(os.path.abspath(os.path.join(cur, n)))
    else:                                                          # 只看顶层
        for n in os.listdir(root):
            p = os.path.join(root, n)
            if os.path.isfile(p) and os.path.splitext(n)[1].lower() in exts:
                out.append(os.path.abspath(p))
    return out


def format_stats(paths):
    """格式分布统计：各类后缀的数量与占比。"""
    total = len(paths)                                             # 文件总数
    by_ext, size_by_ext = {}, {}                                   # 计数与体积
    for p in paths:                                                # 逐文件
        ext = os.path.splitext(p)[1].lower() or "(无后缀)"           # 后缀
        by_ext[ext] = by_ext.get(ext, 0) + 1                       # 计数
        try:
            size_by_ext[ext] = size_by_ext.get(ext, 0) + os.path.getsize(p)
        except OSError:                                            # 读取失败跳过体积
            pass
    rows = []                                                      # 明细行
    for ext in sorted(by_ext, key=lambda e: (-by_ext[e], e)):       # 按数量降序
        rows.append({"ext": ext, "count": by_ext[ext],             # 后缀 / 数量
                     "ratio": round(by_ext[ext] / total, 4) if total else 0.0,
                     "size_bytes": size_by_ext.get(ext, 0),
                     "size_mb": round(size_by_ext.get(ext, 0) / 1048576.0, 2)})
    return {"total": total, "by_ext": rows,
            "total_size_mb": round(sum(size_by_ext.values()) / 1048576.0, 2)}


def md5_of(path, chunk=1 << 20):
    """计算文件 MD5；读取失败返回空串。"""
    h = hashlib.md5()                                              # MD5 句柄
    try:
        with open(path, "rb") as fh:                               # 分块读，避免大文件吃内存
            while True:
                block = fh.read(chunk)
                if not block:
                    break
                h.update(block)
    except (IOError, OSError):
        return ""
    return h.hexdigest()


def extract_text(path, ext):
    """按后缀抽取可提取的文本：返回 (文本, 错误信息)。"""
    try:
        if ext in (".md", ".txt", ".csv", ".json"):
            for enc in ("utf-8", "gbk", "utf-16"):                 # 中文文档常见编码
                try:
                    with open(path, "r", encoding=enc, errors="strict") as fh:
                        return fh.read(), ""
                except UnicodeDecodeError:
                    continue
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read(), "编码非 UTF-8/GBK，已用替换字符解码"
        if ext in (".docx", ".pptx", ".xlsx"):
            return _ooxml_text(path), ""                           # OOXML 容器
    except Exception as exc:                                       # 任何解析异常都记录
        return "", "%s: %s" % (type(exc).__name__, exc)
    return "", ""


def _ooxml_text(path):
    """用标准库从 OOXML（docx/pptx/xlsx）抽取可见文本，避免额外依赖。"""
    parts = []                                                     # 文本片段
    with zipfile.ZipFile(path) as zf:                              # 打开容器
        for name in zf.namelist():                                 # 遍历内部条目
            if not name.endswith(".xml"):                          # 只看 XML
                continue
            if not re.search(r"(word/document|ppt/slides/slide|xl/sharedStrings)", name):
                continue                                           # 只取正文/幻灯片/共享字符串
            try:
                root = ET.fromstring(zf.read(name))                # 解析
            except ET.ParseError:
                continue                                           # 损坏条目跳过
            for node in root.iter():                               # 收集所有文本节点
                if node.tag.endswith("}t") and node.text:
                    parts.append(node.text)                        # 正文文本
    return "\n".join(parts)


def probe_one(path, cfg):
    """探测单个文件，返回统一结构的记录 dict（失败时 error 字段非空）。"""
    ext = os.path.splitext(path)[1].lower()                        # 后缀
    rec = {"path": path, "name": os.path.basename(path), "ext": ext,
           "size_bytes": 0, "text_chars": 0, "text_sample": "",
           "md5": "", "simhash": None, "sensitive": [],
           "page_count": None, "scan_pages": None, "scan_ratio": None,
           "pdf_kind": None, "error": ""}                          # 默认骨架
    try:
        rec["size_bytes"] = os.path.getsize(path)                  # 体积
    except OSError as exc:
        rec["error"] = "读取文件属性失败：%s" % exc                  # 记录后继续
        return rec
    rec["md5"] = md5_of(path)                                      # 精确去重用
    try:
        if ext == ".pdf":                                          # PDF：逐页判定页面类型
            text, pdf_info = pdf_probe.probe_pdf(path, cfg)        # 探测
            rec.update(pdf_info)                                   # 页面级信息
        else:
            text, err = extract_text(path, ext)                    # 其他格式直接抽文本
            if err:
                rec["error"] = err
    except Exception as exc:                                       # 损坏文件不中断整体流程
        rec["error"] = "解析失败：%s: %s" % (type(exc).__name__, exc)
        return rec
    rec["text_chars"] = len(text or "")                            # 有效字符数
    rec["text_sample"] = (text or "")[:_SAMPLE_CHARS]              # 保留样例供人工核验
    if text and len(text) >= cfg.get("dedup.simhash_min_chars", 200):
        rec["simhash"] = simhash_mod.fingerprint(text,              # 近似重复指纹
                                                 cfg.get("dedup.simhash_bits", 64))
    rec["sensitive"] = sensitive.detect(text or "", cfg)           # 敏感信息（带上下文）
    return rec


class StateStore(object):
    """断点续跑状态：已完成的文件记录落盘，重跑时直接跳过。"""

    def __init__(self, state_file, resume=True):
        self.file = state_file                                     # 状态文件路径
        self.resume = resume                                       # 是否启用续跑
        self.records = {}                                          # 路径 → 记录
        if resume and os.path.exists(state_file):                  # 载入已有进度
            try:
                with open(state_file, "r", encoding="utf-8") as fh:
                    self.records = json.load(fh).get("records", {})
            except (ValueError, IOError):
                self.records = {}                                  # 状态损坏则从零开始

    def get(self, path):
        """取已完成记录；无则返回 None。"""
        return self.records.get(path) if self.resume else None

    def put(self, path, record):
        """登记一条记录。"""
        self.records[path] = record

    def dump(self):
        """把当前进度落盘。"""
        d = os.path.dirname(self.file)                             # 目录
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = self.file + ".tmp"                                   # 先写临时文件再改名
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"records": self.records, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                      fh, ensure_ascii=False)
        os.replace(tmp, self.file)                                 # 原子替换，避免半截文件


def probe_files(paths, cfg, state, progress_every=100, on_record=None):
    """逐文件探测主循环：带进度反馈与中断恢复。返回记录列表。"""
    records, reused = [], 0                                        # 结果与复用计数
    t0 = time.time()                                               # 计时
    for i, p in enumerate(paths, start=1):                         # 逐文件
        rec = state.get(p)                                         # 已有记录
        if rec is not None:                                        # 断点续跑命中
            reused += 1
        else:
            rec = probe_one(p, cfg)                                # 现场探测
            state.put(p, rec)                                      # 登记
        records.append(rec)                                        # 收集
        if on_record is not None:                                  # 逐条回调（供缓存按键落盘）
            on_record(rec)
        if i % max(progress_every, 1) == 0 or i == len(paths):     # 定期反馈进度
            state.dump()                                           # 落盘便于中断恢复
            speed = i / max(time.time() - t0, 1e-6)                # 处理速度
            print("[质检进度] %d/%d 已完成（复用 %d，%.1f 文件/秒，累计 %.0f 秒）"
                  % (i, len(paths), reused, speed, time.time() - t0), flush=True)
    state.dump()                                                   # 收尾落盘
    return records
