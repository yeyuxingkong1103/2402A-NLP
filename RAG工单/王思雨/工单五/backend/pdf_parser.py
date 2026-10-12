# 工单编号：人工智能NLP-RAG-Query理解优化
# 文件说明：PDF 解析层。主解析器 MinerU，失败时依次降级 pymupdf / pdfplumber；扫描页转 PaddleOCR。
#           产出统一的"块"结构（含来源文件名、页码、类型、文本），供语义分块与检索使用。
import json                                                                     # 读取 MinerU 的 content_list.json
import re                                                                       # 正则：清洗 HTML 表格与空白
import subprocess                                                               # 调用 MinerU 命令行
import sys                                                                      # 取当前解释器路径，供 MinerU 子进程使用
from pathlib import Path                                                        # 跨平台路径
from typing import List, Dict                                                    # 类型标注

from . import config as C                                                       # 复用集中配置
from .text_clean import clean_text                                              # 复用统一的水印清洗（招股说明书带斜向水印，表格必被污染）


# ==================== 一、MinerU 产物读取（主路径，命中缓存时最快）====================
def _mineru_content_list(pdf_name: str) -> Path:                                # 推算 MinerU 产物路径
    """MinerU 的输出目录形如 data/parsed/<doc>/<doc>/auto/<doc>_content_list.json。"""
    stem = Path(pdf_name).stem                                                  # 文档主名，如 招股说明书1
    return C.PARSED_DIR / stem / stem / "auto" / f"{stem}_content_list.json"    # 拼接标准产物路径


def _table_html_to_text(html: str) -> str:                                      # 表格 HTML 转可检索文本
    """把 MinerU 的 <table> 结构压成"单元格 | 单元格"的行文本；表格题必须能按行命中。"""
    if not html:                                                                # 空表直接返回空串
        return ""                                                               # 结束
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.S | re.I)          # 逐个抽取表格行
    lines = []                                                                  # 收集结果行
    for row in rows:                                                            # 遍历每一行
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.S | re.I)  # 抽取该行单元格
        cells = [clean_text(c) for c in cells]                                  # 去标签 + 剔水印 + 压空白，一次到位
        if any(cells):                                                          # 整行皆空则跳过
            lines.append(" | ".join(cells))                                     # 用竖线分隔，便于阅读与命中
    return "\n".join(lines)                                                     # 汇总为多行文本


def load_mineru_blocks(pdf_name: str) -> List[Dict]:                            # 读取 MinerU 缓存产物
    """把 content_list.json 转成本项目的统一块结构；产物不存在时返回空列表表示"无缓存"。"""
    path = _mineru_content_list(pdf_name)                                       # 定位产物文件
    if not path.exists():                                                       # 没有缓存
        return []                                                               # 交由上层降级
    items = json.loads(path.read_text(encoding="utf-8"))                        # MinerU 的条目列表
    stem = Path(pdf_name).stem                                                  # 文档主名
    img_root = path.parent                                                      # 图片相对路径的根目录
    blocks: List[Dict] = []                                                     # 统一块列表
    for it in items:                                                            # 遍历每个条目
        itype = str(it.get("type", "text"))                                     # 条目类型
        page_no = int(it.get("page_idx", 0)) + 1                                # MinerU 页码 0 起，统一转成 1 起
        text = ""                                                               # 该条目的文本
        image_path = ""                                                         # 该条目的图片路径
        if itype in ("text", "header", "page_number"):                          # 纯文本类
            text = str(it.get("text", "") or "")                                # 直接取文本
        elif itype == "table":                                                  # 表格类
            text = _table_html_to_text(str(it.get("table_body", "") or ""))     # 表格体转行文本
            caps = it.get("table_caption") or []                                # 表题（列表）
            if caps:                                                            # 有表题就前置，便于"某某表"的提问命中
                text = "表题：" + " ".join(map(str, caps)) + "\n" + text          # 拼接表题
            image_path = str(it.get("img_path", "") or "")                      # 表格截图路径，重排/展示可参考
        elif itype in ("image", "chart"):                                       # 图像/图表类
            caps = it.get("image_caption") or it.get("chart_caption") or []     # 图题
            text = " ".join(map(str, caps)) if caps else ""                     # 有图题先记下
            image_path = str(it.get("img_path", "") or "")                      # 图片相对路径
        elif itype == "equation":                                               # 公式类
            text = str(it.get("text", "") or "")                                # 取公式文本
        else:                                                                   # 其它未知类型
            text = str(it.get("text", "") or "")                                # 尽力取文本
        if not text.strip() and not image_path:                                 # 文本与图都没有，丢弃
            continue                                                            # 跳过空条目
        blocks.append({"doc": stem, "page_no": page_no, "block_type": itype,    # 组装统一块
                       "text": text.strip(),                                    # 文本内容
                       "image_path": str((img_root / image_path)) if image_path else ""})  # 图片绝对路径
    return blocks                                                               # 返回全部块


# ==================== 二、MinerU 实跑（无缓存时触发）====================
def run_mineru(pdf_path: Path) -> bool:                                         # 调用 MinerU 命令行解析
    """在无缓存时实跑 MinerU；成功返回 True。中文用户名下必须给 langfix 目录，否则 fasttext 打不开模型。"""
    if not C.MINERU_ENABLED:                                                    # 开关关闭则直接不尝试
        return False                                                            # 交由降级链
    out_dir = C.PARSED_DIR / pdf_path.stem                                      # 输出目录
    out_dir.mkdir(parents=True, exist_ok=True)                                  # 确保目录存在
    env = dict(**__import__("os").environ)                                      # 复制当前环境变量
    if C.MINERU_LANGFIX_DIR:                                                    # 配置了中文路径补丁目录
        env["MINERU_LANGFIX_DIR"] = C.MINERU_LANGFIX_DIR                        # 传给 MinerU 子进程
    exe = C.MINERU_PYTHON or sys.executable                                     # 用配置的解释器，缺省用当前解释器
    cmd = [exe, "-m", "mineru.cli.client", "-p", str(pdf_path), "-o", str(out_dir)]  # MinerU 命令行入口
    try:                                                                        # 解析属于"可能失败"的操作
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True,     # 捕获输出便于排障
                              timeout=C.MINERU_TIMEOUT)                         # 超时保护：548 页约需数分钟
        if proc.returncode != 0:                                                # 非零退出码即失败
            print(f"[PDF解析] MinerU 失败({proc.returncode})：{(proc.stderr or '')[-300:]}")  # 打印尾部错误
            return False                                                        # 转降级
        return _mineru_content_list(pdf_path.name).exists()                     # 以产物是否存在判定成功
    except Exception as exc:                                                    # 超时、找不到模块等
        print(f"[PDF解析] MinerU 异常：{type(exc).__name__}: {exc}")             # 明确报因
        return False                                                            # 转降级


# ==================== 三、降级链：pymupdf → pdfplumber → OCR ====================
def _parse_with_pymupdf(pdf_path: Path) -> List[Dict]:                          # 第一降级：pymupdf 直接抽文字
    """按页抽取文字层；文字过少的页标记为扫描页，交由 OCR 分支处理。"""
    import pymupdf                                                              # 延迟导入，缩短冷启动
    blocks: List[Dict] = []                                                     # 统一块列表
    doc = pymupdf.open(str(pdf_path))                                           # 打开 PDF
    for pno in range(doc.page_count):                                           # 逐页处理
        page = doc[pno]                                                         # 当前页
        text = (page.get_text("text") or "").strip()                            # 抽取文字层
        if len(text) < C.OCR_MIN_CHARS:                                         # 文字太少 → 判定为扫描页
            blocks.append({"doc": pdf_path.stem, "page_no": pno + 1,            # 记录扫描页占位块
                           "block_type": "scanned", "text": "", "image_path": ""})  # 文本留空，待 OCR 填充
        else:                                                                   # 正常文字页
            blocks.append({"doc": pdf_path.stem, "page_no": pno + 1,            # 页码转 1 起
                           "block_type": "text", "text": text, "image_path": ""})  # 记录正文
    doc.close()                                                                 # 关闭释放句柄
    return blocks                                                               # 返回块列表


def _parse_with_pdfplumber(pdf_path: Path) -> List[Dict]:                       # 第二降级：pdfplumber
    """pymupdf 失败时用 pdfplumber 兜底，逐页抽文字。"""
    import pdfplumber                                                           # 延迟导入
    blocks: List[Dict] = []                                                     # 统一块列表
    with pdfplumber.open(str(pdf_path)) as pdf:                                 # 上下文管理器打开
        for pno, page in enumerate(pdf.pages):                                  # 逐页处理
            text = (page.extract_text() or "").strip()                          # 抽取文字
            blocks.append({"doc": pdf_path.stem, "page_no": pno + 1,            # 页码转 1 起
                           "block_type": "text" if text else "scanned",         # 有字为正文，无字为扫描页
                           "text": text, "image_path": ""})                     # 记录内容
    return blocks                                                               # 返回块列表


# ==================== 四、对外主入口 ====================
def parse_pdf(pdf_path: Path) -> List[Dict]:                                    # 解析单篇 PDF
    """三级解析：MinerU 缓存 → MinerU 实跑 → pymupdf → pdfplumber；均失败则抛出明确异常。"""
    blocks = load_mineru_blocks(pdf_path.name)                                  # 先看 MinerU 缓存（最快）
    if blocks:                                                                  # 命中缓存
        print(f"[PDF解析] {pdf_path.name}：命中 MinerU 缓存，共 {len(blocks)} 个块")  # 打印来源，便于确认口径
        return blocks                                                           # 直接返回
    if run_mineru(pdf_path):                                                    # 无缓存则实跑 MinerU
        blocks = load_mineru_blocks(pdf_path.name)                              # 读取新产物
        if blocks:                                                              # 解析成功
            print(f"[PDF解析] {pdf_path.name}：MinerU 实跑完成，共 {len(blocks)} 个块")  # 记录
            return blocks                                                       # 返回
    for fn, name in ((_parse_with_pymupdf, "pymupdf"), (_parse_with_pdfplumber, "pdfplumber")):  # 依次降级
        try:                                                                    # 每一级都独立兜底
            blocks = fn(pdf_path)                                               # 执行该级解析
            if blocks:                                                          # 有结果就用
                print(f"[PDF解析] {pdf_path.name}：MinerU 不可用，降级到 {name}，共 {len(blocks)} 个块")  # 明示降级
                return blocks                                                   # 返回
        except Exception as exc:                                                # 该级失败
            print(f"[PDF解析] {name} 失败：{type(exc).__name__}: {exc}")         # 记录后继续降级
    raise RuntimeError(f"PDF 解析失败：{pdf_path.name}（MinerU/pymupdf/pdfplumber 均不可用）")  # 明确报错，不静默


def parse_all() -> List[Dict]:                                                  # 解析知识库内全部 PDF
    """扫描 data/raw 下的所有 PDF 并汇总为块列表，供入库脚本调用。"""
    all_blocks: List[Dict] = []                                                 # 汇总容器
    for pdf in C.raw_pdfs():                                                    # 遍历知识库
        all_blocks.extend(parse_pdf(pdf))                                       # 逐篇解析并合并
    return all_blocks                                                           # 返回全部块
