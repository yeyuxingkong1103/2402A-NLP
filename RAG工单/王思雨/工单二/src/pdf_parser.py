# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
"""PDF 解析模块：MinerU 为主，PyMuPDF / pdfplumber / PaddleOCR 逐级兜底，输出统一块结构。"""
import html as _html, json, os, re, subprocess, sys                                          # HTML 反转义、读产物、环境变量、正则、起子进程、解释器路径
from pathlib import Path                                                                     # 路径类型
from . import config as C                                                                    # 复用集中配置


def html_table_to_text(t: str) -> str:                                                       # MinerU 表格 HTML → 可检索文本
    """按行还原表格、单元格用 | 分隔：表格内容只在 table_body 字段，只读 text 会丢掉全部表格。"""
    out = []                                                                                 # 每行文本
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", t or "", flags=re.S | re.I):                # 逐个提取表格行
        cells = [re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", c))).strip()     # 每格去标签、反转义、压空白
                 for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.S | re.I)]  # 提取该行所有单元格
        if any(cells): out.append(" | ".join(cells))                                          # 跳过全空行，用 | 连接保持列对应
    return "\n".join(out)                                                                    # 行间换行，便于向量与 BM25 命中


def blocks_from_content_list(raw: list) -> list:                                             # MinerU 产物 → 统一块结构
    """保留正文与标题，表格拼接最近章节标题补足语境；丢弃每页重复的页眉页脚与页码。"""
    blocks, title = [], ""                                                                    # blocks 存结果，title 记最近章节名
    for item in raw:                                                                          # 遍历 MinerU 的每个内容块
        page = int(item.get("page_idx", 0)) + 1                                                # MinerU 页号 0 起，统一转 1 起
        if item.get("type") == "table":                                                        # 表格块
            cap = item.get("table_caption") or ""                                              # 表题可能是列表
            cap = cap[0] if isinstance(cap, list) and cap else str(cap)                        # 统一成字符串
            body = html_table_to_text(item.get("table_body") or "")                             # 表体取自 table_body
            if body: blocks.append(dict(type="table", page_no=page,                            # 记录表格块
                                        content="\n".join(x for x in (title, cap, body) if x)))  # 章节名+表题+表体补足检索语境
        elif item.get("type") == "text" and (item.get("text") or "").strip():                   # 正文或标题块
            title = item["text"].strip() if item.get("text_level") else title                    # text_level 非空即章节标题，更新最近标题
            blocks.append(dict(type="title" if item.get("text_level") else "text", page_no=page,  # 记录块类型与页码
                               content=item["text"].strip()))                                   # 记录正文
    if not blocks: raise RuntimeError("MinerU 未产出有效内容")                                   # 一个块都没有说明解析失败，抛出让上层降级
    return blocks                                                                             # 返回统一结构


def parse_with_mineru(pdf_path: Path) -> list:                                                # MinerU 主解析路径
    """调用 MinerU pipeline 后端解析；命中已有产物则直接复用，避免重复跑几十分钟。"""
    out_root = C.ROOT / "data" / "parsed" / pdf_path.stem                                      # 每篇文档独立输出目录
    cached = list(out_root.rglob("*_content_list.json")) if out_root.exists() else []           # 查找已有解析产物
    if cached and cached[0].stat().st_mtime >= pdf_path.stat().st_mtime:                        # 产物比 PDF 新即可复用
        return blocks_from_content_list(json.loads(cached[0].read_text(encoding="utf-8")))       # 直接转换返回
    env = dict(os.environ)                                                                     # 复制当前环境变量
    env["PYTHONPATH"] = C.MINERU_LANGFIX_DIR + os.pathsep + env.get("PYTHONPATH", "")            # 注入中文路径补丁，否则 fasttext 打不开模型
    env["MINERU_MODEL_SOURCE"] = "modelscope"                                                  # 指定权重下载源为国内镜像
    cmd = [sys.executable, "-c", "from mineru.cli.client import main; main()", "-p", str(pdf_path),  # 以模块入口调用 MinerU CLI
           "-o", str(out_root), "-b", "pipeline", "-l", "ch", "-t", "True"]                     # 输出目录、pipeline 后端、中文、开启表格识别
    subprocess.run(cmd, env=env, check=True, capture_output=True, timeout=5400)                 # 同步执行并设超时保护（548 页需数分钟）
    found = list(out_root.rglob("*_content_list.json"))                                         # 递归查找解析结果
    if not found: raise RuntimeError("MinerU 未生成 content_list.json")                          # 无产物即失败，抛出让上层降级
    return blocks_from_content_list(json.loads(found[0].read_text(encoding="utf-8")))            # 转换并返回


def parse_with_fast(pdf_path: Path) -> list:                                                   # 降级解析：PyMuPDF + pdfplumber
    """PyMuPDF 抽正文、pdfplumber 抽表格，输出与 MinerU 相同的块结构。"""
    import pymupdf, pdfplumber                                                                 # 延迟导入，缩短服务启动时间
    blocks, doc = [], pymupdf.open(str(pdf_path))                                               # 结果列表与 PDF 句柄
    for i in range(doc.page_count):                                                             # 逐页抽取文本
        for para in re.split(r"\n{2,}", doc[i].get_text()):                                      # 按空行切段，避免整页一大块
            if len(para.strip()) >= 10: blocks.append(dict(type="text", page_no=i + 1, content=para.strip()))  # 过短碎片多为页眉页脚，丢弃
    doc.close()                                                                                 # 释放文件句柄
    with pdfplumber.open(str(pdf_path)) as pdf:                                                 # 打开同一 PDF 抽表格
        for i, page in enumerate(pdf.pages):                                                    # 逐页处理
            for table in page.extract_tables() or []:                                            # 每页可能有 0~N 张表
                rows = [r for r in (" | ".join((c or "").replace("\n", " ").strip() for c in row) for row in table) if r.strip(" |")]  # 单元格清洗连接并丢空行
                if len(rows) >= 2: blocks.append(dict(type="table", page_no=i + 1, content="\n".join(rows)))  # 至少两行才算有效表格
    return blocks                                                                               # 返回块列表


def parse_with_ocr(pdf_path: Path, max_pages: int = 20) -> list:                                 # 扫描件兜底：PaddleOCR
    """前若干页渲染成位图走 OCR，仅在文本层为空时使用，把耗时控制在可接受范围。"""
    import pymupdf                                                                              # 延迟导入
    from paddleocr import PaddleOCR                                                             # 延迟导入，避免拖慢正常启动
    ocr, doc, blocks = PaddleOCR(lang="ch", show_log=False), pymupdf.open(str(pdf_path)), []     # 中英文识别器、PDF 句柄、结果列表
    for i in range(min(doc.page_count, max_pages)):                                              # 只处理前若干页
        tmp = C.ROOT / "data" / f"_ocr_{i}.png"                                                   # 位图临时文件路径
        doc[i].get_pixmap(dpi=150).save(str(tmp))                                                # 渲染落盘供 OCR 读取
        lines = [x[1][0] for x in (ocr.ocr(str(tmp), cls=False) or [[]])[0] or []]                # 抽取每行识别文本
        tmp.unlink(missing_ok=True)                                                              # 删除临时图片
        if len("\n".join(lines).strip()) >= 10: blocks.append(dict(type="text", page_no=i + 1, content="\n".join(lines)))  # 有文本才保留该页
    doc.close()                                                                                 # 释放句柄
    return blocks                                                                               # 返回块列表


def parse_pdf(pdf_path: Path, use_mineru: bool = True) -> tuple:                                 # 解析总入口
    """按 MinerU → PyMuPDF/pdfplumber → PaddleOCR 逐级降级，返回 (块列表, 解析器名)。"""
    if not Path(pdf_path).exists(): raise FileNotFoundError(f"知识库文件不存在：{pdf_path}")        # PDF 不存在时明确报错
    if use_mineru:                                                                              # 允许时优先使用 MinerU
        try: return parse_with_mineru(Path(pdf_path)), "mineru"                                  # 成功即返回块列表与解析器名
        except Exception as exc: print(f"[解析降级] MinerU 失败：{type(exc).__name__}: {exc}")      # 覆盖中文路径、超时、OOM 等
    blocks = parse_with_fast(Path(pdf_path))                                                    # 降级到快速解析
    if sum(len(b["content"]) for b in blocks) < 200:                                            # 文本层近乎为空说明是扫描件
        print("[解析降级] 文本层为空，转 PaddleOCR")                                              # 提示进入 OCR 分支
        blocks = parse_with_ocr(Path(pdf_path))                                                 # OCR 兜底
    if not blocks: raise RuntimeError("PDF 解析失败：未提取到任何文本，请确认文件未损坏")           # 三条路径都没结果则明确报错
    return blocks, "fast"                                                                       # 返回块列表与实际解析器名
