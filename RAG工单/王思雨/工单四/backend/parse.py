# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
# 文件说明：PDF 解析层。MinerU 为主（同时产出正文、表格与**图像文件**），pymupdf/pdfplumber/paddleocr 逐级兜底。
import json, os, re, subprocess, sys                                        # 产物读取、环境变量、正则、子进程、解释器
from pathlib import Path                                                    # 路径类型
import config as C                                                          # 集中配置

_HEAD = re.compile(r"^(?:第[一二三四五六七八九十百]+[节章]|[一二三四五六七八九十]+、|（[一二三四五六七八九十]+）|\d+[、.])")  # 编号式章节标题
_TAG = re.compile(r"<[^>]+>")                                               # HTML 标签，用于把表格拍平成文本


def _mineru_python() -> str:                                                # MinerU 所在解释器
    """配置了独立解释器就用它；否则用当前解释器（本机 MinerU 与主程序同环境）。"""
    return C.MINERU_PYTHON if C.MINERU_PYTHON and Path(C.MINERU_PYTHON).exists() else sys.executable  # 判空再判存在


def _rows(html: str, caption: str, section: str, table_id: str = "") -> list:  # 表格 → 逐行块
    """把表格拍平成"表题 | 列名: 值"的逐行文本块：行级块让 BM25 与向量都能精确命中单行，
    同时每行都带 table_id，检索命中任一行时可据此把同表其余行一并取回（见 rag.expand_table_rows）。"""
    trs = re.findall(r"<tr[^>]*>(.*?)</tr>", html or "", re.S)               # 逐行
    cells = [[_TAG.sub("", c).replace("&nbsp;", " ").strip() for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, re.S)] for r in trs]  # 逐格去标签
    cells = [c for c in cells if any(c)]                                    # 丢掉全空行
    if not cells: return []                                                 # 无有效行
    head = cells[0]                                                         # 首行作列名
    out = []                                                                # 结果块
    for k, row in enumerate(cells[1:]):                                     # 逐数据行
        pairs = [f"{head[i]}: {v}" for i, v in enumerate(row) if v and i < len(head) and head[i]]  # 列名: 值
        if not pairs: continue                                              # 空行跳过
        label = row[0] if row else ""                                       # 首列常是行名
        out.append(dict(doc="", type="table_row", table_id=table_id,        # 行块类型与所属表标识
                        row_index=k + 1,                                    # 行号，1 起
                        content=f"[表] {caption}｜{section}｜{label}｜" + "；".join(pairs)))  # 表题+章节+行名+各列值
    return out                                                              # 返回逐行块


def from_content_list(raw: list, doc: str, base: Path) -> tuple:            # MinerU 产物 → 正文块 + 图像任务
    """保留正文、标题、表格；收集 image/chart 项交给图像解析。返回 (块列表, 图像任务列表)。"""
    blocks, images, title, tpage = [], [], "", 0                            # 结果、图像任务、最近章节名及其页码
    for k, item in enumerate(raw):                                          # 遍历内容块
        page = int(item.get("page_idx", 0)) + 1                             # MinerU 页号 0 起，统一为 1 起
        kind = item.get("type")                                             # 块类型
        if kind == "table":                                                 # 表格块
            cap = item.get("table_caption") or ""                           # 表题可能是列表
            cap = (cap[0] if isinstance(cap, list) and cap else str(cap)).strip()  # 统一成字符串
            tid = f"{doc}#p{page}#t{k}"                                      # 表标识：文档+页码+该页内序号，全库唯一
            for b in _rows(item.get("table_body") or "", cap, title, tid):   # 拍平成逐行块（内容只在这张表的 table_body 字段）
                b.update(doc=doc, page_no=page, section=title)              # 补文档、页码与章节
                blocks.append(b)                                            # 收下
            continue                                                        # 表格不参与标题判定
        if kind in ("image", "chart"):                                      # ★工单四新增：图像/图表项
            rel = item.get("img_path") or ""                                # 图像相对路径
            p = (base / rel) if rel else None                               # 拼成绝对路径
            if p and p.exists():                                            # 文件确实存在
                prev = next((raw[j].get("text") or "" for j in range(k - 1, max(-1, k - 6), -1)  # 往前走找最近的正文，作图的语境
                             if raw[j].get("type") == "text" and (raw[j].get("text") or "").strip()), "")  # 取到即用
                if not any(s in prev for s in C.SIGN_PAGE_KEYWORDS):         # 声明页整页签名图，跳过
                    images.append(dict(path=p, page=page, caption=prev.strip()[:120], type=kind))  # 登记图像任务
            continue                                                        # 图像项不产生文本块
        if kind == "text" and (item.get("text") or "").strip():              # 正文或标题
            text = item["text"].strip()                                     # 取文本
            # 标题两个来源：MinerU 的 text_level 标注，以及编号式章节名。
            # 坑：漏判编号标题会让后续表格继承上一节的章节名（"张冠李戴"），答案会混入无关章节的内容。
            is_head = bool(item.get("text_level")) or bool(_HEAD.match(text) and len(text) <= 40)  # 是否标题
            if is_head: title, tpage = text, page                            # 更新章节语境
            elif page - tpage > 3: title = ""                                # 章节语境跨页太多即失效，宁缺勿错
            blocks.append(dict(doc=doc, type="title" if is_head else "text",  # 块类型
                               page_no=page, content=text, section=title))   # 内容与章节
    if not blocks: raise RuntimeError("MinerU 未产出有效正文")                   # 无正文说明解析失败，抛出让上层降级
    return blocks, images                                                   # 返回


def parse_with_mineru(pdf: Path, doc: str) -> tuple:                        # MinerU 主解析路径
    """调 MinerU pipeline 后端；命中比 PDF 更新的已有产物则直接复用，避免重复跑几十分钟。"""
    out = C.PARSED_DIR / doc                                                # 每篇独立输出目录
    hit = sorted(out.rglob("*_content_list.json")) if out.exists() else []   # 已有产物
    if not hit:                                                             # 需要现跑
        env = dict(os.environ)                                              # 复制环境变量
        env["PYTHONPATH"] = C.MINERU_LANGFIX_DIR + os.pathsep + env.get("PYTHONPATH", "")  # 中文路径补丁，否则 fasttext 打不开模型
        env["MINERU_MODEL_SOURCE"] = "modelscope"                           # 权重源用国内镜像
        cmd = [_mineru_python(), "-c", "from mineru.cli.client import main; main()",  # 模块入口调用 MinerU CLI
               "-p", str(pdf), "-o", str(out), "-b", "pipeline",            # 输入、输出、pipeline 后端
               "-l", "ch", "-t", "True"]                                    # 中文文档、开启表格识别
        subprocess.run(cmd, env=env, check=True, capture_output=True, timeout=C.MINERU_TIMEOUT)  # 执行并设超时
        hit = sorted(out.rglob("*_content_list.json"))                       # 查找产物
    if not hit: raise RuntimeError("MinerU 未生成 content_list.json")           # 无产物即失败
    src = hit[0]                                                            # 取内容清单
    return from_content_list(json.loads(src.read_text(encoding="utf-8")), doc, src.parent)  # 图像路径相对清单所在目录


def parse_with_fast(pdf: Path, doc: str) -> tuple:                          # 降级解析：pymupdf 正文 + pdfplumber 表格 + 整页图
    """MinerU 不可用时兜底。图像退化为"整页渲染"：拿不到 MinerU 的精确图块，但图像语义解析链路依然成立。"""
    import pymupdf, pdfplumber                                              # 延迟导入
    blocks, images, h = [], [], pymupdf.open(str(pdf))                      # 结果与句柄
    idir = C.PARSED_DIR / doc / "_pages"                                    # 整页图输出目录
    idir.mkdir(parents=True, exist_ok=True)                                 # 确保存在
    for i in range(h.page_count):                                           # 逐页
        page = h[i]                                                         # 当前页
        for para in re.split(r"\n{2,}", page.get_text()):                   # 按空行切段
            if len(para.strip()) >= 10:                                     # 过短碎片多为页眉页脚
                blocks.append(dict(doc=doc, type="text", page_no=i + 1, content=para.strip(), section=""))  # 正文块
        if page.get_images(full=True):                                      # 该页含图
            f = idir / f"p{i + 1}.png"                                      # 落盘路径
            page.get_pixmap(dpi=200).save(str(f))                           # 200dpi 渲染，保证图内小字可读
            images.append(dict(path=f, page=i + 1, caption="", type="image"))  # 登记整页图
    h.close()                                                               # 释放句柄
    with pdfplumber.open(str(pdf)) as p:                                    # 用 pdfplumber 补表格
        for i, pg in enumerate(p.pages):                                    # 逐页
            for j, t in enumerate(pg.extract_tables() or []):               # 每页 0~N 张表
                rows_ = [r for r in t if any((c or "").strip() for c in r)]  # 丢空行
                if len(rows_) < 2: continue                                 # 至少两行
                html = "<table>" + "".join("<tr>" + "".join(f"<td>{c or ''}</td>" for c in r) + "</tr>" for r in rows_) + "</table>"  # 转成与 MinerU 同构的 HTML
                for b in _rows(html, "", "", f"{doc}#p{i + 1}#t{j}"):       # 统一走行块拍平，表标识口径与 MinerU 路径一致
                    b.update(doc=doc, page_no=i + 1, section="")            # 补文档与页码（降级路径拿不到表题与章节）
                    blocks.append(b)                                        # 收下
    if not blocks: raise RuntimeError(f"PDF 解析失败：{doc} 未提取到任何文本")      # 兜底也没结果则明确报错
    return blocks, images                                                   # 返回


def parse_pdf(pdf: Path, use_mineru: bool = True) -> tuple:                 # 解析总入口
    """按 MinerU → pymupdf/pdfplumber 逐级降级，返回 (正文块, 图像任务, 实际解析器名)。"""
    pdf = Path(pdf)                                                         # 统一 Path
    if not pdf.exists(): raise FileNotFoundError(f"知识库文件不存在：{pdf}")        # 文件缺失明确报错
    doc = pdf.stem                                                          # 文档名
    if use_mineru and C.MINERU_ENABLED:                                     # 允许且启用时优先 MinerU
        try: b, im = parse_with_mineru(pdf, doc); return b, im, "mineru"     # 成功即返回
        except Exception as e: print(f"[解析降级] {doc} MinerU 失败：{type(e).__name__}: {e}")  # 中文路径、超时、OOM 等
    b, im = parse_with_fast(pdf, doc)                                       # 降级
    return b, im, "fast"                                                    # 返回


def ocr_blank_pages(pdf: Path, doc: str, blocks: list) -> list:             # 扫描件兜底
    """文本层为空时用 PaddleOCR 补正文，把耗时控制在可接受范围。"""
    from backend import imaging as I                                        # 复用已配好的 OCR 入口
    import pymupdf                                                          # 延迟导入
    from PIL import Image                                                   # 位图读取
    h = pymupdf.open(str(pdf))                                              # 打开
    have = {b["page_no"] for b in blocks}                                   # 已有文本的页
    tmp = C.PARSED_DIR / doc / "_ocr"                                       # 位图目录
    tmp.mkdir(parents=True, exist_ok=True)                                  # 确保存在
    for i in range(h.page_count):                                           # 逐页
        if (i + 1) in have: continue                                        # 已有文本的页跳过
        f = tmp / f"p{i + 1}.png"                                           # 位图路径
        h[i].get_pixmap(dpi=200).save(str(f))                               # 渲染
        lines = [l[0] for l in I.read_lines(Image.open(f))]                 # 走图像 OCR 同一套清洗（含水印过滤）
        if len("".join(lines)) >= 10: blocks.append(dict(doc=doc, type="text", page_no=i + 1,  # 有内容才收
                                                          content="\n".join(lines), section=""))  # 正文块
    h.close()                                                               # 释放
    return blocks                                                           # 返回
