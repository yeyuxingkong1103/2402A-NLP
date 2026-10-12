# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 文件说明：PDF 解析层。MinerU 为主（开启表格识别），pymupdf/pdfplumber/paddleocr 逐级兜底，输出统一块结构。
import json, os, re, subprocess, sys                                              # 读产物、环境变量、正则、起子进程、解释器路径
from pathlib import Path                                                          # 路径类型
from . import config as C                                                         # 复用集中配置

# 编号式章节标题的常见写法：第X节/章、"三、"、"（二）"、"3." 等，用于补足 MinerU 漏判的标题
_HEAD = re.compile(r"^(?:第[一二三四五六七八九十百]+[节章]|[一二三四五六七八九十]+、|（[一二三四五六七八九十]+）|\d+[、.])")


def _mineru_python() -> str:                                                      # MinerU 所在解释器
    """优先用配置指定的解释器；未配置或路径不存在时回落当前解释器。"""
    # 注意：路径为空串时 Path("") 会指到当前目录且 exists() 为真，必须先判空再判存在
    if C.MINERU_PYTHON and Path(C.MINERU_PYTHON).exists(): return C.MINERU_PYTHON   # 配置了且确实存在
    return sys.executable                                                         # 否则用当前解释器；MinerU 未安装会走降级分支，不会静默出错


def blocks_from_content_list(raw: list, doc: str) -> list:                        # MinerU 产物 → 统一块结构
    """保留正文、标题与表格；丢弃每页重复的页眉页脚与页码，表格原样带出 HTML 供结构化。"""
    blocks, title, tpage = [], "", 0                                              # blocks 收集结果；title/tpage 记最近章节名及其页码
    for item in raw:                                                              # 遍历 MinerU 的每个内容块
        page = int(item.get("page_idx", 0)) + 1                                   # MinerU 页号从 0 起，统一转成与 PDF 一致的 1 起
        kind = item.get("type")                                                   # 块类型
        if kind == "table":                                                       # 表格块
            cap = item.get("table_caption") or ""                                 # 表题可能是字符串也可能是列表
            cap = cap[0] if isinstance(cap, list) and cap else str(cap)           # 统一成字符串，空表题给空串
            html = item.get("table_body") or ""                                   # 表格内容只在 table_body 字段，text 字段里没有
            if html:                                                              # 有表体才登记
                blocks.append(dict(doc=doc, type="table", page_no=page, content="",  # 内容留空，等 table_struct 结构化后回填
                                   html=html, caption=cap.strip(), section=title))  # 带上表题与所属章节，供行块补语境
            continue                                                              # 表格不参与标题判定
        if kind == "text" and (item.get("text") or "").strip():                   # 正文或标题块
            text = item["text"].strip()                                           # 取文本
            # 标题两种来源：MinerU 标的 text_level，以及编号式章节名（如"3、报告期内曾为关联方…"）。
            # 实测坑：漏判编号标题会让后续页的表格继承上一节的章节名，出现"张冠李戴"的语境，
            # 进而让"曾为关联方"的表被当成"不存在控制关系的关联方"表，答案混入无关企业。
            is_head = bool(item.get("text_level")) or bool(_HEAD.match(text) and len(text) <= 40)  # 判定是否标题
            if is_head: title, tpage = text, page                                 # 更新章节名与其页码
            elif page - tpage > C.SECTION_SPAN: title = ""                        # 标题语境超过设定页数即失效，宁缺勿错
            blocks.append(dict(doc=doc, type="title" if is_head else "text",        # 记录块类型
                               page_no=page, content=text))                        # 记录页码与内容
    if not blocks: raise RuntimeError("MinerU 未产出有效内容")                      # 一个块都没有说明解析失败，抛出让上层降级
    return blocks                                                                # 返回统一结构


def parse_with_mineru(pdf_path: Path, doc: str) -> list:                          # MinerU 主解析路径
    """调用 MinerU pipeline 后端解析；命中比 PDF 更新的已有产物则直接复用，避免重复跑几十分钟。"""
    out_root = C.PARSED_DIR / doc                                                 # 每篇文档独立输出目录
    cached = sorted(out_root.rglob("*_content_list.json")) if out_root.exists() else []  # 查找已有解析产物
    if cached and cached[0].stat().st_mtime >= pdf_path.stat().st_mtime:          # 产物比 PDF 新即可复用
        return blocks_from_content_list(json.loads(cached[0].read_text(encoding="utf-8")), doc)  # 直接转换返回
    env = dict(os.environ)                                                        # 复制当前环境变量
    env["PYTHONPATH"] = C.MINERU_LANGFIX_DIR + os.pathsep + env.get("PYTHONPATH", "")  # 注入中文路径补丁，否则 fasttext 打不开模型
    env["MINERU_MODEL_SOURCE"] = "modelscope"                                     # 权重下载源用国内镜像
    cmd = [_mineru_python(), "-c", "from mineru.cli.client import main; main()",   # 以模块入口调用 MinerU CLI
           "-p", str(pdf_path), "-o", str(out_root), "-b", "pipeline",            # 输入 PDF、输出目录、pipeline 后端
           "-l", "ch", "-t", "True"]                                              # 中文文档、开启表格识别（工单核心）
    subprocess.run(cmd, env=env, check=True, capture_output=True, timeout=C.MINERU_TIMEOUT)  # 同步执行并设超时保护
    found = sorted(out_root.rglob("*_content_list.json"))                          # 递归查找解析结果
    if not found: raise RuntimeError("MinerU 未生成 content_list.json")            # 无产物即失败，抛出让上层降级
    return blocks_from_content_list(json.loads(found[0].read_text(encoding="utf-8")), doc)  # 转换并返回


def parse_with_fast(pdf_path: Path, doc: str) -> list:                            # 降级解析：pymupdf + pdfplumber
    """pymupdf 抽正文、pdfplumber 抽表格，输出与 MinerU 相同的块结构（表格带 HTML 便于统一结构化）。"""
    import pymupdf, pdfplumber                                                    # 延迟导入，缩短服务启动时间
    blocks, handle = [], pymupdf.open(str(pdf_path))                              # 结果列表与 PDF 句柄
    for i in range(handle.page_count):                                            # 逐页抽取正文
        for para in re.split(r"\n{2,}", handle[i].get_text()):                    # 按空行切段，避免整页挤成一块
            if len(para.strip()) >= 10: blocks.append(dict(doc=doc, type="text",   # 过短碎片多为页眉页脚，丢弃
                                                          page_no=i + 1, content=para.strip()))  # 记录页码与内容
    handle.close()                                                                # 释放文件句柄
    with pdfplumber.open(str(pdf_path)) as pdf:                                   # 打开同一 PDF 抽表格
        for i, page in enumerate(pdf.pages):                                      # 逐页处理
            for t in page.extract_tables() or []:                                 # 每页可能有 0~N 张表
                rows = [r for r in t if any((c or "").strip() for c in r)]        # 丢掉全空行
                if len(rows) < 2: continue                                        # 至少两行才算有效表格
                html = "<table>" + "".join("<tr>" + "".join(f"<td>{c or ''}</td>" for c in r) + "</tr>" for r in rows) + "</table>"  # 转成与 MinerU 同构的 HTML
                blocks.append(dict(doc=doc, type="table", page_no=i + 1, content="",  # 表格块
                                   html=html, caption="", section=""))            # 降级路径拿不到表题与章节，留空
    return blocks                                                                 # 返回块列表


def _paddle_ocr():                                                                 # 构造 PaddleOCR 识别器
    """优先用纯英文路径的模型；路径不存在时退回默认缓存目录，便于非中文用户名的机器直接跑。"""
    from paddleocr import PaddleOCR                                                # 延迟导入，避免拖慢正常启动
    dirs = [(k, str(Path(C.PADDLEOCR_MODEL_DIR) / sub)) for k, sub in              # 三个模型子目录：检测、识别、方向分类
            (("det_model_dir", "det/ch/ch_PP-OCRv4_det_infer"),                    # 检测模型
             ("rec_model_dir", "rec/ch/ch_PP-OCRv4_rec_infer"),                     # 识别模型
             ("cls_model_dir", "cls/ch_ppocr_mobile_v2.0_cls_infer"))]              # 方向分类模型
    kw = {k: v for k, v in dirs if Path(v).exists()}                                # 只传确实存在的目录，缺哪个就用 PaddleOCR 默认值
    if not kw: raise RuntimeError(                                                  # 一个都没有说明没准备模型
        f"PaddleOCR 模型目录不存在：{C.PADDLEOCR_MODEL_DIR}。"
        "请先执行 cp -r ~/.paddleocr/whl/{det,rec,cls} C:/paddleocr_models/，"
        "或改用 MinerU 解析（用户名含中文时 Paddle 打不开 ~/.paddleocr 下的模型）")
    return PaddleOCR(lang="ch", show_log=False, **kw)                               # 显式指定模型路径


def parse_with_ocr(pdf_path: Path, doc: str, max_pages: int = 20) -> list:         # 扫描件兜底：PaddleOCR
    """前若干页渲染成位图走 OCR，仅在文本层为空时使用，把耗时控制在可接受范围。"""
    import pymupdf                                                                # 延迟导入
    ocr, handle, blocks = _paddle_ocr(), pymupdf.open(str(pdf_path)), []           # 识别器、句柄、结果列表
    for i in range(min(handle.page_count, max_pages)):                            # 只处理前若干页
        tmp = C.ROOT / "data" / f"_ocr_{doc}_{i}.png"                             # 位图临时文件
        handle[i].get_pixmap(dpi=150).save(str(tmp))                              # 渲染落盘供 OCR 读取
        lines = [x[1][0] for x in (ocr.ocr(str(tmp), cls=False) or [[]])[0] or []]  # 抽取每行识别文本
        tmp.unlink(missing_ok=True)                                               # 删除临时图片
        if len("\n".join(lines).strip()) >= 10: blocks.append(dict(doc=doc, type="text",  # 有文本才保留该页
                                                                   page_no=i + 1, content="\n".join(lines)))  # 记录页码与内容
    handle.close()                                                                # 释放句柄
    return blocks                                                                 # 返回块列表


def parse_pdf(pdf_path: Path, use_mineru: bool = True) -> tuple:                  # 解析总入口
    """按 MinerU → pymupdf/pdfplumber → PaddleOCR 逐级降级，返回 (块列表, 实际解析器名)。"""
    pdf_path = Path(pdf_path)                                                     # 统一成 Path
    if not pdf_path.exists(): raise FileNotFoundError(f"知识库文件不存在：{pdf_path}")  # PDF 不存在时明确报错
    doc = pdf_path.stem                                                           # 文档名作为块归属标识
    if use_mineru:                                                                # 允许时优先 MinerU
        try: return parse_with_mineru(pdf_path, doc), "mineru"                     # 成功即返回
        except Exception as exc: print(f"[解析降级] {doc} MinerU 失败：{type(exc).__name__}: {exc}")  # 覆盖中文路径、超时、OOM
    blocks = parse_with_fast(pdf_path, doc)                                       # 降级到快速解析
    if sum(len(b["content"]) for b in blocks) < 200:                              # 文本层近乎为空说明是扫描件
        print(f"[解析降级] {doc} 文本层为空，转 PaddleOCR")                          # 提示进入 OCR 分支
        blocks = parse_with_ocr(pdf_path, doc)                                    # OCR 兜底
    if not blocks: raise RuntimeError(f"PDF 解析失败：{doc} 未提取到任何文本，请确认文件未损坏")  # 三条路径都没结果则明确报错
    return blocks, "fast"                                                         # 返回块列表与实际解析器名
