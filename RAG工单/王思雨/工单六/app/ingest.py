# 工单编号：人工智能NLP-RAG-混合检索任务
# 文件说明：离线入库。PDF → MinerU 解析（pymupdf 兜底）→ 语义分块 → 建多字段 BM25 索引 → 写 Milvus。
import json, os, re, subprocess, sys                                               # 标准库
from pathlib import Path                                                            # 路径
from . import config as C                                                           # 全局配置
from . import rag_core as R                                                         # 复用向量化与 Milvus 客户端

_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)                                     # 表格行
_CELL = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S)                              # 表格单元格
_TAG = re.compile(r"<[^>]+>")                                                       # HTML 标签
_SENT = re.compile(r"(?<=[。！？；!?;])")                                             # 句末标点
_ATOMIC = ("table", "image", "chart")                                               # 原子块类型：不做语义切分

_WM_CJK = re.compile(r"八维(?:教育)?(?:人工智能)?")                                  # 水印机构名（54 处已逐条核实，无合法用法）
_ASCII_RUN = re.compile(r"[A-Za-z]+")                                               # 整段 ASCII 字母串
_CJK = "一-鿿"                                                              # 汉字区间

def clean_cell(s: str) -> str:                                                      # 清洗表格单元格
    """去掉水印碎片。**词表是实测出来的，不是猜的**（见 docs/03-test.md §3.3）。

    本批招股书带「八维教育人工智能」斜向水印（还叠了下载者姓名），MinerU 的 OCR 把碎片读进了表格单元格：
    686 个表格块里 261 个（38.0%）含碎片，且会**把词切断**——「招股意向书」被污染成「招liumi股意向书」，
    「预计发行日期」变成「预计发行日期Liumin」。这既破坏 BM25 的精确匹配，也污染向量语义。

    规则只删两类**可确证**的成分，刻意不碰别的：
      · 机构名「八维/八维教/八维教育/八维人工智能」——54 处逐条看过，全是水印；
      · **紧邻汉字的小写 ASCII 串**（reng/eng/iumin/glium 等拼音碎片，实测 774 处、305 种）。
    为什么不删「教育/人工智能/敏」这些中文碎片：它们完全可能是正文里的合法词，误删代价高于收益。
    为什么只删小写：年份（2018）、合法缩写（IC/ERP/DAE）都是大写或数字，实测确认不会被这条规则误伤。
    """
    if not s:                                                                       # 空串直接返回
        return s                                                                    # 返回
    s = _WM_CJK.sub("", s)                                                          # 第一步：删水印机构名
    out, pos = [], 0                                                                # 第二步：删拼音碎片（按"整段 ASCII 串"判断，避免削掉合法英文的尾巴）
    for m in _ASCII_RUN.finditer(s):                                                # 逐个 ASCII 字母串
        w = m.group()                                                               # 该串
        near_cjk = (m.start() > 0 and _CJK[0] <= s[m.start() - 1] <= _CJK[2]) or (   # 紧邻汉字（左侧）
            m.end() < len(s) and _CJK[0] <= s[m.end()] <= _CJK[2])                   # 或右侧
        if near_cjk and w.islower():                                                # 只有"整串全小写"才判为拼音水印
            out.append(s[pos:m.start()])                                            # 抛弃这一段
            pos = m.end()                                                           # 跳过
    out.append(s[pos:])                                                             # 收尾
    return "".join(out).strip()                                                     # 返回

def table_text(html: str) -> str:                                                   # 表格转可检索文本
    """逐行拼成 "单元格 | 单元格"：让科目名与金额落在同一行，关键词与向量两路都能命中。"""
    rows = []                                                                       # 行文本
    for tr in _ROW.findall(html or ""):                                             # 遍历每行
        cells = [clean_cell(_TAG.sub("", c)) for c in _CELL.findall(tr)]            # 去标签取单元格并清洗水印碎片
        cells = [c for c in cells if c]                                             # 丢弃空格
        if cells:                                                                   # 非空行才保留
            rows.append(" | ".join(cells))                                          # 用竖线连接
    return "\n".join(rows)                                                          # 返回多行文本

def _cl_path(name: str) -> Path:                                                    # MinerU 产物路径
    """按通配定位 content_list.json，兼容两种并存的目录布局（实测踩过）。

    · 本机已有缓存（旧布局）：<out>/<名>/<名>/auto/<名>_content_list.json
    · 当前 MinerU 实跑（新布局）：<out>/<名>/hybrid_auto/<名>_content_list.json
    早期实现只写死了旧布局，导致 MinerU 明明跑成功、却因为找错目录被误判成"解析失败"。
    """
    base = C.PARSED_DIR / name                                                       # 该文档的产物根目录
    hits = sorted(base.rglob(f"{name}_content_list.json"))                           # 递归查找，深度不敏感
    return hits[0] if hits else base / name / "auto" / f"{name}_content_list.json"   # 找不到时返回约定路径供存在性判断

def run_mineru(pdf: Path) -> bool:                                                  # 实跑 MinerU
    """子进程调用 MinerU；中文用户名下必须注入 MINERU_LANGFIX_DIR，否则 fasttext 打不开模型必失败。"""
    out = C.PARSED_DIR / pdf.stem                                                    # 输出目录
    out.mkdir(parents=True, exist_ok=True)                                           # 确保存在
    env = dict(os.environ)                                                          # 复制环境变量
    env["MINERU_LANGFIX_DIR"] = C.MINERU_LANGFIX_DIR                                 # 注入中文路径补丁目录（部分版本读这个变量）
    # 关键：该目录里放的是 sitecustomize.py，只有把目录挂到 PYTHONPATH 才会在解释器启动时被自动导入，
    # 由它把 fast_langdetect 的模型路径改写到纯 ASCII 副本上。只设 MINERU_LANGFIX_DIR 是不够的——
    # 实测这样 MinerU 仍会报 "lid.176.ftz cannot be opened"，因为 fasttext 的 C 扩展打不开含中文的用户路径。
    env["PYTHONPATH"] = C.MINERU_LANGFIX_DIR + os.pathsep + env.get("PYTHONPATH", "")  # 前置到 PYTHONPATH
    cmd = [sys.executable, "-m", "mineru.cli.client", "-p", str(pdf), "-o", str(out)]  # 官方 CLI 入口
    try:                                                                            # 子进程可能超时或崩溃
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=C.MINERU_TIMEOUT)  # 同步执行
    except Exception as exc:                                                        # 超时/找不到模块
        print(f"[PDF解析] MinerU 调用异常：{type(exc).__name__}: {exc}")             # 明确报因
        return False                                                                # 视为失败
    if p.returncode != 0:                                                           # 非零退出码
        print(f"[PDF解析] MinerU 失败({p.returncode})：{(p.stderr or '')[-300:]}")   # 打印尾段错误
        return False                                                                # 视为失败
    return _cl_path(pdf.stem).exists()                                              # 以产物存在为准

def mineru_blocks(pdf: Path) -> list:                                               # 读 MinerU 产物
    """转成统一块结构；header/footer/page_number 直接丢弃——页眉页脚与页码是检索噪声。"""
    p = _cl_path(pdf.stem)                                                          # 产物路径
    if not p.exists():                                                              # 没有缓存
        return []                                                                   # 返回空，触发实跑
    out = []                                                                        # 结果
    for it in json.loads(p.read_text(encoding="utf-8")):                            # 逐条读
        t = it.get("type")                                                          # 条目类型
        page = int(it.get("page_idx") or 0) + 1                                     # 页码统一 1 起
        if t == "table":                                                            # 表格
            txt, bt = table_text(it.get("table_body")), "table"                     # 抽取单元格文本
        elif t in ("image", "chart"):                                               # 图片/图表
            txt, bt = " ".join(it.get("image_caption") or []), t                    # 用图注当文本
        elif t in ("text", "equation"):                                             # 正文/公式
            txt, bt = (it.get("text") or "").strip(), "text"                        # 直接取文本
        else:                                                                       # header/footer/page_number
            continue                                                                # 丢弃噪声
        if txt:                                                                     # 有内容才收
            out.append({"doc": pdf.stem, "page_no": page, "block_type": bt,         # 文档/页码/类型
                        "text": txt, "level": it.get("text_level") or 0})           # 文本与标题级别
    return out                                                                      # 返回块列表

def pymupdf_blocks(pdf: Path) -> list:                                              # pymupdf 兜底解析
    """MinerU 不可用时的降级路径：逐页抽纯文本，保证服务仍能入库（会丢表格结构）。"""
    import fitz                                                                     # pymupdf 的导入名
    doc = fitz.open(pdf)                                                            # 打开 PDF
    out = []                                                                        # 结果
    for i, pg in enumerate(doc):                                                    # 逐页
        txt = (pg.get_text() or "").strip()                                         # 全页文本
        if txt:                                                                     # 非空页
            out.append({"doc": pdf.stem, "page_no": i + 1, "block_type": "text",    # 页码 1 起
                        "text": txt, "level": 0})                                   # 无标题级别
    return out                                                                      # 返回

def pdfplumber_blocks(pdf: Path) -> list:                                           # pdfplumber 兜底解析
    """第二级兜底：pdfplumber 对分栏与表格的纯文本抽取往往比 pymupdf 更稳，代价是更慢。"""
    import pdfplumber                                                               # 延迟导入
    out = []                                                                        # 结果
    with pdfplumber.open(pdf) as doc:                                               # 打开 PDF
        for i, pg in enumerate(doc.pages, 1):                                       # 逐页（页码 1 起）
            txt = (pg.extract_text() or "").strip()                                 # 抽取整页文本
            if txt:                                                                 # 非空页
                out.append({"doc": pdf.stem, "page_no": i, "block_type": "text",    # 与 MinerU 块结构对齐
                            "text": txt, "level": 0})                               # 无标题级别
    return out                                                                      # 返回

def parse_pdf(pdf: Path) -> list:                                                   # 四级解析链
    """MinerU 缓存 → 实跑 MinerU → pymupdf → pdfplumber；全部失败才抛错，绝不静默返回空。

    注：工单技术选型里还列了 paddleocr。本工单的两份招股说明书都带完整文本层（实测 MinerU 与
    pymupdf 均能逐页抽出文字），扫描件识别属于工单四（图像内容解析）的范围，故这里不引入 OCR。
    """
    blks = mineru_blocks(pdf)                                                       # ① 先读缓存（最快）
    if blks:                                                                        # 命中缓存
        print(f"[PDF解析] {pdf.name} 命中 MinerU 缓存，{len(blks)} 块")                # 打印来源
        return blks                                                                 # 返回
    if C.MINERU_ENABLED and run_mineru(pdf):                                        # ② 实跑 MinerU
        blks = mineru_blocks(pdf)                                                   # 读产物
        if blks:                                                                    # 成功
            print(f"[PDF解析] {pdf.name} MinerU 解析完成，{len(blks)} 块")            # 打印
            return blks                                                             # 返回
    for fn, tag in ((pymupdf_blocks, "pymupdf"), (pdfplumber_blocks, "pdfplumber")):  # ③④ 两个兜底依次尝试
        try:                                                                        # 单个兜底失败不应影响下一个
            blks = fn(pdf)                                                          # 解析
        except Exception as exc:                                                    # 兜底自身报错（缺依赖等）
            print(f"[PDF解析] {tag} 失败：{type(exc).__name__}: {exc}")               # 明确报因
            continue                                                                # 试下一个
        if blks:                                                                    # 有结果
            print(f"[PDF解析] {pdf.name} 降级到 {tag}，{len(blks)} 块")                # 如实标注降级
            return blks                                                             # 返回
    raise RuntimeError(f"PDF 解析失败：{pdf.name}（MinerU/pymupdf/pdfplumber 均不可用）")  # 明确报错

def split_sentences(text: str, max_len: int = 200) -> list:                         # 断句
    """先按句末标点切，超长句再按逗号二次切，保证语义单元不超过 max_len。"""
    parts = []                                                                      # 结果
    for seg in _SENT.split(text or ""):                                             # 按句末标点切
        seg = seg.strip()                                                           # 去空白
        if not seg:                                                                 # 空段
            continue                                                                # 跳过
        if len(seg) <= max_len:                                                     # 长度合适
            parts.append(seg)                                                       # 直接收
            continue                                                                # 下一段
        buf = ""                                                                    # 超长句缓冲
        for piece in re.split(r"(?<=[，,])", seg):                                   # 按逗号再切
            if len(buf) + len(piece) > max_len and buf:                             # 超长
                parts.append(buf)                                                   # 冲刷
                buf = piece                                                         # 重新累积
            else:                                                                   # 还能装
                buf += piece                                                        # 累积
        if buf:                                                                     # 收尾
            parts.append(buf)                                                       # 收入
    return parts                                                                    # 返回句列表

def semantic_split(text: str, encode) -> list:                                      # 语义分块
    """按句向量与当前块中心的余弦相似度切块：语义漂移且已达标长才切，块长受 SEM_MAX_CHUNK 硬约束。"""
    sents = split_sentences(text)                                                   # 断句
    if not sents:                                                                   # 无句子
        return []                                                                   # 返回空
    if len(text) <= C.CHUNK_SIZE:                                                   # 短文本整块
        return [text.strip()]                                                       # 不再切
    vecs = encode(sents)                                                            # 句向量（已归一化）
    import numpy as np                                                              # 延迟导入
    out, cur, cur_len = [], [sents[0]], len(sents[0])                               # 结果与当前块
    for i in range(1, len(sents)):                                                  # 逐句判断
        center = np.mean(vecs[max(0, i - len(cur)):i], axis=0)                      # 当前块语义中心
        sim = float(np.dot(vecs[i], center))                                        # 归一化后点积即余弦
        too_long = cur_len + len(sents[i]) > C.SEM_MAX_CHUNK                        # 硬上限
        drift = sim < C.SEM_THRESHOLD and cur_len + len(sents[i]) > C.CHUNK_SIZE    # 语义漂移且够长
        if (too_long or drift) and cur:                                             # 满足切块条件
            body = "".join(cur).strip()                                             # 当前块文本
            if body:                                                                # 非空
                out.append(body)                                                    # 收块
            tail = body[-C.CHUNK_OVERLAP:] if C.CHUNK_OVERLAP else ""               # 重叠前缀
            cur = [tail + sents[i]]                                                 # 新块从重叠处起
            cur_len = len(cur[0])                                                   # 重置长度
        else:                                                                       # 不切
            cur.append(sents[i])                                                    # 并入当前块
            cur_len += len(sents[i])                                                # 累加长度
    body = "".join(cur).strip()                                                     # 收尾块
    if body:                                                                        # 非空
        out.append(body)                                                            # 收入
    return out                                                                      # 返回块列表

def chunk_blocks(blocks: list) -> list:                                             # 归并 + 分块
    """归并同页同章节的连续正文再做语义分块；标题只更新章节名不入块；表格/图片按行切分。"""
    segs, section, buf, bpage = [], "", [], None                                     # 段列表/章节/缓冲/缓冲页码
    for b in blocks:                                                                # 顺序遍历块
        txt, bt = b["text"], b["block_type"]                                        # 文本与类型
        if bt == "text" and b.get("level") and len(txt) < 60:                       # 标题行（MinerU 标了级别且短）
            if buf:                                                                 # 有缓冲先冲刷
                segs.append((section, bpage, "".join(buf), "text"))                 # 收为正文段
                buf = []                                                            # 清空
            section = txt.replace("\n", " ")[:120]                                  # 更新章节名（仅作字段）
            continue                                                                # 标题不入块
        if bt in _ATOMIC:                                                           # 表格/图片：原子块
            if buf:                                                                 # 冲刷正文缓冲
                segs.append((section, bpage, "".join(buf), "text"))                 # 收为正文段
                buf = []                                                            # 清空
            segs.append((section, b["page_no"], txt, bt))                           # 原子段独立
            continue                                                                # 下一块
        if buf and b["page_no"] != bpage:                                           # 正文跨页即断
            segs.append((section, bpage, "".join(buf), "text"))                     # 冲刷
            buf = []                                                                # 清空
        buf.append(txt + "\n")                                                      # 累积正文
        bpage = b["page_no"]                                                        # 记录页码
    if buf:                                                                         # 收尾冲刷
        segs.append((section, bpage, "".join(buf), "text"))                         # 最后一段
    chunks, cid = [], 0                                                             # 结果与块号计数
    for sec, page, text, bt in segs:                                                # 逐段分块
        if bt in _ATOMIC:                                                           # 原子块按行切，不切语义
            pieces, buf2 = [], ""                                                   # 行缓冲
            for line in text.split("\n"):                                           # 逐行
                if len(buf2) + len(line) > C.SEM_MAX_CHUNK and buf2:                # 超长即切
                    pieces.append(buf2)                                             # 收块
                    buf2 = ""                                                       # 重置
                buf2 += line + "\n"                                                 # 累积
            if buf2.strip():                                                        # 收尾
                pieces.append(buf2.strip())                                         # 收入
        else:                                                                       # 正文走语义分块
            pieces = semantic_split(text, lambda ss: R.embed(ss))                   # 复用 bge-m3 句向量
        pieces = [p[i:i + C.SEM_MAX_CHUNK] for p in pieces                          # 超过硬上限的块按字符硬切
                  for i in range(0, max(len(p), 1), C.SEM_MAX_CHUNK)]               # 保证没有任何块超长
        merged = []                                                                 # 合并后的块
        for piece in pieces:                                                        # 逐块处理
            piece = piece.strip()                                                   # 去空白
            if not piece:                                                           # 空块
                continue                                                            # 跳过
            if (merged and len(merged[-1]) < C.SEM_MIN_CHUNK                        # 前一块过小
                    and len(merged[-1]) + len(piece) <= C.SEM_MAX_CHUNK):           # 且合并后不超限
                merged[-1] = merged[-1] + " " + piece                               # 向后合并，避免碎片挤占检索前排
            else:                                                                   # 否则独立成块
                merged.append(piece)                                                # 收入
        for piece in merged:                                                        # 逐块组装
            cid += 1                                                                # 递增块号
            chunks.append({"chunk_id": f"tmp-{cid:05d}", "doc": "",                 # 块号与文档名由 to_records 统一覆盖
                           "title": sec, "text": piece.strip(),                     # 章节标题与正文
                           "page_no": page or 0, "page_end": page or 0,             # 页码
                           "block_type": bt, "section": sec})                       # 块类型与章节
    return chunks                                                                   # 返回

def to_records(chunks: list, doc: str) -> list:                                     # 补齐字段
    """统一补 chunk_id/doc/摘要/定位串：摘要取首 150 字（抽取式，非生成式）。"""
    for i, c in enumerate(chunks, 1):                                               # 重新编号
        c["chunk_id"] = f"{doc}-{i:05d}"                                            # 全局唯一块号
        c["doc"] = doc                                                              # 文档名
        c["source"] = doc                                                           # 工单要求的来源字段
        c["abstract"] = re.sub(r"\s+", " ", c["text"])[:150]                        # 抽取式摘要
        c["locator"] = f"{doc} P{c['page_no']}"                                     # 可读定位串
        c["page_no"] = int(c["page_no"] or 0)                                       # 页码强制整型，Milvus 要求
        c["page_end"] = c["page_no"]                                                # 块内不跨页
    return chunks                                                                   # 返回

def drop_boilerplate(chunks: list) -> list:                                         # 去页眉/单位标注噪声
    """丢弃跨页重复的短块：页眉的公司名行、表格"单位：万元"标注被 MinerU 标成 text 后会成批重复，
    并在向量召回里霸占前排（实测 14 字的公司名行相似度 0.743 高居第一，与问题无关）。
    判据=长度<40 字且全文归一化后出现≥3 次；本机实测该阈值丢掉的 108 块全部是这类噪声。"""
    import collections                                                              # 计数
    norm = lambda s: re.sub(r"[^0-9a-z一-鿿]+", "", (s or "").lower())        # 归一化去标点
    cnt = collections.Counter(norm(c["text"]) for c in chunks if len(c["text"]) < 40)  # 仅统计短块
    noise = {t for t, n in cnt.items() if n >= 3}                                   # 重复≥3 次即判为结构噪声
    keep = [c for c in chunks if not (len(c["text"]) < 40 and norm(c["text"]) in noise)]  # 过滤
    for i, c in enumerate(keep, 1):                                                 # 过滤后重新编号
        c["chunk_id"] = f"{c['doc']}-{i:05d}"                                       # 保持块号连续
    return keep                                                                     # 返回

def ensure_collection() -> None:                                                    # 建 Milvus 集合
    """删旧建新，字段与检索层 _FIELDS 对齐；向量用 IP 度量（已归一化，等价余弦相似度）。"""
    from pymilvus import DataType                                                   # 字段类型
    cli = R.milvus_client()                                                         # 客户端
    if cli.has_collection(C.MILVUS_COLLECTION):                                     # 已存在
        cli.drop_collection(C.MILVUS_COLLECTION)                                    # 先删，避免字段不一致
    sc = cli.create_schema(auto_id=True, enable_dynamic_field=False)                # 主键自增，字段固定
    sc.add_field("id", DataType.INT64, is_primary=True)                             # 自增主键
    sc.add_field("vector", DataType.FLOAT_VECTOR, dim=C.EMBED_DIM)                  # bge-m3 向量
    for name, ml in (("chunk_id", 64), ("doc", 128), ("title", 512), ("text", 65535),  # VARCHAR 字段与长度上限
                     ("abstract", 2048), ("source", 128), ("block_type", 16), ("locator", 256)):  # 与 _FIELDS 对应
        sc.add_field(name, DataType.VARCHAR, max_length=ml)                          # 逐个添加
    sc.add_field("page_no", DataType.INT16)                                         # 起始页
    sc.add_field("page_end", DataType.INT16)                                        # 结束页
    idx = cli.prepare_index_params()                                                # 索引参数
    idx.add_index(field_name="vector", index_type="IVF_FLAT", metric_type="IP",      # 倒排文件索引，十万级块足够
                  params={"nlist": 1024})                                           # 聚类数
    cli.create_collection(collection_name=C.MILVUS_COLLECTION, schema=sc, index_params=idx)  # 建集合

def insert_chunks(chunks: list) -> int:                                             # 写向量库
    """分批编码写入并逐批打印进度：本机 8G 显存下全量入库需数分钟，无进度无法分辨"慢"与"卡死"。"""
    cli, total = R.milvus_client(), 0                                               # 客户端与计数
    cols = ("chunk_id", "doc", "title", "text", "abstract",                   # 需要写入的业务字段
            "source", "page_no", "page_end", "block_type", "locator")                # 与集合定义一致
    for i in range(0, len(chunks), C.MILVUS_BATCH):                                 # 分批
        part = chunks[i:i + C.MILVUS_BATCH]                                         # 本批
        vecs = R.embed([c["text"] for c in part])                                   # 批量编码
        rows = [dict({"vector": v.tolist()}, **{k: (c.get(k) or "") for k in cols})  # 组装行
                for v, c in zip(vecs, part)]                                        # 向量与块一一对应
        for r, c in zip(rows, part):                                                # 修正数值字段类型
            r["page_no"] = int(c.get("page_no") or 0)                               # Milvus 要求整型
            r["page_end"] = int(c.get("page_end") or 0)                             # 同上
        cli.insert(collection_name=C.MILVUS_COLLECTION, data=rows)                   # 写入
        total += len(rows)                                                          # 累加
        print(f"  [Milvus] 已写入 {total}/{len(chunks)} 条", flush=True)              # 进度
    return total                                                                    # 返回写入条数

def run() -> dict:                                                                  # 入库主流程
    """解析 data/raw 下全部 PDF → 分块 → 建多字段 BM25 → 写 Milvus → 落盘索引。"""
    pdfs = sorted(C.RAW_DIR.glob("*.pdf"))                                          # 知识库 PDF
    if not pdfs:                                                                    # 没有 PDF
        raise RuntimeError(f"{C.RAW_DIR} 下没有 PDF，请先放入招股说明书")              # 明确报错
    chunks = []                                                                     # 全部块
    for pdf in pdfs:                                                                # 逐篇处理
        print(f"[入库] 解析 {pdf.name}", flush=True)                                  # 进度
        chunks += to_records(chunk_blocks(parse_pdf(pdf)), pdf.stem)                # 解析→分块→补字段
    before = len(chunks)                                                            # 过滤前块数
    chunks = drop_boilerplate(chunks)                                               # 去掉页眉/单位标注噪声
    print(f"[入库] 去噪声块 {before - len(chunks)} 条，剩 {len(chunks)} 块，"        # 进度
          f"开始建 BM25 与写 Milvus", flush=True)                                   # 起始提示
    store = R.bm25_build(chunks)                                                    # 建多字段 BM25
    R.bm25_save(store)                                                              # 落盘
    C.INDEX_FILE.write_text(json.dumps([{k: v for k, v in c.items() if not k.startswith("_")}  # 落盘纯文本块表
                                        for c in chunks], ensure_ascii=False), encoding="utf-8")  # 中文不转义
    ensure_collection()                                                             # 重建集合
    n = insert_chunks(chunks)                                                       # 写库
    R.bm25_load()                                                                   # 让检索层立即拿到新索引
    return {"pdfs": [p.name for p in pdfs], "chunks": len(chunks), "milvus_rows": n,  # 回执
            "collection": C.MILVUS_COLLECTION}                                      # 集合名

if __name__ == "__main__":                                                          # 允许命令行直接跑
    print(json.dumps(run(), ensure_ascii=False, indent=2))                          # 打印结果
