# -*- coding: utf-8 -*-
"""PDF解析：页眉页脚剥离+水印过滤+表格提取，PyMuPDF dict API。"""
import logging  # 日志：解析完成后输出页眉/页脚/水印/表格的清洗统计
from collections import defaultdict  # 计数专用字典：统计每行文本在多少页重复出现，默认值 0 省去判空
import fitz  # PyMuPDF 的导入名：读取 PDF 的文本块、坐标、颜色、表格等版面信息

logger = logging.getLogger(__name__)  # 本模块日志器

def _detect_hf(doc, sim=0.8):  # 检测页眉页脚：在 ≥80% 页面的同一区域重复出现的行才判定为页眉/页脚
    h, f, n = defaultdict(int), defaultdict(int), len(doc)  # h=页眉候选计数，f=页脚候选计数，n=总页数
    for pg in doc:  # 逐页扫描
        ph = pg.rect.height  # 页面高度，用来算每一行的纵向位置占比
        try: blocks = pg.get_text("dict")["blocks"]  # dict 模式取结构化文本：块→行→span 三级，带坐标等元数据
        except: continue  # 单页解析失败就跳过，不让一页坏稿毁掉整个文档
        for b in blocks:  # 遍历文本块
            if b.get("type") != 0: continue  # type=0 才是文本块，图片块等跳过
            for ln in b.get("lines", []):  # 遍历块内每一行
                t = "".join(s["text"] for s in ln.get("spans", [])).strip()  # 一行可能由多个 span（字体不同的片段）组成，拼成整行
                if not t: continue  # 空行跳过
                y = ln["spans"][0]["bbox"][1]  # 取行首 span 的纵坐标（bbox=[x0,y0,x1,y1]），判断该行在页面顶部还是底部
                (h if y < ph * 0.15 else f if y > ph * 0.85 else None)  # 位置分区：顶部 15% 算页眉区，底部 15% 算页脚区
                if y < ph * 0.15: h[t] += 1  # 落在页眉区，该文本的页眉计数 +1
                elif y > ph * 0.85: f[t] += 1  # 落在页脚区，该文本的页脚计数 +1
    thr = int(n * sim)  # 阈值：至少在 80% 的页面重复出现才算页眉/页脚（偶发行不算）
    return {t for t, c in h.items() if c >= thr}, {t for t, c in f.items() if c >= thr}  # 返回页眉集合与页脚集合，供正文过滤用

def _detect_wm(doc):  # 检测水印：颜色浅（接近白）或明显旋转、且跨页高频出现的文字
    wm, n = defaultdict(int), len(doc)  # 水印候选计数、总页数
    for pg in doc:  # 逐页扫描
        try: blocks = pg.get_text("dict")["blocks"]  # 同样用 dict 模式拿结构化文本
        except: continue  # 坏页跳过
        for b in blocks:
            if b.get("type") != 0: continue  # 只看文本块
            for ln in b.get("lines", []):
                for sp in ln.get("spans", []):  # 水印判定要精确到 span 级：颜色和旋转角度都是 span 的属性
                    t = sp["text"].strip()
                    if not t or len(t) < 2: continue  # 单字符噪声不算水印
                    c = sp.get("color", 0)  # 文本颜色，整数打包的 RGB
                    r, g, b = (c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF  # 位运算拆出 R/G/B 三个通道
                    if (r + g + b) / 3 > 200 or abs(sp.get("rotation", 0)) > 5:  # 灰度均值>200 即浅色字（典型灰色水印），或旋转超过 5° 的斜排字
                        wm[t] += 1  # 命中水印特征，计数 +1
    thr = max(2, int(n * 0.5))  # 至少在一半页面出现（且至少 2 次）才认定为水印
    return {t for t, c in wm.items() if c >= thr}  # 返回水印文本集合

def _extract_tables(pg):  # 提取当页表格并转成 Markdown（纯文本解析最容易丢表格结构，单独抢救）
    md = []  # 本页表格的 Markdown 文本列表
    for tbl in pg.find_tables().tables:  # PyMuPDF 表格识别：基于线条与空白对齐检测表格区域
        try:
            data = tbl.extract()  # 提取为二维数组，每行一个 list
            if data:
                lines = ["| " + " | ".join(str(c).strip() if c else "" for c in row) + " |" +  # 每行拼成 Markdown 表格行：| 单元格 | 单元格 |
                         ("\n| " + " | ".join("---" for _ in row) + " |" if i == 0 else "")  # 首行之后补 |---| 分隔行，Markdown 表格语法要求
                         for i, row in enumerate(data)]
                md.append("\n".join(lines))  # 一个表格拼成一段 Markdown
        except: pass  # 单个表格提取失败就跳过，不影响其余表格
    return md

def parse_pdf(path):  # 解析主函数：整本 PDF → 带 [第N页] 标记的纯文本 + 清洗统计
    doc = fitz.open(str(path))  # 打开 PDF（fitz 要求 str 类型路径）
    stats = {"headers": 0, "footers": 0, "watermarks": 0, "tables": 0}  # 统计清洗掉的页眉/页脚/水印与提取到的表格数
    try:  # try/finally 保证文档一定被关闭，防文件句柄泄漏
        hs, fs = _detect_hf(doc); ws = _detect_wm(doc)  # 先全篇预扫一遍，拿到页眉/页脚/水印三份黑名单
        stats.update(headers=len(hs), footers=len(fs), watermarks=len(ws))  # 黑名单规模记入统计
        pages = []  # 每页产出一段带页码标记的文本
        for i, pg in enumerate(doc):  # 逐页提取正文
            tabs = _extract_tables(pg); stats["tables"] += len(tabs)  # 先单独提表格：表格区域随后会被当普通文本再读一遍但结构已丢，故另存 Markdown
            try:
                blocks = pg.get_text("dict")["blocks"]  # dict 模式拿带坐标的结构化文本
                lines = []  # 本页清洗后的正文行
                for b in blocks:
                    if b.get("type") != 0: continue  # 只要文本块
                    for ln in b.get("lines", []):
                        t = "".join(s["text"] for s in ln.get("spans", [])).strip()  # span 拼成整行
                        if t and t not in hs and t not in fs and t not in ws:  # 三份黑名单过滤：页眉/页脚/水印不进正文——脏数据入库会拉低检索质量
                            lines.append(t)
                pt = "\n".join(lines)  # 本页正文拼成一段
            except: pt = pg.get_text()  # dict 模式失败则降级为纯文本提取：结构丢了但内容保住
            if pt or tabs:  # 本页有正文或有表格才记录
                content = f"[第{i+1}页]\n" + pt  # 写入页码标记：后续切页/切块/引用定位全靠它
                if tabs: content += "\n\n【表格】\n" + "\n\n".join(tabs)  # 表格以 Markdown 附在页尾，行列结构保留给 LLM 看
                pages.append(content)
        return "\n\n".join(pages), stats  # 各页之间空行分隔，拼成整篇返回
    finally: doc.close()  # 无论成功失败都关闭文档

def clean_text(text):  # 二次清洗：删孤立数字行（残留页码）、压缩连续空行
    cl, pe = [], False  # cl=清洗后的行列表；pe=上一行是否为空行的标记
    for line in text.split("\n"):  # 逐行处理
        line = line.strip()  # 去行首尾空白
        if line.isdigit(): continue  # 纯数字行多为残留页码，删掉
        if not line:  # 空行处理：连续空行只保留一个
            if not pe: cl.append("")  # 前一行不是空行才保留这个空行
            pe = True  # 标记已进入空行段
        else: pe = False; cl.append(line)  # 非空行：清标记并保留该行
    return "\n".join(cl).strip()  # 重新拼成整篇并去首尾空白

def parse_and_clean(path):  # 对外入口：解析 + 清洗一步完成
    raw, stats = parse_pdf(path)  # 先拿带页码标记的原始文本与统计
    logger.info("解析完成 页眉%d/页脚%d/水印%d/表格%d",  # 清洗统计写日志，便于评估解析质量
                stats["headers"], stats["footers"], stats["watermarks"], stats["tables"])
    return clean_text(raw)  # 返回清洗后的文本