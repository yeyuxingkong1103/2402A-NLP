# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-功能测试及评估 —— 离线入库：ZIP/PDF → 四级解析 → 语义分块 → Milvus + BM25
import os, re, sys, json, pickle, subprocess, shutil; from collections import Counter  # 路径/正则/子进程调 MinerU/落盘/复制/行频统计
import numpy as np, jieba                                                     # 向量运算 / 中文分词（BM25 用）
from rank_bm25 import BM25Okapi                                               # 关键词检索算法
from . import config as C                                                     # 本工单配置
jieba.setLogLevel(60)                                                         # 关掉 jieba 启动日志，避免刷屏
_LF = "C:/mineru_models/langfix"                                              # MinerU 语言识别模型缓存目录
def doc_meta(path):                                                           # 从（已损坏的）文件名回认文档身份
    b = os.path.basename(path); m, ym = re.search(r"__(\d{6})__", b), re.search(r"^(\d{4})-\d{2}-\d{2}", b)  # 文件名、股票代码（唯一幸存且稳定的标识）、披露日期前缀
    code = m.group(1) if m else "unknown"                                     # 取不到代码则标记 unknown
    year = str(int(ym.group(1)) - 1) if ym else "?"                            # 3 月前披露的年报 → 报告年度 = 披露年 - 1
    return code, C.CODE2NAME.get(code, code), year                             # 返回 代码 / 公司名 / 报告年度
def parse_mineru(pdf):                                                        # ① 主解析器：MinerU（版面分析，能读表格与图内文字）
    out = C.DATA / "parsed"                                                   # 解析产物根目录
    try:                                                                      # MinerU 依赖重，任何异常都必须能降级
        env = dict(os.environ, MINERU_LANGFIX_DIR=_LF)                        # 复制环境变量并指定语言模型目录
        env["PYTHONPATH"] = _LF + os.pathsep + env.get("PYTHONPATH", "")       # 只设变量无效，须挂 PYTHONPATH 才加载 sitecustomize
        safe = C.DATA / "src" / (doc_meta(str(pdf))[0] + ".pdf"); safe.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(pdf, safe)  # 实测：盘上中文名已被 U+FFFD 破坏，MinerU 建临时目录时会二次乱码导致找不到产物，必须先复制成纯 ASCII 名
        subprocess.run([sys.executable, "-m", "mineru.cli.client", "-p", str(safe), "-o", str(out)], env=env, timeout=7200, capture_output=True, check=True)  # 子进程跑官方 CLI，喂 ASCII 副本
        stem = safe.stem                                                       # 产物目录名跟随输入文件名，用 ASCII 名匹配最稳
        for h in out.rglob("*content_list.json"):                              # 产物目录布局新旧不一，用 rglob 找，不能写死路径
            if stem not in str(h):                                             # 不是本 PDF 的产物
                continue                                                       # 跳过
            blocks = {}                                                        # 以页号聚合
            for it in json.loads(h.read_text(encoding="utf-8")):                # MinerU 的块级结构化输出（每项一个版面块）
                t = "\n".join(it.get("table_body") or []) if it.get("type") == "table" else (it.get("text") or "")  # 表格在 table_body 而不在 text
                if t.strip():                                                   # 空块不要
                    blocks.setdefault(int(it.get("page_idx", 0)) + 1, []).append(t.strip())  # page_idx 从 0 起，转 1 起
            if blocks:                                                          # 有内容才算成功
                return [(p, "\n".join(v)) for p, v in sorted(blocks.items())]     # 返回 [(页码, 文本)]
    except Exception as e:                                                     # 降级不静默：打印失败类型便于追因
        print(f"[parse] MinerU 失败，降级：{type(e).__name__}")                   # 只报类型，避免超长栈刷屏
    return None                                                                # 交给下一级
def parse_pymupdf(pdf):                                                        # ② 备用解析器：pymupdf（本语料文本层完好，快且稳）
    try:                                                                       # 依赖缺失/文件损坏都要能降级
        import fitz                                                            # 延迟导入，避免与 MinerU 的依赖打架
        d = fitz.open(pdf)                                                     # 打开 PDF
        r = [(i + 1, (p.get_text() or "").strip()) for i, p in enumerate(d)]    # 逐页取文本，页码 1 起
        d.close()                                                              # 及时释放句柄
        return [x for x in r if x[1]] or None                                   # 全空则视为失败
    except Exception as e:                                                      # 同上，继续降级
        print(f"[parse] pymupdf 失败，降级：{type(e).__name__}")                  # 打印原因
        return None                                                             # 交给下一级
def parse_txt(pdf):                                                            # ③ 兜底解析器：用附件自带的同名 txt
    p = C.CORPUS_DIR / "txt" / (os.path.splitext(os.path.basename(pdf))[0] + ".txt")  # 按同名 txt 找
    try: return [(1, p.read_text(encoding="utf-8"))] if p.exists() else None     # 无页信息，整篇记为第 1 页
    except Exception: return None                                              # 不存在/编码/权限异常则放弃
def parse_pdf(pdf):                                                            # 四级降级入口（工单要求：MinerU 为主 + 备用方案）
    chain = (parse_mineru, parse_pymupdf, parse_txt) if os.getenv("MINERU_ENABLED", "1") == "1" else (parse_pymupdf, parse_txt)  # MinerU 页面多时很慢，可用 MINERU_ENABLED=0 临时关闭
    for fn in chain:                                                            # 依次尝试，先重后轻
        blocks = fn(pdf)                                                        # 调用当前解析器
        if not blocks: continue                                                 # 本级没拿到内容则继续降级
        hd = lambda s: re.sub(r"\s+", "", s)                                   # 行特征：只去空白（PDF 抽取会在中文间插空格，而页眉页脚逐页逐字相同）
        cnt = Counter(k for k in map(hd, [l for _, t in blocks for l in t.splitlines()[:4] + t.splitlines()[-4:]]) if 8 <= len(k) <= 40)  # 只统计每页头尾各 4 行：页眉页脚必在页边，而"合计""不适用"等高频词只在表格中间
        bad = {k for k, v in cnt.items() if v >= max(8, len(blocks) * 0.25)}     # 出现在 ≥25% 页面页边的行即页眉/页脚模板（实测页眉会被拆成多条短行，故不能要求整行匹配）
        blocks = [(p, "\n".join(l for l in t.splitlines() if hd(l) not in bad)) for p, t in blocks]  # 逐页剔掉模板行，防模板噪声淹没正文
        print(f"[parse] {os.path.basename(pdf)[:26]} → {fn.__name__} {len(blocks)} 页（剔模板行 {len(bad)} 条）")  # 记录生效解析器与清洗量
        return blocks, fn.__name__                                              # 一并返回解析器名，写进结果供追溯
    raise RuntimeError(f"三种解析器均失败：{pdf}")                                 # 全失败才抛错
def sem_chunk(text, encode):                                                    # 语义分块：按与当前块语义中心的相似度漂移切块
    sents = []                                                                  # 断句结果
    for s in re.split(r"(?<=[。！？；!?;\n])", text):                             # 在句末标点/换行后切开
        s = s.strip()                                                           # 去空白
        while len(s) > 200:                                                     # 超长句（表格行等）按长度硬切
            sents.append(s[:200]); s = s[200:]                                   # 切出定长片段
        sents += [s] if s else []                                               # 非空才收
    if not sents: return []                                                    # 空文本直接返回
    embs = encode(sents)                                                        # 批量编码（逐句编码 7 万句会慢一个数量级）
    chunks, cur, vecs = [], [], []                                              # 已完成块 / 当前块文本 / 当前块向量集合
    for s, v in zip(sents, embs):                                                # 逐句累积
        if cur:                                                                 # 已有内容才算相似度
            c = np.mean(vecs, axis=0); c = c / (np.linalg.norm(c) + 1e-9)        # 当前块语义中心
            sim = float(np.dot(v / (np.linalg.norm(v) + 1e-9), c))               # 与中心的余弦相似度
            drift = sim < C.SEM_THRESHOLD and len("".join(cur)) + len(s) > C.CHUNK_SIZE  # 语义漂移且已够长
            if drift or len("".join(cur)) > C.SEM_MAX_CHUNK:                     # 触发切块，或到硬上限
                chunks.append("".join(cur))                                     # 收下当前块
                cur, vecs = ["".join(cur)[-C.CHUNK_OVERLAP:]], []                # 新块以尾部重叠开头，防答案被切断
        cur.append(s); vecs.append(v)                                            # 累积当前句
    if cur:                                                                     # 收尾
        chunks.append("".join(cur))                                             # 最后一并收下
    return [c for c in chunks if len(c.strip()) > 30]                           # 丢掉过短的噪声块
def build():                                                                    # 入库主流程
    from .rag_core import embed, milvus                                          # 延迟导入，避免循环依赖
    pdfs = sorted(p for p in (C.CORPUS_DIR / "pdf").iterdir() if p.suffix.lower() == ".pdf")  # 取全部 PDF
    cli = milvus(); cli.drop_collection(C.MILVUS_COLLECTION)                     # 重建集合，保证可重复执行
    cli.create_collection(C.MILVUS_COLLECTION, dimension=C.EMBED_DIM, metric_type="COSINE", auto_id=True, enable_dynamic_field=True)  # 动态字段：正文/公司/页号随向量一起存
    recs = []                                                                   # 全部分块记录
    for pdf in pdfs:                                                            # 逐份 PDF
        code, name, year = doc_meta(str(pdf))                                    # 回认文档身份
        for page, txt in (bl := parse_pdf(str(pdf)))[0]:                         # 解析（含降级）后逐页分块，bl[1] 是实际生效的解析器名
            recs += [dict(text=ck, company=name, code=code, year=year, page=page, parser=bl[1]) for ck in sem_chunk(txt, embed)]  # 一页的语义分块，记录解析器供追溯
    vecs = embed([r["text"] for r in recs])                                      # 批量向量化（bge-m3）
    for i in range(0, len(recs), 256):                                            # 分批写入，单次太大易超时
        cli.insert(C.MILVUS_COLLECTION, [dict(r, vector=v.tolist()) for r, v in zip(recs[i:i + 256], vecs[i:i + 256])])  # 写 Milvus
    toks = [[w for w in jieba.lcut(r["text"]) if w.strip()] for r in recs]       # BM25 语料：jieba 分词
    (C.DATA / "bm25.pkl").write_bytes(pickle.dumps({"bm25": BM25Okapi(toks), "recs": recs}))  # BM25 索引与分块表一并落盘
    (C.DATA / "index.json").write_text(json.dumps(recs, ensure_ascii=False), encoding="utf-8")  # 纯文本分块表，便于人工排查
    print(f"[ingest] 完成：{len(recs)} 块 / {len(pdfs)} 份文档"); return len(recs)  # 汇总并返回块数
