# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-功能测试及评估 —— MinerU 主解析路径冒烟验证
# 为什么单独跑：批量建索引时 MINERU_ENABLED=0（9 份共 2812 页，MinerU 耗时不可接受），
# 但「主路径真的能用」必须有实测证据。做法：把目标页切出来做成 ASCII 名的 PDF，再走**真实的** app.ingest.parse_mineru。
# 验收判据（可核验）：解析产物里必须出现标准答案的原文串「绿色贷款」与金额「2,433.01」。
import os, sys, time, json, shutil, pathlib, traceback                # 切片、计时、落盘
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # 项目根
from app import config as C                                            # 配置
from app.ingest import parse_mineru                                    # 直接调用生产代码路径，不另写一份
START, END = 55, 65                                                    # 0 起算页范围，覆盖标准答案所在的第 61 页

def main():                                                            # 主流程
    C.RESULTS.mkdir(exist_ok=True)                                     # 结果目录
    pdf = next(f for f in (C.CORPUS_DIR / "pdf").iterdir()             # 找到邮储银行那份 PDF
               if f.suffix.lower() == ".pdf" and "601658" in f.name)
    import fitz                                                        # 本机用 pymupdf 做切片（不参与被验证的解析路径）
    doc = fitz.open(pdf); slice_pdf = C.DATA / "src" / "2020-03-26__slice__601658__slice__2019__slice.pdf"  # 切片目标：文件名带股票代码，便于 doc_meta 回认
    slice_pdf.parent.mkdir(parents=True, exist_ok=True)                # 目录就绪
    doc.select(list(range(START, END + 1)))                            # 只保留目标页范围（本版 pymupdf 的 select 原地修改，无返回值）
    doc.save(slice_pdf)                                                # 存成 ASCII 名的小 PDF，成本有界
    print(f"切片完成：{slice_pdf.name}（{END - START + 1} 页）")          # 提示
    res = {"input": str(pdf), "slice_pages": f"{START}-{END}", "ok": False}  # 结果骨架
    try:                                                               # 解析失败也要留下证据
        t0 = time.time()                                               # 计时
        blocks, parser = parse_mineru(str(slice_pdf)), None            # 调生产代码；parse_mineru 只返回 blocks 或 None
        res["elapsed_s"] = round(time.time() - t0, 1)                  # 耗时
        res["parser"] = "parse_mineru"                                 # 走的就是主路径
        res["pages_parsed"] = len(blocks) if blocks else 0             # 解析出的页数
        text = "\n".join(t for _, t in (blocks or []))                 # 汇总全文
        res["verify"] = {"绿色贷款": "绿色贷款" in text,                 # 关键串①：正文
                         "2433.01": "2,433.01" in text.replace(" ", "")}  # 关键串②：金额（去空格后比对，PDF 抽取会在数字间插空格）
        res["sample"] = text[:300]                                     # 留一段原文便于人工核对
        res["ok"] = bool(blocks) and all(res["verify"].values())        # 验收判据
    except Exception as e:                                             # 记录失败原因，不吞
        res["error"] = f"{type(e).__name__}: {e}"; res["trace"] = traceback.format_exc()[-800:]
    (C.RESULTS / "mineru_smoke.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")  # 落盘
    print(json.dumps({k: v for k, v in res.items() if k != "trace"}, ensure_ascii=False, indent=1))  # 打印
    return 0 if res["ok"] else 1                                       # 不通过则非 0 退出

if __name__ == "__main__":                                             # 直接运行
    sys.exit(main())
