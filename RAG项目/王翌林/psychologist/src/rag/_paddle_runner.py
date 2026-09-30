"""PaddleOCR 调用脚本 - 在 WSL 的 .venv-paddle 中运行

用法:
  python _paddle_runner.py <pdf_or_image_path> [--lang ch] [--pages 1-5|all] [--out <output.md>] [--json] [--no-cache]

设计要点（内存安全）：
  逐页处理 —— 渲染一页 → OCR 一页 → 立即释放该页图像，内存占用与 PDF 总页数无关。
  结果按页增量写入 --out 文件并 flush，进程被超时杀死时已完成的页仍保留在磁盘上。

OCR 结果缓存：
  以「文件内容哈希 + 页码范围」为 key 缓存识别结果（data/ocr_cache/）。
  同一本书在不同角色目录下各存一份（路径不同但内容相同）时可直接复用，避免重复 OCR。
  只有完整跑完的文件才会写缓存；用 --no-cache 可跳过缓存。
"""
import gc
import hashlib
import json
import os
import shutil
import sys
import tempfile
import argparse

# 缓存目录：优先环境变量，其次项目根/data/ocr_cache（本文件位于 <root>/src/rag/）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.environ.get("PADDLE_OCR_CACHE_DIR") or os.path.join(_PROJECT_ROOT, "data", "ocr_cache")


def _file_digest(path: str) -> str:
    """文件内容 sha256（分块读取，避免一次性载入大文件）"""
    h = hashlib.sha256()
    # 用 iter(callable, sentinel) 每次只读 1MB：大 PDF 可能数百 MB，
    # 若 f.read() 一次性读入会把文件整体载进内存，与"省内存"目标相悖。
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _cache_file(file_path: str, pages_str: str) -> str:
    """计算缓存文件路径；内容或页码范围不同则不会命中。失败返回 None（不影响正常 OCR）"""
    try:
        key = f"{_file_digest(file_path)}|{pages_str}"
        return os.path.join(CACHE_DIR, hashlib.sha1(key.encode()).hexdigest()[:24] + ".md")
    except Exception:
        return None


def _save_cache(src_path: str, cache_path: str) -> bool:
    """写缓存（先写临时文件再原子替换，避免半截文件被当成有效缓存）。失败返回 False"""
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        # 先写"同目录"临时文件、再原子替换：同目录才能保证 os.replace 是同一
        # 文件系统内的原子操作（跨盘会退化为复制）。若直接写 cache_path，中途
        # 崩溃/断电会留下"半截文件"，下次会被误当成有效缓存读入。原子替换保证
        # 缓存文件要么完整、要么根本不存在。
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(cache_path), suffix=".tmp")
        os.close(fd)
        shutil.copyfile(src_path, tmp)
        os.replace(tmp, cache_path)
        return True
    except Exception as e:
        print(f"[paddle] 写缓存失败(忽略，不影响本次识别): {e}", file=sys.stderr, flush=True)
        return False


def _parse_pages(pages_str: str, total: int):
    """解析 '1-5,8' 这种页码字符串,返回 0-based 索引列表"""
    if not pages_str or pages_str.lower() == "all":
        return list(range(total))
    result = []
    for part in pages_str.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            result.extend(range(int(a) - 1, int(b)))
        else:
            result.append(int(part) - 1)
    return [i for i in result if 0 <= i < total]


def _extract_texts(result) -> list:
    """从 PaddleOCR 3.x 的返回结果中提取文字行"""
    texts = []
    for res in (result or []):
        if res is None:
            continue
        try:
            data = res.json
            if not isinstance(data, dict):
                data = json.loads(data)
            rec_texts = data.get("res", {}).get("rec_texts", []) or []
            texts.extend([t for t in rec_texts if t])
        except Exception as e:
            print(f"[paddle] parse result error: {e}", file=sys.stderr, flush=True)
    return texts


def _ocr_one_page(ocr, pdf, index: int) -> list:
    """渲染单页并 OCR。函数返回后该页图像内存由引用计数自动回收。"""
    import numpy as np

    page = pdf[index]
    try:
        width, _ = page.get_size()
        # 目标最长边 ~1500px，保证 OCR 精度的同时限制单页内存
        scale = min(1500.0 / width, 2.0) if width > 0 else 1.5
        bitmap = page.render(scale=scale)
        img = bitmap.to_pil()
        arr = np.asarray(img)
        return _extract_texts(ocr.predict(input=arr))
    finally:
        # 内存安全关键：无论成功还是抛异常，都在 finally 里显式 page.close()，
        # 及时释放该页图像缓冲；配合上层"逐页渲染"，峰值内存只与单页相关，
        # 与 PDF 总页数无关——这是能扛住 1000+ 页扫描件而不 OOM 的根本原因。
        page.close()


def run_paddle(file_path: str, lang: str = "ch", pages_str: str = "all",
               out_path: str = None, use_cache: bool = True) -> list:
    """对 PDF 或图像逐页执行 OCR

    out_path 指定时按页增量写入该文件；否则在内存中累积并返回（适合少量页测试）。
    """
    # ---- 缓存命中：直接复用，连模型都不用加载 ----
    cache_path = _cache_file(file_path, pages_str) if (use_cache and out_path) else None
    if cache_path and os.path.exists(cache_path):
        shutil.copyfile(cache_path, out_path)
        print(f"[paddle] 命中 OCR 缓存，跳过识别：{os.path.basename(cache_path)}", flush=True)
        return []

    from paddleocr import PaddleOCR

    print(f"[paddle] init model, lang={lang}", flush=True)
    # PaddleOCR 3.x 最小参数集：只保留语言，显式关掉「文档方向分类」「文档去扭曲」
    # 两个可选前置模型。前者对已经摆正的扫描页几乎无用，后者会额外加载模型、
    # 拖慢速度并多占显存；关掉后既保证识别精度，又显著降低初始化时间与内存占用。
    ocr = PaddleOCR(
        lang=lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )

    # 内存安全三板斧之二的落地：结果按页增量写盘（三是每页 flush，见 _emit）。
    # 与"逐页渲染"+"page.close()"配合，整条链路任一时刻都只持有单页数据。
    fout = open(out_path, "w", encoding="utf-8") if out_path else None
    pages = []
    completed = False

    try:
        if file_path.lower().endswith(".pdf"):
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(file_path)
            try:
                total = len(pdf)
                page_indices = _parse_pages(pages_str, total)
                print(f"[paddle] pdf total pages={total}, processing {len(page_indices)} pages",
                      flush=True)
                for n, idx in enumerate(page_indices, 1):
                    try:
                        texts = _ocr_one_page(ocr, pdf, idx)
                    except Exception as e:
                        print(f"[paddle] page {idx + 1} error: {e}", file=sys.stderr, flush=True)
                        texts = []
                    _emit(fout, pages, idx, texts)
                    if n % 50 == 0:
                        # 每 50 页手动触发一次 GC：长跑任务里 numpy/PIL/paddle 会
                        # 产生大量循环引用与小对象，Python 分代回收可能滞后，主动
                        # collect 能及时释放，避免内存随页数缓慢增长（页数越多越明显）。
                        gc.collect()  # 定期回收,防止长任务碎片累积
                completed = True
            finally:
                pdf.close()
        else:
            # 单张图像
            texts = _extract_texts(ocr.predict(input=file_path))
            _emit(fout, pages, 0, texts)
            completed = True
    finally:
        if fout:
            fout.close()

    # ---- 只有完整跑完才写缓存 ----
    # completed 表示"所有目标页都成功处理"；中途异常或被杀则为 False。
    # 若对半成品也写缓存，下次会直接命中并复用半份结果，导致丢页却无从察觉。
    if completed and cache_path and out_path:
        if _save_cache(out_path, cache_path):
            print(f"[paddle] 已写 OCR 缓存：{os.path.basename(cache_path)}", flush=True)

    return pages


def _emit(fout, pages: list, idx: int, texts: list) -> None:
    """输出单页结果：写文件（增量 flush）或累积到内存"""
    text = "\n".join(texts)
    if fout is not None:
        fout.write(f"## Page {idx + 1}\n\n{text}\n\n")
        fout.flush()
    else:
        pages.append({"page": idx + 1, "text": text})
    print(f"[paddle] page {idx + 1}: {len(texts)} lines", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path")
    parser.add_argument("--lang", default="ch")
    parser.add_argument("--pages", default="all", help="页码,如 '1-3,8' 或 'all'")
    parser.add_argument("--out", default=None, help="输出 markdown 文件路径")
    parser.add_argument("--no-cache", action="store_true", help="不使用 OCR 结果缓存")
    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"[paddle] file not found: {args.file_path}", file=sys.stderr)
        sys.exit(1)

    pages = run_paddle(args.file_path, args.lang, args.pages,
                       out_path=args.out, use_cache=not args.no_cache)

    if args.out:
        print(f"[paddle] written: {args.out}", flush=True)
    else:
        print(json.dumps(pages, ensure_ascii=False))


if __name__ == "__main__":
    main()