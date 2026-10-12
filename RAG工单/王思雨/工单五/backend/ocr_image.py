# 工单编号：人工智能NLP-RAG-Query理解优化
# 文件说明：图像文字提取层。用 PaddleOCR 把招股说明书里的图（组织结构图、流程图、图表）转成可检索文本，
#           同时承担扫描件的整页 OCR 兜底。所有 OCR 结果都要过一遍水印过滤，避免噪声进入知识库。
import os                                                                       # 读取环境变量（Paddle 需要关掉联网检查）
from typing import List, Dict, Tuple                                            # 类型标注

from . import config as C                                                       # 复用集中配置
from .text_clean import clean_text, is_noisy                                    # 复用统一清洗与污染判定

_OCR = None                                                                     # PaddleOCR 单例，加载一次常驻
# 坑：用户名含中文时，Paddle 的 C++ 推理层打不开默认缓存目录 ~/.paddleocr/，
# 报 "Cannot open file ... pdmodel" 但文件其实存在。故模型另存到纯英文路径并显式传入。
os.environ.setdefault("FLAGS_use_mkldnn", "0")                                  # 关闭 mkldnn，规避部分 CPU 下的算子兼容问题
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")          # 关闭联网模型源检查，避免离线环境卡住


def ocr_engine():                                                               # 懒加载 PaddleOCR
    """首次调用加载检测/识别/方向模型并常驻；显式指向纯英文模型目录以绕开中文用户名。"""
    global _OCR                                                                 # 声明单例
    if _OCR is None:                                                            # 尚未加载
        # 坑（实测，必读）：Windows 上"先导入 paddle/paddleocr、后导入 torch"会直接崩：
        #   OSError: [WinError 127] 找不到指定的程序。 Error loading "...\torch\lib\shm.dll"
        # 原因是两者自带的 MKL/OpenMP 运行时 DLL 冲突，且顺序敏感。反过来（先 torch 后 paddle）正常，
        # 所以这里强制先把 torch 拉起来，再导入 paddleocr。缺了这两行，入库到"分块"阶段必崩。
        try:                                                                    # torch 缺失时不该阻断 OCR
            import torch  # noqa: F401                                          # 仅为占位加载，不直接使用
        except Exception as exc:                                                # 没装 torch
            print(f"[OCR] 预加载 torch 失败（不影响 OCR，但后续向量化会不可用）：{exc}")  # 只告警
        from paddleocr import PaddleOCR                                         # 延迟导入，缩短服务冷启动
        det = os.path.join(C.PADDLEOCR_MODEL_DIR, "det")                        # 检测模型目录
        rec = os.path.join(C.PADDLEOCR_MODEL_DIR, "rec")                        # 识别模型目录
        cls = os.path.join(C.PADDLEOCR_MODEL_DIR, "cls")                        # 方向分类模型目录
        kw = {"use_angle_cls": True, "lang": "ch", "show_log": False}           # 打开方向分类，竖排文字才能正确读出
        if os.path.isdir(det): kw["det_model_dir"] = det                        # 目录存在才传，否则用默认
        if os.path.isdir(rec): kw["rec_model_dir"] = rec                        # 同上
        if os.path.isdir(cls): kw["cls_model_dir"] = cls                        # 同上
        _OCR = PaddleOCR(**kw)                                                  # 实例化并常驻
    return _OCR                                                                 # 返回单例


def _lines_from_result(res) -> List[Tuple[str, float]]:                         # 解析 PaddleOCR 返回结构
    """把 PaddleOCR 的嵌套结构压成 [(文本, 置信度)]；识别为空时返回空列表而不是抛错。"""
    out: List[Tuple[str, float]] = []                                           # 结果容器
    for page in (res or []):                                                    # PaddleOCR 按页返回
        for item in (page or []):                                               # 每个检测框
            try:                                                                # 结构异常时跳过而不是崩
                txt, conf = item[1][0], float(item[1][1])                       # 文本与置信度
            except Exception:                                                   # 结构不符预期
                continue                                                        # 跳过该框
            out.append((str(txt), conf))                                        # 收集
    return out                                                                  # 返回


def ocr_text(image_path: str, min_conf: float = 0.5) -> str:                    # 对整图做 OCR 并清洗
    """整图 OCR：过滤低置信与水印行，按行返回文本。用于普通插图与扫描页。"""
    eng = ocr_engine()                                                          # 取引擎
    res = eng.ocr(image_path, cls=True)                                         # 识别
    keep = []                                                                   # 保留的行
    for txt, conf in _lines_from_result(res):                                   # 逐行过滤
        t = clean_text(txt)                                                     # 先清洗水印
        if conf >= min_conf and len(t) > 1 and not is_noisy(t):                 # 低置信、单字、水印一律丢弃
            keep.append(t)                                                      # 保留
    return "\n".join(keep)                                                      # 汇总


def ocr_box(image_path: str, box: Tuple[int, int, int, int]) -> str:            # 对指定矩形区域做 OCR
    """裁出矩形区域后识别；用于流程图里"每个方框一个标签"的场景。

    实测：内缩 3~4 像素会把窄框（75px 宽）的笔画切掉，OCR 退化成单字（"财务部"→"部"）；
    内缩 2 像素、并且同时试 0°/90°/270° 三种方向后取"字符数最多、置信度最高"的结果才稳定。
    """
    from PIL import Image                                                       # 延迟导入
    import numpy as np                                                          # 延迟导入
    x, y, w, h = box                                                            # 拆解矩形
    im = Image.open(image_path).convert("RGB")                                  # 打开并转 RGB
    best_text, best_key = "", (-1, -1.0)                                        # 最优结果及其排序键（字符数, 平均置信度）
    for margin in (2, 5):                                                       # 内缩量：2 优先，5 作为兜底再试一次
        crop = im.crop((x + margin, y + margin, x + w - margin, y + h - margin))  # 裁出区域
        if crop.width < 12 or crop.height < 12:                                 # 框太小，裁完没内容
            continue                                                            # 跳过该内缩量
        for rot in (None, Image.ROTATE_90, Image.ROTATE_270):                   # 竖排/横排混排，三个方向都试
            img = crop.rotate(rot, expand=True) if rot is not None else crop    # 旋转副本（None 即原方向）
            res = ocr_engine().ocr(np.array(img), cls=True)                     # 识别该变体
            parts = [(clean_text(t), c) for t, c in _lines_from_result(res)]    # 清洗后的 (文本, 置信度)
            parts = [(t, c) for t, c in parts if t]                             # 丢掉空文本
            if not parts:                                                       # 本变体没读出东西
                continue                                                        # 换下一个变体
            text = "".join(t for t, _ in parts)                                 # 多行拼接（竖排常被切成两段）
            conf = sum(c for _, c in parts) / len(parts)                        # 平均置信度
            key = (len(text), conf)                                             # 先比字符数再比置信度
            if conf >= 0.5 and key > best_key:                                  # 更优且可信
                best_text, best_key = text, key                                 # 更新最优
    return best_text                                                            # 返回最优文本（都不达标时为空串）


def ocr_pdf_page(pdf_path: str, page_no: int) -> str:                           # 扫描页整页 OCR
    """把指定页按 OCR_SCALE 渲染成位图后 OCR；1x 会把图表里的负号读丢，故默认放大 3 倍。"""
    import pymupdf                                                              # 延迟导入
    import numpy as np                                                          # 延迟导入
    from PIL import Image                                                       # 延迟导入
    doc = pymupdf.open(pdf_path)                                                # 打开 PDF
    try:                                                                        # 保证句柄一定关闭
        page = doc[page_no - 1]                                                 # pymupdf 页码 0 起
        mat = pymupdf.Matrix(C.OCR_SCALE, C.OCR_SCALE)                          # 放大矩阵
        pix = page.get_pixmap(matrix=mat)                                       # 渲染为位图
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)      # 转 PIL 图像
        res = ocr_engine().ocr(np.array(img), cls=True)                         # 整页识别
        keep = [clean_text(t) for t, c in _lines_from_result(res)               # 过滤低置信与水印
                if c >= 0.5 and len(clean_text(t)) > 1 and not is_noisy(clean_text(t))]
        return "\n".join(keep)                                                  # 汇总为文本
    finally:                                                                    # 无论如何
        doc.close()                                                             # 关闭文档


def enrich_scanned_blocks(blocks: List[Dict], pdf_path) -> List[Dict]:           # 为扫描页占位块补文本
    """解析降级到 pymupdf/pdfplumber 时，扫描页会留下空文本块；这里用 OCR 把它们补齐。"""
    fixed = 0                                                                   # 统计补齐页数
    for b in blocks:                                                            # 遍历块
        if b.get("block_type") != "scanned" or b.get("text"):                   # 只看空文本的扫描页
            continue                                                            # 跳过
        try:                                                                    # OCR 可能失败（缺模型等）
            b["text"] = ocr_pdf_page(str(pdf_path), int(b["page_no"]))          # 整页 OCR
            b["block_type"] = "ocr"                                             # 标记来源为 OCR，便于溯源
            fixed += 1                                                          # 计数
        except Exception as exc:                                                # 失败不阻断整体解析
            print(f"[OCR] 第 {b.get('page_no')} 页 OCR 失败：{type(exc).__name__}: {exc}")  # 明确报因
    if fixed:                                                                   # 有补齐
        print(f"[OCR] {pdf_path}：扫描页 OCR 补齐 {fixed} 页")                    # 记录
    return blocks                                                               # 返回原列表（原地修改）
