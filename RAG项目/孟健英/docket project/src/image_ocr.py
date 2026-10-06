# -*- coding: utf-8 -*-
"""图片 OCR：PaddleOCR 单例懒加载，未装包时降级返回空串。"""
import logging  # 日志：记录模型加载与识别耗时
import os  # 改 paddle 环境变量用
import tempfile  # 落临时文件供 paddle 读取
import time  # 各阶段计时
from pathlib import Path  # 操作模型缓存目录与临时文件

logger = logging.getLogger(__name__)  # 本模块日志器
os.environ.setdefault("FLAGS_use_mkldnn", "false")  # 规避 paddle 3.4.x oneDNN PIR 崩溃

_OCR = None  # 单例：进程内只加载一次
_TRIED = False  # 已尝试标记：失败不重试，避免每次上传都卡 22 秒


def ocr_available() -> bool:  # 探测 OCR 是否可用
    """是否安装了 paddleocr。"""
    try:  # 仅试探导入
        import paddleocr  # noqa: F401
        return True  # 装了就可用
    except ImportError:  # 没装 paddleocr
        return False  # 降级：UI 不显示上传区


def _cache_size_mb() -> float:  # 模型缓存目录大小探测
    """OCR 模型缓存目录大小（~/.paddlex），用于判断是否需联网下载。"""
    d = Path.home() / ".paddlex"  # paddle 模型默认缓存在用户目录
    if not d.exists():  # 无缓存目录
        return 0.0  # 视为 0，意味着首次需联网下载
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6  # 递归求和换算成 MB


def _get_ocr():  # 单例加载入口
    global _OCR, _TRIED  # 改模块级单例
    if _OCR is None and not _TRIED:  # 未加载且没试过才进
        _TRIED = True  # 先打标记：即使失败也不再重复尝试
        try:  # 加载可能失败（缺依赖/下载失败）
            t0 = time.perf_counter()  # 计时起点
            cache_mb = _cache_size_mb()  # 加载前记录缓存大小，用于判断是否首次下载
            from paddleocr import PaddleOCR  # 延迟导入：光 import 就要几秒
            t1 = time.perf_counter()  # import 与初始化耗时的分界点
            _OCR = PaddleOCR(lang="ch", enable_mkldnn=False)  # 中文模型；关 mkldnn 规避崩溃
            t2 = time.perf_counter()  # 初始化完成时刻
            logger.info("PaddleOCR 加载完成：import=%.1fs 初始化=%.1fs（初始化前模型缓存 %.0fMB，缓存 %.0fMB）",  # 两段耗时与缓存变化写日志
                        t1 - t0, t2 - t1, cache_mb, _cache_size_mb())
        except Exception as e:  # 加载失败
            logger.warning("PaddleOCR 加载失败：%s", e)  # 告警但不抛异常：OCR 整体降级
    return _OCR  # 失败时返回 None，调用方判空


def _shrink(img_bytes: bytes, max_side: int = 1200) -> bytes:  # 图片压缩入口：长边超限才压
    """长边超过 max_side 的图先等比压缩再 OCR：大图是 CPU 推理慢的主因（2661px 实测 67s）。"""
    import io  # 内存字节流
    from PIL import Image  # Pillow 处理图片
    im = Image.open(io.BytesIO(img_bytes))  # 从字节解码图片
    if max(im.size) <= max_side:  # 长边不超阈值
        return img_bytes  # 原样返回，省一次重编码
    t0 = time.perf_counter()  # 压缩计时
    w, h = im.size  # 原图尺寸
    scale = max_side / max(w, h)  # 等比缩放系数
    im = im.convert("RGB").resize((int(w * scale), int(h * scale)), Image.LANCZOS)  # 转 RGB 后 LANCZOS 高质量缩放，兼顾识别精度
    buf = io.BytesIO()  # 输出缓冲
    im.save(buf, "JPEG", quality=90)  # JPEG 质量 90：体积与清晰度的平衡点
    logger.info("图片压缩：%dx%d → %s（%.1fs）", w, h, im.size, time.perf_counter() - t0)  # 压缩前后尺寸与耗时写日志
    return buf.getvalue()  # 返回压缩后的字节


def ocr_image(img_bytes: bytes) -> str:  # 对外识别入口
    """识别图片字节，返回拼接后的纯文本。失败返回空串。"""
    ocr = _get_ocr()  # 取单例，未加载则此时加载
    if ocr is None:  # 加载失败
        return ""  # 降级返回空串，不阻塞提问主流程
    try:  # 压缩也可能失败（坏图/格式异常）
        img_bytes = _shrink(img_bytes)  # 长边超 1200px 先压缩
    except Exception as e:  # 压缩失败
        logger.warning("图片压缩失败，按原图识别：%s", e)  # 退回原图继续识别
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:  # paddle 只收文件路径，先落临时文件；delete=False 防 Windows 下文件占用
        f.write(img_bytes)  # 写入图片字节
        path = f.name  # 记下路径待清理
    try:  # 识别主流程
        t0 = time.perf_counter()  # 推理计时起点
        res = ocr.predict(path) if hasattr(ocr, "predict") else ocr.ocr(path, cls=True)  # 兼容 paddleocr 3.x 与 2.x 两套 API
        t1 = time.perf_counter()  # 推理结束时刻
        lines = []  # 收集文本行
        if res and isinstance(res[0], dict):  # paddleocr 3.x
            for pg in res:  # 逐页取结果
                lines += list(pg.get("rec_texts", []))  # 3.x 按 rec_texts 字段取文本
        else:  # 2.x
            for ln in (res[0] if res else []):  # 2.x 结构逐行取
                lines.append(ln[1][0])  # 取文本部分，丢掉置信度
        lines = [t.strip() for t in lines if len(t.strip()) >= 2]  # 去空白并丢弃单字噪声行；过滤单字噪声行（如 "R"、"H" 处方符号），保留有意义文本
        logger.info("OCR 识别完成：推理=%.1fs 解析=%.1fs（共 %d 行）",  # 推理/解析耗时与行数写日志
                    t1 - t0, time.perf_counter() - t1, len(lines))
        return "\n".join(lines)  # 按行拼成纯文本
    except Exception as e:  # 识别过程异常
        logger.exception("OCR 识别失败：%s", e)  # 带完整堆栈告警
        return ""  # 降级返回空串
    finally:  # 无论成败都清理
        Path(path).unlink(missing_ok=True)  # 删临时文件；missing_ok 防文件已不存在时报错
