# -*- coding: utf-8 -*-
"""OCR 图片文字识别调用模块（PaddleOCR-VL）。

封装 PaddleOCR-VL 的识别流程，把「图片 → 文字 → 保存为 txt」串成一次调用，
方便在 RAG 流程中对扫描件 / 截图 / 图片型 PDF 做文字提取后入库。

实现要点：

1. **懒加载**：``paddleocr`` 体积大、仅安装在 ``ocr_`` 环境，本模块顶层不 import，
   首次真正识别时才加载模型，导入本模块本身零开销；
2. **取文本**：PaddleOCR-VL 的结果对象 ``PaddleOCRVLResult`` 提供 ``markdown`` 属性，
   返回 dict，识别文本位于 ``res.markdown["markdown_texts"]``；
   本模块从该字段取文本，并对旧版/经典 OCR 结果（``rec_texts``）做兜底；
3. **落盘**：把识别文本写为 UTF-8 的 ``<原图名>.txt``（可选同时保存 .md / .json）。

运行环境：

    必须使用已安装 PaddleOCR 的 ocr_ 环境 Python 运行，例如：

        D:/an/envs/ocr_/python.exe ocr.py
        D:/an/envs/ocr_/python.exe -c "from ocr import ocr_image; r = ocr_image('a.png'); print(r.text)"

用法示例（详见同目录 README.md）：

    from ocr import ocr_image, ocr_images, ocr_directory

    r = ocr_image("扫描件.png")            # 识别并保存 扫描件.txt（同目录）
    print(r.text)                          # 识别文本
    print(r.txt_path)                      # 保存的 txt 路径

    results = ocr_images(["a.png", "b.jpg"], output_dir="out_txt")   # 批量
    results = ocr_directory("images", output_dir="out_txt")          # 整目录
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("rag2.ocr")

# 默认模型权重路径（与引用资料一致，需本地已存在）
DEFAULT_VL_MODEL_DIR = r"D:\modelscope\PaddleOCR-VL-1.6"
DEFAULT_LAYOUT_MODEL_DIR = r"D:\modelscope\PP-DocLayoutV3"
DEFAULT_VL_MODEL_NAME = "PaddleOCR-VL-1.6-0.9B"

# 识别为图片的文件后缀
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


@dataclass
class OCRResult:
    """单张图片的 OCR 识别结果。"""

    image_path: str
    text: str                       # 识别文本（默认等同 markdown，plain=True 时去格式）
    markdown: str = ""              # 原始 Markdown 文本
    json_data: Any = None           # 结构化输出（dict / list，缺失时为 None）
    txt_path: str | None = None     # 已保存的 txt 路径（未保存为 None）
    markdown_path: str | None = None
    json_path: str | None = None
    _pages: list = field(default_factory=list)  # 原始结果对象列表（调试用）

    def to_dict(self) -> dict:
        """转成普通 dict（便于序列化）。"""
        return {
            "image_path": self.image_path,
            "text": self.text,
            "markdown": self.markdown,
            "json_data": self.json_data,
            "txt_path": self.txt_path,
            "markdown_path": self.markdown_path,
            "json_path": self.json_path,
        }

    def __repr__(self) -> str:  # pragma: no cover - 调试打印
        n = len(self.text)
        return f"OCRResult({Path(self.image_path).name}, {n} 字符)"


# ---------------------------------------------------------------------- 文本提取
def _result_to_text(res: Any) -> str:
    """从单个 PaddleOCR-VL 结果对象中提取识别文本（带多重兜底）。"""
    # 1) 首选：markdown 属性 → dict["markdown_texts"]（PaddleOCR-VL 3.7）
    try:
        md = res.markdown
        if isinstance(md, dict):
            text = md.get("markdown_texts") or ""
            if isinstance(text, (list, tuple)):
                text = "\n\n".join(str(x) for x in text)
            if text and str(text).strip():
                return str(text).strip()
    except Exception:  # noqa: BLE001 - 逐级降级，不因属性缺失中断
        pass

    # 2) 兜底：经典 PaddleOCR 结果（rec_texts）
    try:
        texts = res.get("rec_texts") or []
        if texts:
            return "\n".join(str(t) for t in texts).strip()
    except Exception:  # noqa: BLE001
        pass

    # 3) 兜底：字符串化
    try:
        s = str(res)
        if s.strip():
            return s.strip()
    except Exception:  # noqa: BLE001
        pass

    return ""


def _result_to_json(res: Any) -> Any:
    """提取结构化输出（dict / list）；拿不到返回 None。"""
    try:
        data = res.json
        if isinstance(data, (dict, list)):
            return data
    except Exception:  # noqa: BLE001
        pass
    try:
        data = dict(res)
        if data:
            return data
    except Exception:  # noqa: BLE001
        pass
    return None


# ------------------------------------------------------------------ Markdown 去格式
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_CODE_BLOCK = re.compile(r"```.*?```", re.S)
_MD_INLINE_CODE = re.compile(r"`([^`]*)`")
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)
_MD_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_MD_ITALIC = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)|(?<!_)_([^_]+)_(?!_)")
_MD_STRIKE = re.compile(r"~~([^~]+)~~")
_MD_TABLE_SEP = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$", re.M)


def _markdown_to_plain(md: str) -> str:
    """把 Markdown 文本粗略转成纯文本（去掉图片/链接/代码围栏/强调/标题号/表格分隔行）。"""
    # 逐条正则规则剥掉 Markdown 语法标记（图片/链接/加粗/斜体/标题号/表格分隔线…），
    # 让入库文本只剩可检索的「正文」，避免一堆 ** 和 # 号干扰分词与检索。
    text = md
    text = _MD_IMAGE.sub("", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_CODE_BLOCK.sub(lambda m: m.group(0).strip("`").strip(), text)
    text = _MD_INLINE_CODE.sub(r"\1", text)
    text = _MD_HEADING.sub("", text)
    text = _MD_STRIKE.sub(r"\1", text)
    text = _MD_BOLD.sub(r"\1", text)
    text = _MD_ITALIC.sub(r"\1", text)
    text = _MD_TABLE_SEP.sub("", text)
    # 压缩多余空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------- 主类
class ImageOCR:
    """PaddleOCR-VL 图片文字识别器（懒加载，可复用）。"""

    def __init__(
        self,
        vl_model_dir: str | None = None,
        layout_model_dir: str | None = None,
        vl_model_name: str = DEFAULT_VL_MODEL_NAME,
        backend: str | None = None,
        **pipeline_kwargs: Any,
    ) -> None:
        """初始化（此时不加载模型，首次识别时才加载）。

        参数：
            vl_model_dir:      PaddleOCR-VL 权重目录；缺省用 ``D:\\modelscope\\PaddleOCR-VL-1.6``。
            layout_model_dir: 版面检测权重目录；缺省用 ``D:\\modelscope\\PP-DocLayoutV3``。
            vl_model_name:     VL 识别模型名，默认 "PaddleOCR-VL-1.6-0.9B"。
            backend:           VL 推理后端；PaddleOCR 3.7 支持
                               native / vllm-server / sglang-server / fastdeploy-server /
                               mlx-vlm-server / llama-cpp-server，缺省 native（本地推理）。
            **pipeline_kwargs: 其余参数透传给 PaddleOCRVL 构造器。
        """
        self.vl_model_dir = vl_model_dir or DEFAULT_VL_MODEL_DIR
        self.layout_model_dir = layout_model_dir or DEFAULT_LAYOUT_MODEL_DIR
        self.vl_model_name = vl_model_name
        self.backend = backend
        self.pipeline_kwargs = pipeline_kwargs
        self._pipeline: Any = None

    @property
    def pipeline(self) -> Any:
        """懒加载并缓存 PaddleOCRVL 流水线。"""
        if self._pipeline is None:
            try:
                from paddleocr import PaddleOCRVL
            except ImportError as exc:  # pragma: no cover - 依赖缺失提示
                raise RuntimeError(
                    "未找到 paddleocr，请使用 ocr_ 环境运行：D:/an/envs/ocr_/python.exe。"
                ) from exc
            kwargs: dict[str, Any] = {
                "vl_rec_model_name": self.vl_model_name,
                "vl_rec_model_dir": self.vl_model_dir,
                "layout_detection_model_dir": self.layout_model_dir,
            }
            # 3.7 起引擎参数为 vl_rec_backend；None 等价于 native，可显式传入后端
            if self.backend:
                kwargs["vl_rec_backend"] = self.backend
            kwargs.update(self.pipeline_kwargs)
            logger.info("加载 PaddleOCRVL：model=%s dir=%s", self.vl_model_name, self.vl_model_dir)
            self._pipeline = PaddleOCRVL(**kwargs)
        return self._pipeline

    # ------------------------------------------------------------------ 核心识别
    def predict(self, input: str | Path | list) -> list:
        """直接调用 PaddleOCRVL.predict，返回原始结果对象列表（高级用法）。

        注意：PaddleOCR-VL 的 predict 只接受 ``str`` / ``numpy.ndarray``，
        不接受 ``Path``，这里统一转成 ``str``。
        """
        if isinstance(input, Path):
            input = str(input)
        elif isinstance(input, (list, tuple)):
            input = [str(x) if isinstance(x, Path) else x for x in input]
        return list(self.pipeline.predict(input))

    def recognize(
        self,
        image_path: str | Path,
        output_dir: str | Path | None = None,
        *,
        plain: bool = False,
        save_txt: bool = True,
        save_markdown: bool = False,
        save_json: bool = False,
    ) -> OCRResult:
        """识别单张图片并（可选）保存为 txt。

        参数：
            image_path:    图片路径。
            output_dir:    输出目录；缺省为图片所在目录。
            plain:         True 时 ``text`` 为去除 Markdown 格式后的纯文本，否则保留 Markdown。
            save_txt:      是否保存 ``<原图名>.txt``。
            save_markdown: 是否同时保存 ``<原图名>.md``。
            save_json:     是否同时保存 ``<原图名>.json``。

        返回：
            OCRResult 对象。
        """
        image = Path(image_path)
        if not image.exists():
            raise FileNotFoundError(f"图片不存在：{image}")
        out = Path(output_dir) if output_dir else image.parent
        out.mkdir(parents=True, exist_ok=True)

        res_list = self.predict(image)
        if not res_list:
            raise RuntimeError(f"OCR 未返回任何结果：{image}")

        markdown = ""
        json_data = None
        for res in res_list:
            markdown += _result_to_text(res) + "\n"
        markdown = markdown.strip()
        if res_list:
            json_data = _result_to_json(res_list[0])

        text = _markdown_to_plain(markdown) if plain else markdown

        result = OCRResult(
            image_path=str(image),
            text=text,
            markdown=markdown,
            json_data=json_data,
            _pages=res_list,
        )

        if save_txt:
            result.txt_path = self._write(out / f"{image.stem}.txt", text)
        if save_markdown:
            result.markdown_path = self._write(out / f"{image.stem}.md", markdown)
        if save_json and json_data is not None:
            import json
            result.json_path = self._write(
                out / f"{image.stem}.json",
                json.dumps(json_data, ensure_ascii=False, indent=2),
            )
        return result

    @staticmethod
    def _write(path: Path, content: str) -> str:
        path.write_text(content, encoding="utf-8")
        logger.info("已保存：%s（%d 字符）", path, len(content))
        return str(path)

    # ------------------------------------------------------------------ 批量
    def recognize_batch(
        self,
        image_paths: list[str | Path],
        output_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> list[OCRResult]:
        """批量识别多张图片，返回结果列表。"""
        return [self.recognize(p, output_dir=output_dir, **kwargs) for p in image_paths]

    def recognize_directory(
        self,
        directory: str | Path,
        output_dir: str | Path | None = None,
        *,
        recursive: bool = False,
        **kwargs: Any,
    ) -> list[OCRResult]:
        """识别目录下所有图片，返回结果列表。"""
        root = Path(directory)
        if not root.is_dir():
            raise NotADirectoryError(f"目录不存在：{root}")
        pattern = "**/*" if recursive else "*"
        paths = [
            p for p in sorted(root.glob(pattern))
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        ]
        return self.recognize_batch(paths, output_dir=output_dir, **kwargs)


# ------------------------------------------------------------------ 全局单例
_pipeline_singleton: ImageOCR | None = None


def get_pipeline(**kwargs: Any) -> ImageOCR:
    """返回全局缓存的 ImageOCR 实例（多张图复用同一模型，避免重复加载）。"""
    global _pipeline_singleton
    if _pipeline_singleton is None or kwargs:
        _pipeline_singleton = ImageOCR(**kwargs)
    return _pipeline_singleton


# ---------------------------------------------------------------------- 便捷函数
def ocr_image(
    image_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    plain: bool = False,
    save_txt: bool = True,
    save_markdown: bool = True,
    save_json: bool = False,
    **pipeline_kwargs: Any,
) -> OCRResult:
    """便捷函数：识别单张图片并保存为 txt（等价于 ``ImageOCR(**kw).recognize(...)``）。"""
    return get_pipeline(**pipeline_kwargs).recognize(
        image_path,
        output_dir=output_dir,
        plain=plain,
        save_txt=save_txt,
        save_markdown=save_markdown,
        save_json=save_json,
    )


def ocr_images(
    image_paths: list[str | Path],
    output_dir: str | Path | None = None,
    **kwargs: Any,
) -> list[OCRResult]:
    """便捷函数：批量识别多张图片。"""
    return get_pipeline().recognize_batch(image_paths, output_dir=output_dir, **kwargs)


def ocr_directory(
    directory: str | Path,
    output_dir: str | Path | None = None,
    **kwargs: Any,
) -> list[OCRResult]:
    """便捷函数：识别目录下所有图片。"""
    return get_pipeline().recognize_directory(directory, output_dir=output_dir, **kwargs)


__all__ = [
    "OCRResult",
    "ImageOCR",
    "ocr_image",
    "ocr_images",
    "ocr_directory",
    "get_pipeline",
    "_markdown_to_plain",
]


if __name__ == "__main__":  # pragma: no cover - 需 ocr_ 环境与真实模型
    import sys

    # 默认示例：识别 data/img/paddleocr_vl_demo.png，输出 txt 到 data/ 目录。
    # 命令行可覆盖：D:/an/envs/ocr_/python.exe ocr.py <图片路径> [输出目录]
    DEFAULT_IMG = r"D:\桌面D\专高\专高六\项目\RAG_2\data\img\paddleocr_vl_demo.png"
    DEFAULT_OUT = r"D:\桌面D\专高\专高六\项目\RAG_2\data"

    img = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMG
    out = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    r = ocr_image(img, output_dir=out)
    print("=== 输入图片 ===", img)
    print("=== 识别文本 ===")
    print(r.text)
    print("=== 保存位置 ===", r.txt_path)
