# 工单4 - 图片内容解析与检索优化

`figure_metadata.py` 从 PDF 的 `/XObject` 图像资源提取尺寸、格式、页码及文件名，并为每张图预留 OCR/图像向量字段。没有 `pypdf` 时会输出可解释的降级结果。

```powershell
python figure_metadata.py --input 规格说明.pdf --output figures.json
```

