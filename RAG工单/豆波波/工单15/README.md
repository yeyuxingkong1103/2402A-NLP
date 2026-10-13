# 工单15 - 技术图纸与文本跨模态检索

`cross_modal_retrieval.py` 将 PDF 图片元数据和文本块统一为检索记录，支持文本关键词命中图纸标题/尺寸，并预留 OCR 与 image embedding 字段。

```powershell
python cross_modal_retrieval.py --input 规格说明.pdf --query "管线尺寸" --output cross_modal.json
```

