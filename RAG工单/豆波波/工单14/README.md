# 工单14 - 低质量工业 PDF 解析与信息损失修复

`industrial_pdf_repair.py` 先保留页边界，再做 Unicode/断行修复、空页和疑似 OCR 页诊断；同时复制了 RAGFlow DeepDoc 集成说明。工业扫描 PDF 建议在真实 OCR 服务上继续复核。

```powershell
python industrial_pdf_repair.py --input 规格说明.pdf --output repaired.json
```

