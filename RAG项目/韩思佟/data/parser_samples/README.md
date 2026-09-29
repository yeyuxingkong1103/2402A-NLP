# 四种 PDF 解析结果

四个 TXT 都来自《国家基层高血压防治管理指南2020版》原 PDF 的前 3 页，分别由对应工具真实运行生成。

| 文件 | 现场怎么说 |
|---|---|
| `pymupdf.txt` | PyMuPDF 直接读取数字 PDF 的文字，速度快，作为普通 PDF 主解析器。 |
| `pdfplumber.txt` | PDFPlumber 读取文字并保留表格行列，补充表格解析能力。 |
| `paddleocr.txt` | PaddleOCR 先把页面转成图片，再识别图片中的中文，适合扫描件。 |
| `mineru.txt` | MinerU 分析页面版式和标题层级，适合复杂排版文档。 |

生成代码在 `scripts/export_parser_samples.py`，四种工具的项目实现入口在 `app/internal/offline_engine.py`。
