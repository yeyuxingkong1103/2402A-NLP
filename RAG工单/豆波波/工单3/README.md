# 工单3 - PDF 文本解析与抽取优化

`parse_report.py` 记录 PDF 文本页数、字符数、空页比例、Unicode 清洗结果和分块统计，可用于定位扫描件、乱码和过短页。

```powershell
python parse_report.py --input 规格说明.pdf --output parse_report.json
```


