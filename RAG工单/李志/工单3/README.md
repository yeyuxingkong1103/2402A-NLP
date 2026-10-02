# 工单3 - PDF 表格解析及检索优化

解析对齐文本/管道符形式的表格行，与正文统一建索引并保留页码。

```powershell
python .\工单3\main.py --docs "招股说明书1.pdf" "招股说明书2.pdf" --query "募集资金项目"
```

产出：`outputs/tables_and_search.json`。
