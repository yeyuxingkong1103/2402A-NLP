# 工单5 - Query 理解优化

实现意图识别、口语归一、复合问题拆分及分别检索。

```powershell
python .\工单5\main.py --docs "招股说明书1.pdf" --query "这家公司干什么，钱用哪？"
```

产出：`outputs/query_understanding.json`。

浏览器界面：`python .\工单5\web.py "招股说明书1.pdf"`，打开 `http://127.0.0.1:8005`。
