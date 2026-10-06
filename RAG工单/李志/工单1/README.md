# 工单1 - 基于 PDF 的问答系统

实现 PDF/文本解析、分块、混合检索、带页码溯源的答案生成。

```powershell
python .\工单1\main.py --docs "C:\Users\Administrator\Desktop\工单\工单1-13\附件\招股说明书1-无水印.pdf" --query "公司的主营业务是什么？"
```

结果：`outputs/result.json`。不传 `--docs` 即运行内置演示。

浏览器界面：`python .\工单1\web.py "招股说明书1.pdf"`，打开 `http://127.0.0.1:8001`。
