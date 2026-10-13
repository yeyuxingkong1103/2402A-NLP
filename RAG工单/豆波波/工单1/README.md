# 工单1 - PDF 基础问答系统

`qa.py` 完成 PDF/文本读取、固定重叠分块、混合检索和证据输出。它是一个可离线运行的 baseline，真实生成式回答可以把 `results` 交给任意 LLM。

```powershell
python qa.py --input sample_docs.txt --query "泵的作用" --top-k 3 --output answer.json
```

验收：输出包含 query、检索方法、文档 id、分数和原文证据；把 `--input` 换成 PDF 需要安装 `pypdf`。

