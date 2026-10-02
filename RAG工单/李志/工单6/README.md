# 工单6 - 混合检索

支持 `bm25`、`vector`、`hybrid` 三种可切换策略，混合模式采用分数归一化加权融合。

```powershell
python .\工单6\main.py --docs "附件目录" --query "研发中心和技术风险" --mode hybrid
```

产出：`outputs/retrieval.json`。

浏览器界面：`python .\工单6\web.py "招股说明书1.pdf"`，打开 `http://127.0.0.1:8006`。
