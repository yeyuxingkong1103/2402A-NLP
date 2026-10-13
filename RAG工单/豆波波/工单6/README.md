# 工单6 - 混合检索

`hybrid_compare.py` 同时运行 lexical、dense proxy 和 hybrid 三种结果，输出每种方法的排序与分数，方便调权和回归。

```powershell
python hybrid_compare.py --input sample_corpus.txt --query "阀门维护" --output comparison.json
```

