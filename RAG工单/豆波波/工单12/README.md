# 工单12 - LightRAG 优化

`lightrag.py` 实现轻量级的 local/global 双层索引、增量更新和联合检索，接口与 LightRAG 的组织方式一致但不依赖外部数据库。

```powershell
python lightrag.py index --input sample_corpus.txt --index index.json
python lightrag.py query --index index.json --query "企业风险"
```

