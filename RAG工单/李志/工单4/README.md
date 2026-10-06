# 工单4 - PDF 图像内容解析及检索

从图题、结构图附近文本生成图像语义描述，与正文共同检索。生产环境可把生成的描述替换为 CLIP/多模态模型结果。

```powershell
python .\工单4\main.py --docs "招股说明书2.pdf" --query "组织结构图中的销售部门"
```

产出：`outputs/image_index.json`。
