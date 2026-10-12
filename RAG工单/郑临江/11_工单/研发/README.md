# 研发：Embedding 微调

处理流程：问答对生成 → 嵌入训练/损失 → 模型输出 → 检索评估。

源码与配置集中在本目录，原文件间的模块关系保留。测试/优化入口已补充研发模块路径。
从工单根目录执行：

```powershell
python -m pip install -r "研发/requirements.txt"
python run.py 研发/data_gen.py
```

Embedding 完整训练：`python run.py 研发/train.py`；训练完成后再运行测试评估。

训练、模型下载和问答服务启动应在附件、模型及依赖准备好之后执行。
