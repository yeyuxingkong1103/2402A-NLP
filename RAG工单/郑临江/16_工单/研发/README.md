# 研发：视觉语言模型微调

处理流程：IMDR 转换 → 数据集注册 → LoRA 训练 → 专业指标评估。

源码与配置集中在本目录，原文件间的模块关系保留。测试/优化入口已补充研发模块路径。
从工单根目录执行：

```powershell
python -m pip install -r "研发/requirements.txt"
python run.py 研发/data_convert.py
```



训练、模型下载和问答服务启动应在附件、模型及依赖准备好之后执行。
