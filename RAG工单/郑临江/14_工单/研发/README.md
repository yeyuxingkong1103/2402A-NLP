# 研发：低质量工业 PDF 解析

处理流程：扫描页识别 → OCR 兜底 → 模拟任务队列 → 分块与检索。

源码与配置集中在本目录，原文件间的模块关系保留。测试/优化入口已补充研发模块路径。
从工单根目录执行：

```powershell
python -m pip install -r "研发/requirements.txt"
python run.py 研发/main.py
```



训练、模型下载和问答服务启动应在附件、模型及依赖准备好之后执行。
