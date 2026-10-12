# 研发：多轮 Query 理解

处理流程：历史记录 → 实体与意图识别 → 指代消解/重写 → 检索生成。

源码与配置集中在本目录，原文件间的模块关系保留。测试/优化入口已补充研发模块路径。
从工单根目录执行：

```powershell
python -m pip install -r "研发/requirements.txt"
python run.py 研发/app.py
```



训练、模型下载和问答服务启动应在附件、模型及依赖准备好之后执行。
