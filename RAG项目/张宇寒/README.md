# 知途 · 民事法律 RAG 问答系统

提供法律知识、依据和参考处理建议，不代替律师评估，不代表法院裁判结论。

## 启动问答服务

先启动Docker Desktop，确认已有数据库服务运行，再在项目根目录执行：

```powershell
docker compose start
& "D:\Program File\conda\envs\legal_rag\python.exe" run.py
```

当前页面地址：[http://127.0.0.1:7294](http://127.0.0.1:7294)。端口以本地配置为准。
这是已有部署的启动方式；首次部署需要单独准备现有持久化卷和配置，不能随意创建空库替代法律库。

## 公共资料处理

公共资料统一由 `data_pipeline/main.py` 处理：读取、解析、清洗检查、保存。
扫描页保留本地OCR；单份PDF或图片选择 `--pdf-method auto-vision` 后，OCR出现明显问题才用已有多模态模型补读（会上传页面并产生接口费用）。默认只在本地读取、不连接数据库。
用户上传仍走原有后台入口，与公共知识库分开。

在项目根目录处理现有公共资料：

```powershell
& "D:\Program File\conda\envs\legal_rag\python.exe" -m data_pipeline.main
```

已有数据库可以继续用于问答，不需要重新入库。
## 项目文档

- [需求文档](docs/requirements.md)
- [技术文档](docs/technical.md)：项目分工、数据库约定与资料处理用法。
- [测试文档](docs/testing.md)：自动化测试命令与手工验收说明。
