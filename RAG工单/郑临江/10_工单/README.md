# 工单 10：金融问答系统部署

**工单编号**：人工智能NLP-RAG-金融问答系统部署

## 一、项目简介

将金融问答系统以 Docker 容器形式部署到服务器，提供 HTTP 问答服务，
支持数据卷持久化、容器间数据共享与网络配置。

## 二、目录结构

```
10_工单/
├── app.py             # Flask 问答服务（/ 界面 + /ask 接口）
├── rag.py             # RAG 核心（PDF 解析 + BM25 检索 + 生成）
├── config.py          # 配置（DATA_DIR 环境变量覆盖）
├── Dockerfile         # 容器镜像定义
├── docker-compose.yml # 编排（卷、网络）
├── deploy.md          # 部署文档与问题记录
├── requirements.txt
└── README.md
```

## 三、本地运行（非容器）

```bash
pip install -r requirements.txt
python app.py
# 访问 http://localhost:8000
```

## 四、容器部署

见 `deploy.md`，核心命令：`docker build -t finance-qa .` 与
`docker run -d -p 8000:8000 -v "附件目录:/data:ro" finance-qa`。

## 五、验收对照

- 部署后问答服务测试通过；
- 容器启动运行正常、无异常日志；
- Docker 卷持久化数据、支持容器间共享；
- 网络配置正确（可与其他服务通信）；
- 代码注释含工单编号：人工智能NLP-RAG-金融问答系统部署。
