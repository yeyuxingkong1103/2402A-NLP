# 核心代码

本目录是当前 RAG 客服问答系统的可运行源码交付副本。

## 目录

- `app/`：FastAPI、知识库摄取、RAG、记忆、LLM 和安全模块；
- `static/`：基础 Web 问答页面；
- `sql/`：MySQL 初始化脚本；
- `build_knowledge_base.py`：离线知识库构建入口；
- `requirements.txt`：Python 依赖；
- `.env.example`：环境变量模板。

## 推荐运行方式

请从 `RAG项目/` 根目录使用部署脚本：

```bash
bash 05-部署/check_env.sh
bash 05-部署/install.sh --mode cpu
bash 05-部署/run.sh --mode development
bash 05-部署/shutdown.sh
```

如果使用 NVIDIA GPU，将安装命令改为：

```bash
bash 05-部署/install.sh --mode cuda
```

部署脚本会自动进入本目录，不需要手动修改 Python 路径。

## 手动启动

```bash
cd 02-研发/核心代码
cp .env.example .env
conda activate rag-kf
python -m app.main
```

`.env`、运行日志、PID、上传文件和模型权重属于运行数据，不应提交到代码仓库。
