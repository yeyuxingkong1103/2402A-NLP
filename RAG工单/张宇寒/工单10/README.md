# 工单10：Docker金融GraphRAG服务

本项目把工单9的金融年报GraphRAG问答和评测系统打包为可部署到服务器的Docker服务。

## 快速启动

Windows Docker Desktop：

```powershell
.\deploy.ps1
```

Linux服务器：

```bash
chmod +x deploy.sh
./deploy.sh
```

访问：`http://127.0.0.1:8021/`，远程服务器使用`http://服务器IP:8021/`。

## 交付内容

- `Dockerfile`：Python 3.11、非root用户、镜像健康检查。
- `docker-compose.yml`：端口、安全限制、资源限制、自动恢复。
- `.dockerignore`：禁止API Key和本地缓存进入镜像。
- `deploy.ps1`、`deploy.sh`：一键构建、启动和验收。
- `container_check.py`：健康、检索质量和响应时间自动验收。
- `部署说明.md`：服务器部署与运维命令。
- `部署问题记录.md`：部署过程问题、原因和解决方法。

## 验收目标

- 容器状态为healthy。
- GraphRAG索引返回ready。
- Context Precision >= 0.8。
- Context Recall >= 0.9。
- 所有参考题响应时间 < 3秒。

## DeepSeek

在`.env`中填写：

```text
DEEPSEEK_API_KEY=
```

留空时仍能运行，系统自动使用本地GraphRAG证据摘要。

详细步骤见[部署说明.md](部署说明.md)。
