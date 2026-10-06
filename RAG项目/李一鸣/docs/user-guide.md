# 项目使用文档

## 1. 项目位置

- 项目目录：`D:\rag-roleplay-system`
- Conda 环境：`D:\develop_tool1\anaconda3\envs\zhuangao6`
- D 盘依赖目录：`D:\rag-roleplay-system\conda_packages`
- 配置文件：`D:\rag-roleplay-system\.env`
- 运行日志：`D:\rag-roleplay-system\logs\app.log`

`zhuangao6` 的 Conda 环境目录由管理员组保护，项目依赖通过 `PYTHONPATH` 从 D 盘 `conda_packages` 加载。这样启动解释器仍然是 Conda 的 `zhuangao6\python.exe`，同时不会使用 C 盘用户级 Python 包。

## 2. 启动项目

### 推荐启动方式

在 PowerShell 中执行：

```powershell
Set-Location D:\rag-roleplay-system
.\run_zhuangao6.ps1
```

脚本会自动完成以下设置：

1. 使用 Conda `zhuangao6` 的 Python。
2. 设置 `PYTHONNOUSERSITE=1`，禁止加载 C 盘用户包。
3. 设置 `PYTHONDONTWRITEBYTECODE=1`，避免生成无用的 Python 字节码缓存。
4. 设置 `PYTHONPATH=D:\rag-roleplay-system\conda_packages`。
5. 启动 `run.py` 和 FastAPI 服务。

### 手动启动方式

```powershell
conda activate zhuangao6
Set-Location D:\rag-roleplay-system
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONPATH = "D:\rag-roleplay-system\conda_packages"
python run.py
```

默认地址：`http://127.0.0.1:8000`

Swagger：`http://127.0.0.1:8000/docs`

OpenAPI：`http://127.0.0.1:8000/openapi.json`

## 3. 配置说明

`.env` 是本机配置，`.env.example` 是配置模板。当前开发模式默认完全离线：

```dotenv
APP_ENV=dev
PORT=8000
DATABASE_URL=sqlite+aiosqlite:///D:/rag-roleplay-system/data/rag_roleplay.db
REDIS_ENABLED=false
MILVUS_ENABLED=false
EMBEDDING_PROVIDER=hash
LLM_PROVIDER=mock
RAGAS_ENABLED=false
```

接入 OpenAI 兼容模型时修改：

```dotenv
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=你的密钥
LLM_MODEL=deepseek-chat
```

接入真实 RAGAS 评测时，同时设置：

```dotenv
RAGAS_ENABLED=true
```

没有配置真实模型或密钥时，评测接口使用本地代理指标，不会发起外部请求。

## 4. 第一次使用

### 查看角色

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/roles | ConvertTo-Json -Depth 5
```

系统首次启动会自动创建 5 个示例角色。记录返回结果中的 `id`，后续聊天请求需要使用它。

### 上传知识库文档

支持格式：PDF、TXT、MD、CSV、JSON。

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/documents/upload" `
  -F "file=@D:\rag-roleplay-system\data\demo_knowledge.txt" `
  -F "source=demo" `
  -F "role_id=global"
```

处理流程是：保存文件、计算 SHA-256、解析文本、分块、生成向量、写入本地索引或 Milvus，并更新文档状态。

### 查看知识库文档

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/documents | ConvertTo-Json -Depth 5
```

### 执行检索

```powershell
$body = @{ query = "高血压 监测"; top_k = 5 } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/api/v1/search `
  -Method Post -ContentType "application/json; charset=utf-8" -Body $body |
  ConvertTo-Json -Depth 8
```

返回结果包含：`content`、`source`、`score`、`dense_score`、`lexical_score` 和 `retrieval_method`。正常混合检索的 `retrieval_method` 为 `hybrid`。

## 5. 聊天接口

### 普通聊天

```powershell
$roleId = "从 /api/v1/roles 获取的角色 ID"
$body = @{
  user_id = "demo-user"
  role_id = $roleId
  conversation_id = "demo-conversation"
  message = "高血压管理有哪些注意事项？"
  stream = $false
  top_k = 5
} | ConvertTo-Json

Invoke-RestMethod http://127.0.0.1:8000/api/v1/chat `
  -Method Post -ContentType "application/json; charset=utf-8" -Body $body |
  ConvertTo-Json -Depth 8
```

返回字段：

- `answer`：角色回答。
- `citations`：召回的知识库引用。
- `conversation_id`：会话标识。
- `trace_id`：本次 RAG 链路标识，可用于日志排查。

### 流式聊天

将请求中的 `stream` 设置为 `true`：

```json
{
  "user_id": "demo-user",
  "role_id": "角色 ID",
  "conversation_id": "demo-conversation",
  "message": "请介绍一下这个角色。",
  "stream": true,
  "top_k": 5
}
```

响应类型是 `text/event-stream`，事件包括：

- `meta`：trace id 和引用。
- `token`：回答片段。
- `done`：完整回答和耗时。

## 6. 多轮记忆

`user_id + role_id + conversation_id` 共同隔离会话。

- 短期记忆：优先使用 Redis；Redis 不可用时使用进程内内存降级。
- 长期记录：保存到 SQLite 或 MySQL 的 `conversation_messages` 表。
- `conversation_id` 不同，会话上下文互相隔离。

启用 Redis：

```dotenv
REDIS_ENABLED=true
REDIS_URL=redis://127.0.0.1:6379/0
```

## 7. RAGAS 评测

请求：`POST /api/v1/evaluate`

```json
{
  "samples": [
    {
      "question": "高血压管理包括什么？",
      "answer": "包括生活方式管理和规范测量。",
      "contexts": ["高血压管理包括生活方式干预、规范测量。"],
      "ground_truth": "生活方式管理和规范测量。"
    }
  ]
}
```

默认返回 `provider=local_proxy`。开启 `RAGAS_ENABLED=true` 并配置非 Mock 大模型后，会执行真实 RAGAS 评测；外部模型调用失败时自动记录异常并回退。

## 8. 日志与排错

应用日志：

```powershell
Get-Content D:\rag-roleplay-system\logs\app.log -Tail 50 -Wait
```

日志通常包含：时间、级别、模块、request id、用户、角色、会话、召回数量、重排结果和阶段耗时。

健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/health | ConvertTo-Json -Depth 5
```

常见状态：

- `vector_store.ready=true`：向量索引可用。
- `memory=local_fallback`：Redis 未启用或不可连接，系统仍可运行。
- `llm=mock`：当前是离线演示模式。
- `embedding=hash`：当前使用本地确定性向量，生产环境可切换 BGE-m3。

## 9. 测试

```powershell
Set-Location D:\rag-roleplay-system
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONPATH = "D:\rag-roleplay-system\conda_packages"
conda run -n zhuangao6 python -s -m pytest -q -p no:tmpdir -p no:cacheprovider
```

当前测试覆盖分块、记忆、后处理、混合检索和离线评测。

## 10. 停止服务

开发环境可在运行窗口按 `Ctrl+C`。也可以执行：

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -like '*D:\develop_tool1\anaconda3\envs\zhuangao6\python.exe*run.py*' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

不要删除 `data`、`logs` 或 `conda_packages`，它们分别保存运行数据、诊断日志和当前 Conda 启动所需的依赖。
