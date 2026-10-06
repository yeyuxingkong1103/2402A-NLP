# 技术文档

## 数据流

PDF 上传后计算 SHA-256 并建立文档记录。Celery Worker 先调用远程 MinerU，失败时使用 PyMuPDF。文本按页和章节切片，保留物理页码；`bge-m3` 生成向量写入 Milvus，同时建立 jieba + BM25 本地索引。查询时由 DeepSeek 输出 Query Plan，向量召回和 BM25 召回使用 RRF 合并，再通过 BGE Reranker 重排，最后仅将高相关证据交给 DeepSeek。

## 证据约束

回答必须包含 `[证据N]` 引用。API 返回引用 chunk、PDF 物理页码、章节、文本和重排分数。无证据时拒答。后续增强阶段应加入金额、比例、年份的程序化校验和一次自动修正。

## 视觉模型

复杂图片和表格使用阿里云百炼的 OpenAI 兼容 API 调用 `qwen-vl-max`。默认地址为 `https://dashscope.aliyuncs.com/compatible-mode/v1`，通过 `DASHSCOPE_API_KEY`、`DASHSCOPE_BASE_URL` 和 `DASHSCOPE_VL_MODEL` 配置。不同地域或业务空间使用百炼控制台提供的对应兼容端点。视觉请求只发送必要页面，不在本地日志保存完整图片或密钥。


PostgreSQL 保存业务数据，Milvus 保存向量，Redis 保存 Celery 队列。`data/` 保存上传 PDF、解析 JSON、页面资源和评估输出。Milvus 的 etcd/MinIO 使用独立 Docker volumes。

## 安全与运维

API 密钥仅来自 `.env`；管理接口要求 `X-Admin-Token`。基础设施服务不映射到宿主机，只在 Docker 内网通信。日志不记录密钥或完整远程请求。生产部署前应替换默认口令、限制 CORS、启用反向代理和 HTTPS。

## 已知边界

当前版本已提供百炼 Qwen-VL-Max 客户端和配置字段；视觉页面识别与人工审核队列仍属于增强阶段；远程 MinerU 的具体 API 字段因部署版本不同，集中在 `MinerUClient` 中适配。Ragas 与纯 DeepSeek 基线需要补充已人工确认的 Ground Truth 后运行。
