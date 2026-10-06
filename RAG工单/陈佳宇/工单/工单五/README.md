# 招股说明书RAG智能问答系统
项目：工单1
姓名：陈佳宇

## 项目简介
本项目基于RAG检索增强生成技术，实现对招股说明书PDF文档的解析、文本分块、向量入库、语义检索与大模型问答。用户可通过Web页面输入问题，系统检索PDF文档相关片段，交给大模型生成基于文档内容的回答。

## 技术栈
- Web框架：Flask
- PDF解析：PyMuPDF
- 向量数据库：Milvus
- 大模型：DeepSeek Chat API
- 编程语言：Python3.12

## 项目目录结构
├── main.py                # Flask 网页后端，RAG 检索 + 问答接口
├── create_collection.py   # Milvus 集合创建脚本（建表、索引）
├── pdf_loader.py          # PDF 读取、文本分块、向量入库脚本
├── gen_pdf.py             # 自动生成模拟招股说明书 PDF
├── README.md              # 项目说明文档
├── 设计文档.md
├── 研发文档.md
├── 测试文档.md
├── 优化文档.md
├── 部署文档.md
└── 招股说明书.pdf         # 生成的 PDF 文档


## 环境依赖安装
```bash
# 进入虚拟环境
source venv_gongdan1/bin/activate
# 安装依赖
pip install flask pymilvus pymupdf numpy openai
## 项目启动步骤

1. 启动 Milvus 服务（本地 Docker）
2. 创建 Milvus 向量集合
python3 create_collection.py
生成招股说明书 PDF
python3 gen_pdf.py
PDF 文档解析，向量存入 Milvus
python3 pdf_loader.py
启动 Flask 网页服务
python3 main.py
6. 浏览器访问：`http://172.17.189.52:7860`，输入问题进行问答

## 功能说明

1. PDF 文档加载与文本提取
2. 文本切片分块处理
3. 文本向量化，存入 Milvus 向量库
4. 用户问题向量检索，召回文档相关片段
5. 问题改写，结合参考文档调用 DeepSeek 大模型生成答案
6. Web 前端页面，浏览器直接交互问答

## 注意事项

- 本项目使用简易向量用于演示，保证 RAG 完整链路；
- 需要配置 DeepSeek API 密钥，才能调用大模型；
- Milvus 服务需要提前启动，端口默认 19530。


---

## 2. 设计文档.md
```markdown
# 设计文档｜招股说明书RAG智能问答系统
## 1. 项目背景与目标
### 目标
实现基于PDF招股说明书的RAG问答系统，用户通过Web页面提问，系统基于文档内容回答，完整实现RAG全链路，完成工单实训任务。

### 核心需求
1. 支持读取PDF招股说明书，提取文档文本
2. 文本进行分块处理，向量化存入Milvus向量数据库
3. 用户输入问题，进行向量相似度检索，召回相关文档片段
4. 将召回片段+用户问题送入大模型，生成基于文档的回答
5. 提供Web网页界面，支持浏览器访问与提问交互

## 2. 总体架构设计
整体分为4大模块：
1. **文档处理模块**：PDF读取、文本提取、文本分块
2. **向量存储模块**：文本向量化、Milvus集合创建、向量插入、向量检索
3. **大模型问答模块**：问题改写、上下文拼接、调用DeepSeek API生成答案
4. **Web前端模块**：Flask后端，网页表单提交，问答结果展示

## 3. 数据库设计（Milvus向量集合）
集合名称：`rag_workorder5`
| 字段名 | 类型 | 属性 | 说明 |
| ---- | ---- | ---- | ---- |
| id | INT64 | 主键，自增 | 每条向量唯一ID |
| vector | FLOAT_VECTOR(128) | 向量字段 | 文本向量，维度128 |
| text | VARCHAR(2000) | 字符串 | 原始文本块内容 |

索引：IVF_FLAT，距离度量L2

## 4. 接口设计
### 网页首页 GET /
返回问答网页HTML页面
### 问答接口 POST /ask
入参：`user_input` 用户问题
出参：大模型返回的回答文本
流程：用户提问 → 问题改写 → 向量检索召回文档 → 拼接提示词 → 调用大模型 → 返回答案

## 5. 业务流程设计
1. 预处理阶段：读取PDF → 文本分块 → 生成向量 → 写入Milvus
2. 问答阶段：用户输入问题 → 问题向量化 → Milvus相似度检索，召回Top3文本块 → 把文档片段和问题一起提交给DeepSeek → 返回回答，页面展示结果
