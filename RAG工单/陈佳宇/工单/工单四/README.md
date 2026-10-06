# 工单4｜PDF图像多模态RAG检索系统
项目名称：基于CLIP+Milvus的PDF图像多模态检索系统
实训工单：工单4
作者：陈佳宇

## 项目简介
本项目实现多模态RAG检索，将PDF招股说明书每一页渲染为图片，使用CLIP多模态模型对图片进行向量编码存入Milvus向量数据库。用户通过Web页面输入自然语言问题，将文本转为向量，在向量库中检索相似度最高的PDF页面图片，返回匹配页面名称、相似度分数与检索耗时。

## 技术栈
- Python 3.12
- 多模态模型：openai/clip-vit-base-patch32
- 向量数据库：Milvus
- Web后端：Flask
- PDF处理：PyMuPDF
- 向量处理：Torch、Transformers、Numpy

## 项目目录
工单 4/
├── main.py             # Flask 主程序，向量检索 + Web 服务
├── gen_pdf.py          # 生成模拟招股说明书 PDF
├── create_collection.py# Milvus 集合创建脚本
├── requirements.txt    # 项目依赖
├── pdf_images/         # PDF 转图片输出目录
└── README.md           # 项目说明文档

## 快速启动
1. 创建虚拟环境，安装依赖
```bash
pip install -r requirements.txt
生成模拟招股说明书 PDF
python gen_pdf.py
创建 Milvus 向量集合
python create_collection.py
python main.py
5. 浏览器访问：[http://172.17.189.52:7862](http://172.17.189.52:7862)

## 使用说明

1. 在网页输入框输入问题，例如：`销售部有几个部门？`
2. 点击【提问检索】
3. 系统返回相似度最高的 PDF 页面图片名称、相似度得分、检索耗时。

## 功能清单

1. PDF 文档生成
2. PDF 页面渲染为图片
3. CLIP 多模态向量生成
4. Milvus 向量存储
5. 文本向量检索
6. Flask Web 可视化交互界面

## 版本信息

- Milvus: 3.x
- transformers
- torch
- Flask

## 2. 设计.md
```markdown
# 工单4 - 设计文档
## 1. 需求分析
### 业务需求
实现PDF招股说明书多模态检索系统。用户输入自然语言问题，系统自动找到PDF中最相关的页面图片，用于文档资料检索。

### 功能需求
1. 支持PDF文件读取，将PDF每一页转为图片
2. 使用CLIP多模态模型，对图片生成向量嵌入
3. 向量存入Milvus向量数据库
4. Web页面接收用户文本提问，文本转为向量
5. 在向量库执行相似度检索，返回Top2匹配页面
6. Web页面展示检索结果、相似度、耗时

### 非功能需求
1. 检索响应速度 < 1s
2. 支持跨主机访问Web页面（WSL环境）
3. 模块化代码，便于后续扩展OCR+大模型问答

## 2. 总体架构设计
整体分为四层：
1. 文档层：PDF招股说明书，使用PyMuPDF渲染为图片
2. 向量编码层：CLIP（vit-base-patch32）多模态模型，支持图片/文本统一向量空间
3. 向量存储层：Milvus向量数据库，存储512维向量
4. Web服务层：Flask，提供前端页面与检索接口

### 流程设计
1. 预处理：PDF -> 页面图片 -> CLIP图片向量 -> Milvus入库
2. 检索流程：用户输入问题 → CLIP文本向量 → Milvus相似度搜索 → 返回Top2结果 → Web展示

## 3. 数据库设计
Milvus集合名称：`rag_waterwork4`
- 字段1: id 主键，auto_id自动生成
- 字段2: image_name VARCHAR，图片文件名
- 字段3: image_path VARCHAR，图片本地路径
- 字段4: vector FLOAT_VECTOR，维度512，CLIP输出向量

索引类型：IVF_FLAT，距离度量：L2距离（数值越小相似度越高）

## 4. 接口设计
### POST /ask
入参：`{"q":"用户问题"}`
返回：JSON字符串，包含检索耗时、匹配图片名称、相似度分数。

## 5. UI设计
简单单页面：输入框、提交按钮、结果展示区域。页面展示检索耗时，匹配图片名称、相似度得分。
