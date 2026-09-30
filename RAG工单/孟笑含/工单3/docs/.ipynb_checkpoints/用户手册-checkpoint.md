# 用户手册 — 基于 PDF 文档的问答系统

> 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

## 一、系统启动

### 1.1 安装依赖

pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

### 1.2 启动 Web 服务

export HF_ENDPOINT=https://hf-mirror.com

python3 app.py

成功输出：

✅ PDF 解析：548 页，477 个表格
✅ 文本切分：2010 块
✅ 表格块加入：455 块，总计 2465 块
✅ 模型加载完成，向量维度：768
✅ 索引构建完成：2465 块，维度 768
✅ 系统初始化完成
* Running on local URL:  http://0.0.0.0:7860

### 1.3 访问界面

- 本地：浏览器打开 http://127.0.0.1:7860
- AutoDL：控制台 → 自定义服务 → 添加端口 7860

## 二、界面使用

### 2.1 问答 Tab

1. 在输入框输入问题，或点击下方示例
2. 点击「发送」按钮
3. 对话记录区显示答案 + 参考来源 + 响应时间

示例问题：

- 报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？
- 武汉兴图新科电子股份有限公司参与制定了哪个技术标准？
- 武汉兴图新科电子股份有限公司注册资本是多少？
- 武汉兴图新科电子股份有限公司法定代表人是谁？

### 2.2 文档管理 Tab

当前版本自动加载 ./data/招股说明书1.pdf

### 2.3 评估报告 Tab

- 点击「运行批量评估」
- 显示 RAG vs 纯 LLM 的对比表格

## 三、常见问题

### Q1：启动报 ImportError: cannot import name HfFolder

解决：pip install --upgrade gradio -i https://mirrors.aliyun.com/pypi/simple/

### Q2：模型加载报 safetensors 错误

解决：

export HF_ENDPOINT=https://hf-mirror.com

python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer("BAAI/bge-base-zh-v1.5", model_kwargs={"use_safetensors": True})"

### Q3：Web 启动后 127.0.0.1:7860 打不开

解决：AutoDL 需通过「自定义服务」映射端口。

### Q4：点发送后报 Data incompatible with messages format

解决：确保 app.py 中 Chatbot 声明为 type=messages，chat_interface 返回 role/content 格式。

### Q5：响应时间太慢

- 第一次启动较慢（模型加载 + 索引构建约 10s）
- 后续每次查询 0.01s 左右

## 四、关闭服务

终端按 Ctrl + C。

后台运行：

nohup python3 app.py > server.log 2>&1 &
tail -f server.log
pkill -f "python3 app.py"

## 五、联系信息

- 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
- 创建人：王洪荣
- 创建时间：2025-01-21
- 版本：V1.1
