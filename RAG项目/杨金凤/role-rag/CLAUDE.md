# 项目约定

## 环境
- MySQL：库 role_rag（生产/开发）、role_rag_test（测试），用户 rag
- MYSQL_URL 从 .env 读，测试里替换库名为 role_rag_test
- Redis：本地 6379，db0
- DeepSeek：DEEPSEEK_API_KEY1 从 .env 读，模型 deepseek-flash
- Embedding：EMBED_MODEL / EMBED_DIM 从 .env 读，默认 BGE-m3 本地路径 /mnt/d/models/models/BAAI--bge-m3/snapshots/master、1024 维
- PDF 解析：PyMuPDF（正文，import pymupdf）、pdfplumber（表格），版本见 requirements.txt
- MinerU 集成：
  - 装在 ~/mineru_venv（独立 venv，不进主项目）
  - 走远程 API 解析（--remote），本地无 GPU
  - 转换脚本 scripts/mineru_to_ingest.py（MinerU JSON → ingest 格式）
  - 独立 collection：hypertension_guide_mineru（主知识库 hypertension_guide 不受影响）
  - 主项目代码零改动（文件系统交互 + 独立 collection）

## 代码约定
- 单文件 ≤ 300 行，超了就拆
- 技术文档引用代码只写文件名 + 函数名，不写行号
- db.py 的 ORM 类 ChatSession（避免和 sqlalchemy.orm.Session 同名），对应表 sessions

## 已知坑
- 不要 str(make_url(url))，密码会被掩码成 ***；用 render_as_string(hide_password=False)
- ingest.py 的 CHUNK_OVERLAP 只作用于超长单句硬切分支，正常路径无重叠
- similarity 字段双语义：rerank 成功是 sigmoid(logits)，降级是 1-cosine

## 测试
- pytest tests/ 默认跑 49 个单测
- pytest -m slow 跑 e2e（6 个）+ DB 测试（1 个），需真实 DeepSeek/Redis/MySQL
- 新增测试默认标 @pytest.mark.slow（如果需要外部依赖）

## 当前批次进度
- B1 ✅ 后处理 + 混合检索
- B2 ✅ 单测 + e2e
- B3.1 ✅ MySQL 接入
- B3.2 🔄 下一步：多角色切换
