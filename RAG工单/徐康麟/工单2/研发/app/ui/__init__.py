"""网页界面子包：Streamlit 前端与标准库备用界面。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：研发 / 接口层

- ``streamlit_app.py``：真正的 Streamlit 应用（算力云部署用；本机无 streamlit 依赖）；
- ``serve_fallback.py``：纯标准库 ``http.server`` 备用界面（本机演示与在线测试用）。
两者共用同一套 ``app/core`` 业务逻辑与同一份事件契约。
"""
