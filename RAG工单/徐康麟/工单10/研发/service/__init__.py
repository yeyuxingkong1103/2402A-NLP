# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG 项目-金融问答系统部署

容器对外服务包（``研发/service``）：

* :mod:`service.config`       —— 容器部署层配置（全部阈值/端口/路径均环境变量可覆盖）
* :mod:`service.session_store` —— 会话共享存储（Redis 优先 + 文件卷兜底，容器间共享）
* :mod:`service.api`          —— FastAPI 正式对外 API（``POST /api/v1/ask``、``GET /health``）

设计约束：业务逻辑一律复用 ``研发/app``（工单7 基座），本包**不复制**任何 RAG 逻辑。
"""

from __future__ import annotations

__all__ = ["api", "config", "session_store"]
