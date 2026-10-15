# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-金融问答系统部署
# 模块职责：FastAPI HTTP 服务入口——提供健康检查、问答、检索与统计接口，供容器对外提供服务
# 启动方式：uvicorn api_server:app --host 0.0.0.0 --port 8000（容器内由 ENTRYPOINT 调用）

import os                                                       # 导入 os 用于路径注册
import sys                                                      # 导入 sys 用于模块搜索路径

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 把应用目录加入模块搜索路径

from typing import Optional                                     # 导入可选类型用于请求体字段
from fastapi import FastAPI, HTTPException                      # 导入 FastAPI 框架与异常类型
from pydantic import BaseModel, Field                           # 导入数据校验模型与字段定义
import config                                                   # 导入本工单配置模块
from logging_setup import init_logging, log_startup_banner      # 导入日志初始化函数
from qa_service import QAService                                # 导入问答服务

init_logging()                                                  # 初始化日志（控制台 + 文件）
log_startup_banner()                                            # 输出启动配置摘要

app = FastAPI(title="金融问答服务",                              # 创建应用并设置标题
              version=config.APP_VERSION,                       # 设置版本
              description="RAG 金融文档问答系统部署服务")          # 设置描述

SERVICE = QAService()                                           # 全局问答服务实例（进程内单例）


class AskRequest(BaseModel):                                    # 定义问答请求体模型
    question: str = Field(..., description="用户问题", min_length=1)  # 问题文本，必填且非空
    mode: str = Field("hybrid", description="检索模式：hybrid / bm25 / vector")  # 检索模式
    top_k: Optional[int] = Field(None, description="返回上下文条数，默认取服务配置")  # 可选条数


class RetrieveRequest(BaseModel):                               # 定义检索请求体模型
    question: str = Field(..., description="用户问题", min_length=1)  # 问题文本
    mode: str = Field("hybrid", description="检索模式：hybrid / bm25 / vector")  # 检索模式
    top_k: Optional[int] = Field(None, description="返回条数")      # 可选条数


@app.on_event("startup")                                        # 注册启动事件回调
def on_startup():                                               # 定义启动回调
    """服务启动时预热索引，避免首个请求等待过久。"""
    try:                                                        # 尝试预热
        SERVICE.build()                                         # 构建或加载索引
    except Exception as e:                                      # 预热失败
        from logging_setup import LOGGER                        # 延迟导入日志对象
        LOGGER.error("索引预热失败，服务仍将启动（请求时重试）：%r", e)  # 记录错误但不阻止启动


@app.get("/health", summary="健康检查")                          # 注册健康检查接口
def health():                                                   # 定义健康检查处理函数
    """返回服务健康状态与就绪标记，供 docker healthcheck 与容器编排探针使用。"""
    return {"status": "ok",                                     # 服务存活状态
            "ready": SERVICE.ready,                             # 索引是否就绪
            "version": config.APP_VERSION}                      # 服务版本


@app.get("/stats", summary="运行统计")                           # 注册统计接口
def stats():                                                    # 定义统计处理函数
    """返回索引规模、向量开关与历史问答数等运行指标。"""
    SERVICE.build()                                             # 确保索引就绪（幂等）
    return SERVICE.stats()                                      # 返回统计信息


@app.post("/ask", summary="金融问答")                            # 注册问答接口
def ask(req: AskRequest):                                       # 定义问答处理函数
    """执行检索增强问答，返回答案、引用来源与分阶段耗时。"""
    SERVICE.build()                                             # 确保索引就绪
    if req.mode not in ("hybrid", "bm25", "vector"):            # 校验检索模式
        raise HTTPException(status_code=400, detail="mode 必须为 hybrid / bm25 / vector")  # 返回 400
    try:                                                        # 尝试执行问答
        return SERVICE.answer(req.question, mode=req.mode, top_k=req.top_k)  # 返回问答结果
    except Exception as e:                                      # 捕获异常
        from logging_setup import LOGGER                        # 延迟导入日志
        LOGGER.error("问答接口异常：%r", e)                      # 记录错误
        raise HTTPException(status_code=500, detail="服务内部错误：%s" % e)  # 返回 500 错误


@app.post("/retrieve", summary="仅检索")                         # 注册检索接口
def retrieve(req: RetrieveRequest):                             # 定义检索处理函数
    """只执行检索不调用大模型，便于压测与评估检索性能。"""
    SERVICE.build()                                             # 确保索引就绪
    try:                                                        # 尝试检索
        chunks, cost = SERVICE.retrieve(req.question, mode=req.mode, top_k=req.top_k)  # 执行检索
        return {"question": req.question,                       # 回显问题
                "retrieve_seconds": cost,                       # 检索耗时
                "count": len(chunks),                           # 命中条数
                "contexts": [{"source": c["source"], "page": c["page"],  # 来源与页码
                              "id": c["id"], "text": c["text"]}  # 块 ID 与正文
                             for c in chunks]}                  # 组装结果列表
    except Exception as e:                                      # 捕获异常
        from logging_setup import LOGGER                        # 延迟导入日志
        LOGGER.error("检索接口异常：%r", e)                      # 记录错误
        raise HTTPException(status_code=500, detail="检索失败：%s" % e)  # 返回 500 错误


if __name__ == "__main__":                                      # 兼容直接以 python api_server.py 启动
    import uvicorn                                              # 导入 ASGI 服务器
    uvicorn.run(app, host=config.SERVICE_HOST,                  # 运行应用
                port=config.SERVICE_PORT)                       # 使用配置端口
