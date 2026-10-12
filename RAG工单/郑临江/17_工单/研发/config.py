# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-解决API服务并发瓶颈与资源泄漏工单
配置：并发参数、连接池、限流、资源限制。
"""
import os

# 接口
API_HOST = "0.0.0.0"
API_PORT = 8000

# 线程池与队列（异步处理检索+生成，避免 API Server 阻塞）
MAX_WORKERS = 16          # 工作线程数
QUEUE_SIZE = 100          # 内存请求队列长度
RATE_LIMIT_PER_SEC = 50   # 限流：每秒最大请求数

# 连接池（数据库/向量库，避免连接泄漏）
DB_POOL_SIZE = 10
VECTOR_POOL_SIZE = 10

# 重量级组件单例缓存开关（避免每请求重复初始化 DeepDoc / VLM）
SINGLETON_PARSER = True
SINGLETON_VLM = True

# Docker 资源限制（模拟生产约束，提前暴露问题）
CONTAINER_MEMORY = "8g"
CONTAINER_CPUS = "4.0"

# 目标验收指标
P95_TARGET = 3.0       # 场景A P95 ≤3s
MEMORY_GROWTH_MAX = 0.10   # 容器内存波动 10% 以内
LEAK_GROWTH_MAX = 0.20     # 12 小时 RSS 增长 ≤20%
