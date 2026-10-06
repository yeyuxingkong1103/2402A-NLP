# -*- coding: utf-8 -*-
"""轻量指标框架（纯标准库，零第三方依赖）。

设计目标
--------
日志回答「发生了什么」，指标回答「有多快、多稳」。本模块给整条 RAG 链路
（入库 / 嵌入 / 检索 / 生成 / 服务）提供统一的度量口径：

* ``Counter``   —— 只增不减的计数（请求数、失败数、缓存命中数）；
* ``Gauge``     —— 可增可减的瞬时值（并发中的请求数、队列长度）；
* ``Histogram`` —— 观测值分布，能算 count / sum / min / max / **P50 / P95 / P99**。

命名规范（务必遵守，t2/t3/t4/t6 注册新指标时也照此办）::

    <域>_<对象>_<单位>

    llm    _ tokens   _ per_second
    embed  _ seconds
    vector _ hits

单位统一用基本单位：秒写 ``seconds``，字节写 ``bytes``，不要写 ms / MB。

**Counter 命名约定（唯一口径，务必遵守 —— 这是本模块曾经出过的真实 bug）**：

* **注册名必须自带 ``_total`` 后缀**（如 ``docs_total`` / ``requests_total`` / ``llm_error_total``）；
  仓库现有 **14** 个 counter（见 :data:`METRIC_CATALOG`）**全部**符合此约定。
* :func:`render_prometheus` 渲染时按 ``name if name.endswith("_total") else f"{name}_total"``
  **归一化**：已带后缀的原样保留，**不会**再追加一次。因此
  **TYPE/HELP 行里的名字与真实样本名永远一致**。
* 即：注册 ``docs_total`` → 样本名就是 ``docs_total``（**不是** ``docs_total_total``）。

> 历史坑（已修）：早期 docstring 写「注册时写 ``requests_total`` 即可，`_total` 由渲染器自动补全」，
> 而渲染器当时对**所有** counter 无条件追加 ``_total`` —— 两处口径相反，导致
> ``# TYPE docs_total counter`` 挂在了真实样本 ``docs_total_total`` 上，
> Prometheus 抓到的序列**退化为 untyped**，元数据失效。现约定见上，实现见 `_prometheus_counter_name()`。

**累计 vs 存量（读数约定，避免误读）**：

* **Counter = 累计量，且是「本次进程累计」—— 进程重启即归零**，
  既不是存量、也不是跨重启的持久累计。所有 counter 的 HELP 都显式标注该口径。
* **Gauge = 当前存量**（如 ``vector_store_rows``，取自 ``store.count()``，会随删除下降）。
* 因此同屏看到 ``docs_total=1`` 与 ``vector_store_rows=18`` 并不矛盾：
  前者是「本进程处理过 1 篇」，后者是「库里现有 18 行」。**判断数据量一律看 Gauge 存量。**

两种输出
--------
1. :func:`emit_metrics` —— 打一条 ``METRIC {json}`` 行 + 一条人类可读摘要行，
   便于 ``grep METRIC``、ELK/Loki 采集；
2. :func:`render_prometheus` —— 返回 Prometheus 文本格式，供 ``GET /metrics``
   （t4 负责挂路由），为「Prometheus + Grafana」铺路。

采集点由各业务模块自己调用 ``counter("...").inc()`` / ``histogram("...").observe(x)``。
本模块**绝不吞掉业务异常**：指标自身出错只记一条 warning 后返回兜底对象。

开关：``METRICS_LOG_ENABLED``（默认 true）。置 0/false 时只保留内存登记，
不打印任何 METRIC 日志行（Prometheus 文本导出不受影响）。
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "Counter", "Gauge", "Histogram", "MetricRegistry", "MetricsSnapshot",
    "registry", "metrics_enabled", "counter", "gauge", "histogram",
    "observe", "inc", "dec", "set_gauge", "snapshot", "reset_metrics", "clear_metrics",
    "render_prometheus", "emit_metrics", "emit_request_metrics",
    "flush_metrics", "summarize_metrics",
]

_LOGGER = logging.getLogger(__name__)

# 度量名 -> (Prometheus 类型, 一句话说明, 单位)
METRIC_CATALOG: dict[str, tuple[str, str, str]] = {
    # ---------- B-9（r16 §20）对话存储分层：长期记忆写入 / Redis 窗口裁剪 ----------
    # 口径：长期记忆 = Milvus `legal_rag_memory`，只收录**滚出 5 组上下文窗口**的旧轮次。
    # 写失败必须可见（`AC-ST-9`）：status=failed 时调用方**不得**裁剪 Redis。
    "longterm_write_total": (
        "counter",
        "长期记忆写入次数（按结果与 collection 分标签；status=failed 时调用方跳过 Redis 裁剪）",
        "count",
    ),
    "longterm_skip_total": (
        "counter",
        "长期记忆未写的原因（reason=disabled 未启用 / no_rollout 本轮没有滚出窗口的轮次）",
        "count",
    ),
    "session_trim_total": (
        "counter",
        "Redis 上下文窗口按「组」裁剪完成次数（status=trimmed）",
        "count",
    ),
    "session_trim_skipped_total": (
        "counter",
        "Redis 裁剪被跳过的次数（reason=longterm_failed：写长期记忆失败 ⇒ 暂不裁剪，下轮补写）",
        "count",
    ),
    # ---------- B-9 长期记忆**召回**（读路径；与上面的"写路径"分开计量）----------
    # 口径：`LONGTERM_RECALL_ENABLED=true` 时 `/chat` 才会召回；status=disabled 说明没开。
    # **写进去 ≠ 想得起来**：只测写入会得到"看起来有记忆、实际用不上"的假象（实机踩过）。
    "longterm_recall_total": (
        "counter",
        "长期记忆召回次数（status=ok 命中并注入 / empty 无命中 / failed 召回异常 / disabled 未开启）",
        "count",
    ),
    "longterm_recall_hits_total": (
        "counter",
        "长期记忆召回**命中条数**累计（只统计真正注入提示词的那些）",
        "count",
    ),
    # ---------- 上传解析（把"看起来成功、实际检索不到"变成可见）----------
    "upload_zero_chunks_total": (
        "counter",
        "上传文件**抽出 0 个分块**的次数（疑似扫描件/纯图片 PDF 或空文件；"
        "当前只做文本层抽取、不支持 OCR）—— 这类文件状态是 completed 但**检索命不中**",
        "count",
    ),
    # ---------- 关系库历史结构（内容寻址 message_id 的已知副作用）----------
    "orphan_assistant_message_total": (
        "counter",
        "同一会话重问同一问题时出现「孤儿助手行」的次数（用户消息按幂等被跳过、"
        "助手消息却新增）；长期记忆侧已按问题去重，这里只做可见化",
        "count",
    ),
    # ---------- LLM 生成（t3 采集）----------
    "llm_ttft_seconds": ("histogram", "首 token 延迟（time to first token）", "seconds"),
    # 下面 3 条的 HELP **不绑定单一后端**：Ollama 原生路径来自 prompt_eval_count /
    # eval_count / eval_duration；OpenAI 兼容路径（generate/openai_compat.py）来自
    # usage.prompt_tokens / usage.completion_tokens（或客户端 chunk 计数）与
    # total-ttft 导出量。写死 eval_* 会让 /metrics 在 openai 路径上指向错误来源。
    "llm_tokens_per_second": (
        "histogram",
        "生成速度（标签 mode=official 为解码阶段速度、mode=wall 为整段吞吐；"
        "wall 的分母按 provider 不同，见 generate/llm_metrics.py 模块 docstring）",
        "tokens/second",
    ),
    "llm_prompt_tokens": (
        "histogram",
        "提示词 token 数（Ollama: prompt_eval_count；OpenAI 兼容: usage.prompt_tokens）",
        "tokens",
    ),
    "llm_output_tokens": (
        "histogram",
        "生成 token 数（Ollama: eval_count；OpenAI 兼容: usage.completion_tokens，"
        "无 usage 时可能退化为客户端内容 chunk 计数）",
        "tokens",
    ),
    "llm_prefill_seconds": (
        "histogram",
        "预填充耗时（Ollama: prompt_eval_duration；OpenAI 兼容协议不上报此项）",
        "seconds",
    ),
    "llm_decode_seconds": (
        "histogram",
        "解码耗时（Ollama: eval_duration；OpenAI 兼容: total-ttft 导出量）",
        "seconds",
    ),
    "llm_load_seconds": (
        "histogram",
        "模型载入耗时（load_duration）：把模型权重加载进显存/内存的耗时，不是请求延迟；"
        "实测冷启动 CPU 约 44.7s / 热态约 0.12s，GPU(RTX 2060) 冷启动约 3.5s / 热态约 0.005s；"
        "冷启动诊断专用，勿与 llm_total_seconds 混读",
        "seconds",
    ),
    "llm_total_seconds": ("histogram", "一次生成的总耗时", "seconds"),
    "llm_degraded_total": ("counter", "降级到兜底回答的次数", "count"),
    "llm_timeout_total": ("counter", "大模型调用超时次数", "count"),
    "llm_error_total": ("counter", "大模型调用失败次数", "count"),
    "llm_failure_total": (
        "counter",
        "大模型调用失败次数（标签 provider/model/reason；reason 用**可定位**新口径："
        "missing_api_key 没配 key / connect_failed 连不上 / http_5xx 服务端故障 / "
        "http_4xx 请求被拒 / timeout 超时 / model_missing 模型名不对 / auth 鉴权失败 / "
        "rate_limit 限流 / bad_request 参数被拒 / bad_response 响应不可解析 / unknown 未知）。"
        "与 llm_error_total 的分工：后者保留**旧口径** reason（connect/http/timeout…，"
        "有测试钉住），本指标用新口径并额外带 provider+model，便于一眼定位哪条后端哪一类故障",
        "count",
    ),
    # ---------- 检索（t2 采集）----------
    "embed_seconds": ("histogram", "一次嵌入调用的耗时", "seconds"),
    "embed_batch_size": ("histogram", "单次嵌入的文本条数", "items"),
    "embed_cache_hit_total": ("counter", "嵌入缓存命中次数", "count"),
    "embed_cache_miss_total": ("counter", "嵌入缓存未命中次数", "count"),
    "embedding_dim": ("gauge", "当前嵌入模型输出维度", "dim"),
    "vector_hits": ("histogram", "向量检索命中条数", "items"),
    "bm25_hits": ("histogram", "关键词检索命中条数", "items"),
    "fused_hits": ("histogram", "融合后候选条数", "items"),
    "rerank_seconds": ("histogram", "重排耗时", "seconds"),
    "top1_score": ("histogram", "最终首条结果的相关性分数", "score"),
    # ---------- 相关性闸门（t109 落地 / t115 重标定）----------
    # 口径（t115 改口径，旧口径作废）：**词面证据是准入门槛、融合分只在门槛内做补充**。
    #   判据①（尺度无关）：问题的实义词元（长度>=2 的双字/英文词）至少
    #     ``retrieve.hybrid::_MIN_CONTENT_TERMS`` = 3 个落在同一条碎片正文里；
    #   判据②（尺度相关）：融合分 >= ``RetrievalConfig.relevance_gate``（上限 = 两通道权重和 2.0），
    #     仅用于在①已过的前提下补充保留高分碎片。
    # 两条判据的分工由实测决定：`劳动争议仲裁的时效是多久？` 候选融合分只有 0.9577~1.0000
    # （关键词通道零命中）却共享 6~8 个词元 —— 只看分数必然误杀；`推荐几部好看的科幻电影`
    # 融合分 1.8977（靠单字命中）却只共享 2 个词元 —— 只看分数必然伪依据。
    "retrieval_gate_dropped_total": (
        "counter",
        "被相关性闸门丢弃的碎片数（reason=no_lexical_evidence 本批候选与问题没有任何实质词面"
        "关系，整批丢 / below_evidence 门槛已过但该条既无词面证据、分数也不达标；"
        "被丢的碎片不进上下文、也不进引用）",
        "count",
    ),
    "retrieval_gate_mode_total": (
        "counter",
        "相关性闸门**本轮用的是哪套判据**（mode=score_and_lexical 分数尺度可用：词面证据为准入、"
        "融合分为补充 / lexical_only 尺度不可比（关键词通道未就绪或降级）：只执行尺度无关的"
        "词面证据判据 / skip 阈值关闭或后端未标定；reason=ok|channel_degraded|disabled|"
        "uncalibrated_backend；store=实际后端）",
        "count",
    ),
    "retrieval_gate_max_overlap": (
        "histogram",
        "单次召回里「问题的实义词元最多落进同一条碎片多少个」——准入门槛"
        "（retrieve.hybrid::_MIN_CONTENT_TERMS）就是按它的实测分布标定的",
        "terms",
    ),
    "chat_route_total": (
        "counter",
        "三分类路由的命中次数（route=legal 有依据走法律路径 / clarify 没依据但有法律信号，"
        "引导补充或换说法 / general 无可信依据且无法律信号，走通用对话：不注入资料、"
        "引用恒为空、不引法规）。三者之和 = 本轮问过的次数",
        "count",
    ),
    "retrieval_gate_skipped_total": (
        "counter",
        "本轮**不执行**闸门的次数（reason=disabled 人工关闭 / uncalibrated_backend 内存·离线"
        "后端没有标定的分数尺度；两者都保持既有行为，不等于闸门失效。注意 t115 起"
        "关键词通道未就绪**不再**走这里，而是走 lexical_only 判据）",
        "count",
    ),
    "retrieval_warmup_notice_total": (
        "counter",
        "答案里带了「资料还在预热、结果可能不全」可见提示的次数（内容见 "
        "retrieve/hybrid.py::WARMUP_NOTE；标签 store=实际后端）。>0 说明服务刚启动、"
        "关键词通道还没就绪，此时**不得**给伪依据（引用为空或只给向量通道命中的依据）",
        "count",
    ),
    "context_chars": ("histogram", "拼进提示词的上下文字符数", "chars"),
    "law_scope_recall_total": (
        "counter",
        "法名定向召回命中次数（标签 law=识别出的法名）。评测 v1/v2 的失分主因是"
        "「该法条没进候选池」，这一路专门去该法的文件里补候选（只加不删），"
        "见 retrieve/law_scope.py；=0 说明问题里没点法名或索引没建起来",
        "count",
    ),
    "law_scope_hits": ("histogram", "法名定向召回补进来的候选条数", "count"),
    "keyword_index_cache_total": (
        "counter",
        "BM25 索引落盘缓存的读写结果（标签 result=hit|miss|written|corrupt|write_failed）。"
        "真机踩坑：不缓存时每次进程重启都要重建 11.9 万块（约 160s），期间关键词通道为空、"
        "用户会看到「资料还在预热」；corrupt/miss 偏多说明缓存被误删或签名总在变",
        "count",
    ),
    "version_filter_dropped_total": (
        "counter",
        "版本过滤剔掉的候选条数（同一文件的多个版本里只留现行版，见 "
        "retrieve/version_filter.py）。真机发现语料里 411 个 __hash 副本是**旧版本**、"
        "与基名文件没有一对字节相同；>0 说明确实拦住了旧条文",
        "count",
    ),
    "retrieval_source_quota_total": (
        "counter",
        "来源配额挤掉的候选条数（同一来源文件最多留 N 条，见 "
        "retrieve/hybrid.py::apply_source_quota）。真机发现 L23 的 top-3 是同一份批复的"
        "3 个拷贝、把名额吃光；>0 说明确实拦住了重复文档",
        "count",
    ),
    # ---------- 入库（t2/t5 采集）----------
    "docs_total": ("counter", "入库文档总数", "count"),
    "chunks_total": ("counter", "入库分块总数", "count"),
    "parse_seconds": ("histogram", "单篇文档解析耗时", "seconds"),
    "chunk_seconds": ("histogram", "单篇文档分块耗时", "seconds"),
    "upsert_seconds": ("histogram", "写入向量库耗时", "seconds"),
    "ingest_failed_total": ("counter", "入库失败次数", "count"),
    "dim_mismatch_total": ("counter", "向量维度不匹配次数", "count"),
    "vector_store_rows": (
        "gauge",
        "向量库当前活条数（存量，取自 store.count()，标签 store=memory|milvus）；"
        "会随删除下降，**不是** num_entities / row_count（它们含 upsert 墓碑，会明显虚报）",
        "rows",
    ),
    # ---------- HTTP 服务（t4 采集）----------
    "requests_total": ("counter", "HTTP 请求数（标签 path/status）", "count"),
    "request_seconds": ("histogram", "HTTP 请求耗时（标签 path）", "seconds"),
    "inflight_requests": ("gauge", "正在处理中的请求数", "requests"),
    "queue_wait_seconds": ("histogram", "排队等待耗时", "seconds"),
    "requests_rejected_total": (
        "counter",
        "被限流拒绝的请求数（排队超时，标签 reason=queue_full）",
        "count",
    ),
    "upload_rejected_total": (
        "counter",
        "被拒的上传数（标签 reason=类型/大小/数量等分类，由 uploads.validate_upload 给出）",
        "count",
    ),
    "upload_files_total": (
        "counter",
        "成功上传的文件数（标签 suffix=文件后缀）",
        "count",
    ),
    "upload_bytes_total": (
        "counter",
        "成功上传的字节数（标签 suffix=文件后缀）",
        "bytes",
    ),
    # ---------- 静默缺口的可定位性（t8 诊断 + t10 三处对账）----------
    "vector_recall_degraded_total": (
        "counter",
        "向量召回失败、本次已降级为「仅关键词(BM25)」的次数（标签 store/embedder）；"
        ">0 表示有提问拿不到向量召回（质量下降），日志关键词「【已降级】」",
        "count",
    ),
    "milvus_requery_fallback_total": (
        "counter",
        "search 的字段回查不一致（code=2200 inconsistent requery result）后"
        "退到「两步取回」的次数（标签 collection）",
        "count",
    ),
    "milvus_requery_missing_total": (
        "counter",
        "两步取回里按 id 仍读不回来、被丢弃的命中条数（标签 collection）",
        "count",
    ),
    "milvus_all_chunks_short_total": (
        "counter",
        "all_chunks 实际取到条数少于**权威 count(*)** 的缺口条数（标签 collection）；"
        ">0 表示服务端少给行（BM25 语料/文档删除会受影响），日志关键词「[SHORT]」",
        "count",
    ),
    "milvus_get_fallback_total": (
        "counter",
        "get() 空手而归、靠 query(id ==) 兜底才取到单条的次数（标签 collection）；"
        "父块上下文回填会走这条路",
        "count",
    ),
    "milvus_get_missing_total": (
        "counter",
        "get() 与 query(id ==) 都取不到、最终返回 None 的次数（标签 collection）",
        "count",
    ),
    "milvus_delete_mismatch_total": (
        "counter",
        "按 filter 删除时「匹配 N 条 vs 实际删 M 条」不一致的缺口条数（标签 collection）；"
        ">0 表示可能留下残留分块",
        "count",
    ),
    "documents_delete_mismatch_total": (
        "counter",
        "删除文档时「应删 N 条 vs 实删 M 条」不一致的缺口条数（标签 store）；"
        ">0 表示「删了还能检索到」的残留风险",
        "count",
    ),
    "milvus_varchar_clipped_total": (
        "counter",
        "写入 Milvus 前按 varchar 上限裁剪字段的次数（标签 field）。**上限按 UTF-8 字节算**"
        "（text 8192 / doc_id 256 / source 512 …）；>0 说明分块或标识超限，日志关键词"
        "「VARCHAR 上限裁剪」——2026-09-17 曾因此让 13 篇法规整篇不入库",
        "count",
    ),
    # ---------- 系统层（t6 采集，本任务只登记口径）----------
    "sys_cpu_percent": ("gauge", "进程/整机 CPU 使用率", "percent"),
    "sys_memory_percent": ("gauge", "内存使用率", "percent"),
    "sys_memory_used_bytes": ("gauge", "已用内存", "bytes"),
    "sys_disk_used_percent": ("gauge", "磁盘使用率", "percent"),
    "sys_load1": ("gauge", "1 分钟平均负载", "load"),
    "sys_gpu_util_percent": ("gauge", "GPU 利用率", "percent"),
    "sys_gpu_memory_used_bytes": ("gauge", "GPU 显存占用", "bytes"),
    "sys_ollama_up": ("gauge", "Ollama 健康（1/0）", "bool"),
    "sys_redis_up": ("gauge", "Redis 健康（1/0）", "bool"),
    "sys_milvus_up": ("gauge", "Milvus 健康（1/0）", "bool"),
    # ---------- P0.5 耗时评估：函数级耗时（`observability.timed` 采集）----------
    # 口径：**标签 `func` 是函数的 qualname（低基数，约百级）**，不要放 session_id 之类
    # 高基数维度，否则 Prometheus 序列会爆炸（与本文件「标签基数」约定、以及
    # ARCHITECTURE.md §7 的「路径模板」约定一致）。
    # 与日志的分工：这一项**永不被日志级别过滤**，所以它是"速度评估"的可信数据源；
    # 而 `[SLOW]` 行只是给人看的清单（见 observability.timed 的 docstring）。
    "func_seconds": (
        "histogram",
        "函数/代码块耗时（标签 func=qualname）。由 observability.timed / timed_scope 采集；"
        "关掉日志级别也照样统计，用于模块与函数的性能回归对比",
        "seconds",
    ),
}

# 默认延迟分桶（秒）：覆盖 10ms ~ 32s，适配 CPU 场景上秒级的 LLM 调用
DEFAULT_BUCKETS: tuple[float, ...] = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0,
)

#: **函数/代码块耗时**专用分桶（秒）：**10µs ~ 64s**。
#:
#: 为什么不能复用 :data:`DEFAULT_BUCKETS`（实测发现）：那套桶从 5ms 起跳，于
#: ``func_seconds`` 这种"既有 0.2ms 的纯计算函数、又有 30s 的全量入库"的分布上
#: **分辨率极差** —— 一次 30ms 的调用会落进 ``le=0.25`` 桶，P50/P95 全部塌到同一档。
#:
#: 为什么还要**亚毫秒档**（10µs/25µs/50µs/100µs/250µs/500µs）：实测发现纯计算类
#: 函数（如 ``demo.beta``，真值 ~12µs）会**全部落进第一个桶**，此时
#: Prometheus 侧的桶插值只能给出"≤ 首桶上界"的上限值（1ms），与真值差两个数量级。
#: 补上亚毫秒档后，这类函数也能被分辨出来。
FUNC_DURATION_BUCKETS: tuple[float, ...] = (
    0.00001, 0.000025, 0.00005, 0.0001, 0.00025, 0.0005,
    0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
    1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0,
)

#: 指标名 → 专用分桶。未列出的指标走 :data:`DEFAULT_BUCKETS`（或 ``METRICS_BUCKETS``）。
BUCKET_OVERRIDES: dict[str, tuple[float, ...]] = {
    "func_seconds": FUNC_DURATION_BUCKETS,
}

_TRUTHY = {"1", "true", "yes", "on", "y", "t"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in _TRUTHY


def _env_buckets(name: str = "METRICS_BUCKETS") -> tuple[float, ...]:
    raw = os.environ.get(name, "")
    if not raw:
        return DEFAULT_BUCKETS
    values: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            continue
    return tuple(sorted(values)) or DEFAULT_BUCKETS


def metrics_enabled() -> bool:
    """METRICS_LOG_ENABLED：是否打印 METRIC 日志行（默认 true）。"""
    return _env_bool("METRICS_LOG_ENABLED", True)


def _label_key(labels: Mapping[str, Any] | None) -> tuple[tuple[str, str], ...]:
    if not labels:
        return ()
    return tuple(sorted((str(k), str(v)) for k, v in labels.items() if v is not None))


def _format_labels(labels: Mapping[str, Any] | None) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{k}="{_escape_label(str(v))}"' for k, v in labels.items())
    return "{" + inner + "}"


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """线性插值分位数（与 numpy.percentile 的 linear 方式一致）。"""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = q * (len(sorted_values) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return float(sorted_values[low])
    weight = rank - low
    return float(sorted_values[low] * (1 - weight) + sorted_values[high] * weight)


def _stats_of(values: Sequence[float]) -> dict[str, float]:
    """对一批观测值算 count/sum/min/max/avg/P50/P95/P99。"""
    ordered = sorted(values)
    total = float(sum(ordered))
    return {
        "count": float(len(ordered)),
        "sum": total,
        "min": float(ordered[0]) if ordered else 0.0,
        "max": float(ordered[-1]) if ordered else 0.0,
        "avg": (total / len(ordered)) if ordered else 0.0,
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
    }


class Counter:
    """只增不减的计数器（线程安全）。"""

    kind = "counter"

    def __init__(self, name: str, description: str = "", unit: str = "count") -> None:
        self.name = name
        self.description = description
        self.unit = unit
        self._values: dict[tuple[tuple[str, str], ...], float] = {}
        self._lock = threading.Lock()

    def inc(self, value: float = 1.0, **labels: Any) -> float:
        key = _label_key(labels)
        with self._lock:
            total = self._values.get(key, 0.0) + float(value)
            self._values[key] = total
            return total

    add = inc

    def value(self, **labels: Any) -> float:
        return self._values.get(_label_key(labels), 0.0)

    def samples(self) -> list[tuple[dict[str, str], float]]:
        with self._lock:
            return [
                (dict(key), value) for key, value in sorted(self._values.items()) if value
            ] or [({}, 0.0)]

    def snapshot(self) -> dict[str, Any]:
        return {
            "type": "counter",
            "unit": self.unit,
            "description": self.description,
            "value": self.samples()[0][1] if not self._values else sum(self._values.values()),
            "samples": [
                {"labels": labels, "value": value} for labels, value in self.samples()
            ],
        }

    def reset(self) -> None:
        with self._lock:
            self._values.clear()


class Gauge:
    """可增可减的瞬时值（线程安全），支持带标签的多个取值。"""

    kind = "gauge"

    def __init__(self, name: str, description: str = "", unit: str = "count") -> None:
        self.name = name
        self.description = description
        self.unit = unit
        self._values: dict[tuple[tuple[str, str], ...], float] = {}
        self._lock = threading.Lock()

    def set(self, value: float, **labels: Any) -> float:
        key = _label_key(labels)
        with self._lock:
            self._values[key] = float(value)
            return float(value)

    # 兼容 metrics.Gauge(...).set_value() 写法
    set_value = set

    def inc(self, value: float = 1.0, **labels: Any) -> float:
        key = _label_key(labels)
        with self._lock:
            current = self._values.get(key, 0.0) + float(value)
            self._values[key] = current
            return current

    def dec(self, value: float = 1.0, **labels: Any) -> float:
        return self.inc(-float(value), **labels)

    def value(self, **labels: Any) -> float:
        return self._values.get(_label_key(labels), 0.0)

    def samples(self) -> list[tuple[dict[str, str], float]]:
        with self._lock:
            return [
                (dict(key), value) for key, value in sorted(self._values.items())
            ] or [({}, 0.0)]

    def snapshot(self) -> dict[str, Any]:
        return {
            "type": "gauge",
            "unit": self.unit,
            "description": self.description,
            "value": self.samples()[0][1],
            "samples": [
                {"labels": labels, "value": value} for labels, value in self.samples()
            ],
        }

    def reset(self) -> None:
        with self._lock:
            self._values.clear()


class Histogram:
    """观测值分布：count / sum / min / max + P50/P95/P99 + 可配置分桶。"""

    kind = "histogram"

    def __init__(
        self,
        name: str,
        description: str = "",
        unit: str = "",
        buckets: Iterable[float] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.unit = unit
        self.buckets: tuple[float, ...] = tuple(sorted(buckets)) if buckets else _env_buckets()
        self._values: dict[tuple[tuple[str, str], ...], list[float]] = {}
        self._lock = threading.Lock()

    def observe(self, value: float, **labels: Any) -> None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return
        if number != number:  # NaN
            return
        key = _label_key(labels)
        with self._lock:
            self._values.setdefault(key, []).append(number)

    record = observe

    def count(self, **labels: Any) -> int:
        return len(self._values.get(_label_key(labels), ()))

    def sum(self, **labels: Any) -> float:
        return float(sum(self._values.get(_label_key(labels), ())))

    def min(self, **labels: Any) -> float:
        values = self._values.get(_label_key(labels), ())
        return float(min(values)) if values else 0.0

    def max(self, **labels: Any) -> float:
        values = self._values.get(_label_key(labels), ())
        return float(max(values)) if values else 0.0

    def percentile(self, q: float, **labels: Any) -> float:
        values = sorted(self._values.get(_label_key(labels), ()))
        return _percentile(values, q)

    def p50(self, **labels: Any) -> float:
        return self.percentile(0.50, **labels)

    def p95(self, **labels: Any) -> float:
        return self.percentile(0.95, **labels)

    def p99(self, **labels: Any) -> float:
        return self.percentile(0.99, **labels)

    def stats(self, **labels: Any) -> dict[str, float]:
        return _stats_of(self._values.get(_label_key(labels), ()))

    def value(self, **labels: Any) -> float:
        """默认取值：没有样本返回 0，否则返回平均值。"""
        return self.stats(**labels)["avg"]

    def bucket_counts(self, **labels: Any) -> list[tuple[float, int]]:
        values = self._values.get(_label_key(labels), ())
        result: list[tuple[float, int]] = []
        cumulative = 0
        for bound in self.buckets:
            cumulative = sum(1 for v in values if v <= bound)
            result.append((bound, cumulative))
        return result

    def samples(self) -> list[tuple[dict[str, str], dict[str, float]]]:
        with self._lock:
            keys = sorted(self._values.keys())
        return [(dict(key), self.stats(**dict(key))) for key in keys]

    def snapshot(self) -> dict[str, Any]:
        """快照含 count/sum/min/max/P50/P95/P99，并附带分桶累计值（供 Prometheus 还原 le）。"""
        with self._lock:
            keys = sorted(self._values.keys())
            raw_by_key = {key: list(self._values[key]) for key in keys}
        detail: list[dict[str, Any]] = []
        for key in keys:
            raw = raw_by_key[key]
            stats = _stats_of(raw)
            buckets = [(bound, sum(1 for v in raw if v <= bound)) for bound in self.buckets]
            detail.append({"labels": dict(key), "buckets": buckets, **stats})
        if not detail:
            stats = self.stats()
            buckets = [(bound, 0) for bound in self.buckets]
            detail.append({"labels": {}, "buckets": buckets, **stats})
        first = detail[0]
        return {
            "type": "histogram",
            "unit": self.unit,
            "description": self.description,
            "value": first["avg"],
            "count": first["count"],
            "sum": first["sum"],
            "min": first["min"],
            "max": first["max"],
            "p50": first["p50"],
            "p95": first["p95"],
            "p99": first["p99"],
            "samples": detail,
        }

    def reset(self) -> None:
        with self._lock:
            self._values.clear()


class MetricsSnapshot:
    """某一时刻的全部指标快照，便于一次性写日志/断言/导出。"""

    def __init__(self, payload: Mapping[str, Any], timestamp: float | None = None) -> None:
        self.metrics: dict[str, Any] = dict(payload)
        self.timestamp = time.time() if timestamp is None else timestamp

    def to_dict(self) -> dict[str, Any]:
        return {"ts": round(self.timestamp, 6), "metrics": self.metrics}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, default=str)

    def values(self) -> dict[str, float]:
        """扁平化为 ``名称[标签]`` -> 数值，方便断言与摘要打印。"""
        flat: dict[str, float] = {}
        for name, payload in self.metrics.items():
            kind = payload.get("type")
            for sample in payload.get("samples") or [{}]:
                labels = sample.get("labels") or {}
                if kind == "histogram" and "avg" in sample:
                    value = float(sample.get("avg") or 0.0)
                else:
                    value = float(sample.get("value", 0.0) or 0.0)
                if not labels:
                    flat[name] = value
                suffix = "{" + ",".join(f"{k}={v}" for k, v in labels.items()) + "}"
                flat[f"{name}{suffix}"] = value
        return flat

    def __contains__(self, name: str) -> bool:
        return name in self.metrics


class MetricRegistry:
    """指标注册表：按名字取（或建）Counter / Gauge / Histogram，线程安全。"""

    def __init__(self) -> None:
        self._metrics: dict[str, Counter | Gauge | Histogram] = {}
        self._lock = threading.RLock()

    # ---------- 注册 ----------
    def counter(self, name: str, description: str = "", unit: str = "count") -> Counter:
        return self._get(name, Counter, description, unit)  # type: ignore[return-value]

    def gauge(self, name: str, description: str = "", unit: str = "count") -> Gauge:
        return self._get(name, Gauge, description, unit)  # type: ignore[return-value]

    def histogram(
        self,
        name: str,
        description: str = "",
        unit: str = "",
        buckets: Iterable[float] | None = None,
    ) -> Histogram:
        with self._lock:
            existing = self._metrics.get(name)
            if existing is not None:
                if not isinstance(existing, Histogram):
                    raise TypeError(f"指标 {name!r} 已注册为 {existing.kind}，不能当 histogram 用")
                return existing
            meta = METRIC_CATALOG.get(name)
            hist = Histogram(
                name,
                description or (meta[1] if meta else ""),
                unit or (meta[2] if meta else ""),
                # 显式传入 > 指标专用分桶（BUCKET_OVERRIDES）> 默认/环境变量分桶。
                # 例：`func_seconds` 用 FUNC_DURATION_BUCKETS（1ms 起跳），
                # 否则会落进 DEFAULT_BUCKETS 的 5ms 起步桶里、分位数全部塌档。
                buckets if buckets is not None else BUCKET_OVERRIDES.get(name),
            )
            self._metrics[name] = hist
            return hist

    def _get(self, name: str, cls, description: str, unit: str):
        with self._lock:
            existing = self._metrics.get(name)
            if existing is not None:
                if not isinstance(existing, cls):
                    raise TypeError(f"指标 {name!r} 已注册为 {existing.kind}")
                return existing
            meta = METRIC_CATALOG.get(name)
            metric = cls(name, description or (meta[1] if meta else ""),
                         unit or (meta[2] if meta else "count"))
            self._metrics[name] = metric
            return metric

    # ---------- 读取 ----------
    def get(self, name: str):
        return self._metrics.get(name)

    def names(self) -> list[str]:
        return sorted(self._metrics)

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            payload = {name: metric.snapshot() for name, metric in self._metrics.items()}
        return MetricsSnapshot(payload)

    def reset(self) -> None:
        """清零所有指标数值（保留已注册的名字/描述）。"""
        with self._lock:
            for metric in self._metrics.values():
                metric.reset()

    def clear(self) -> None:
        """注销全部指标（测试隔离用；运行期请用 :meth:`reset`）。"""
        with self._lock:
            self._metrics.clear()


_REGISTRY = MetricRegistry()
_REGISTRY_LOCK = threading.Lock()


def registry() -> MetricRegistry:
    """全局注册表（延迟创建，避免导入期副作用）。"""
    return _REGISTRY


# ---------------- 模块级便捷函数（t2/t3/t4/t6 直接用这些）----------------

def counter(name: str, description: str = "", unit: str = "count") -> Counter:
    return registry().counter(name, description, unit)


def gauge(name: str, description: str = "", unit: str = "count") -> Gauge:
    return registry().gauge(name, description, unit)


def histogram(name: str, description: str = "", unit: str = "",
              buckets: Iterable[float] | None = None) -> Histogram:
    return registry().histogram(name, description, unit, buckets)


def inc(name: str, value: float = 1.0, **labels: Any) -> float:
    return counter(name).inc(value, **labels)


def dec(name: str, value: float = 1.0, **labels: Any) -> float:
    return gauge(name).dec(value, **labels)


def set_gauge(name: str, value: float, **labels: Any) -> float:
    return gauge(name).set(value, **labels)


def observe(name: str, value: float, **labels: Any) -> None:
    histogram(name).observe(value, **labels)


def snapshot() -> MetricsSnapshot:
    return registry().snapshot()


def reset_metrics() -> None:
    """清零所有指标的数值（保留注册）。"""
    registry().reset()


def clear_metrics() -> None:
    """注销全部指标（测试隔离用）。"""
    registry().clear()


# ---------------- 输出 1：日志行 ----------------

def _human_summary(snap: MetricsSnapshot, limit: int = 12) -> str:
    parts: list[str] = []
    for name, payload in snap.metrics.items():
        kind = payload.get("type")
        samples = payload.get("samples") or [{}]
        for sample in samples:
            labels = sample.get("labels") or {}
            tag = f"{name}{_format_labels(labels)}"
            if kind == "histogram":
                average = sample.get("avg", sample.get("value", 0.0))
                parts.append(
                    f"{tag} n={int(sample.get('count', 0))}"
                    f" p50={sample.get('p50', 0):.3f} p95={sample.get('p95', 0):.3f}"
                    f" avg={average:.3f}"
                )
            else:
                parts.append(f"{tag}={sample.get('value', 0):g}")
    if len(parts) > limit:
        parts = parts[:limit] + [f"...(+{len(parts) - limit})"]
    return "; ".join(parts)


def emit_metrics(
    snap: MetricsSnapshot | None = None,
    logger: logging.Logger | None = None,
    prefix: str = "METRIC",
) -> str:
    """打一条 METRIC JSON 行 + 一条人类可读摘要行，返回 JSON 字符串。

    参数 ``snap`` 为空时导出全局注册表快照；注册表为空时按
    :data:`METRIC_CATALOG` 补一份零值清单，保证首次抓取就能看到全部指标名。
    受 ``METRICS_LOG_ENABLED`` 控制；关闭时只返回 JSON 不打印。
    """
    snap = snap or _snapshot_with_catalog()
    payload = snap.to_json()
    log = logger or _LOGGER
    if metrics_enabled():
        try:
            log.info("%s %s", prefix, payload)
            if snap.metrics:
                log.info("METRIC_SUMMARY %s", _human_summary(snap))
        except Exception:  # pragma: no cover - 日志系统自身异常绝不外溢
            _LOGGER.warning("指标日志写出失败", exc_info=True)
    return payload


def _snapshot_with_catalog() -> MetricsSnapshot:
    """注册表快照；为空时返回 METRIC_CATALOG 的零值清单（便于固定字段名）。"""
    snap = snapshot()
    if snap.metrics:
        return snap
    payload: dict[str, Any] = {}
    for name, (kind, description, unit) in METRIC_CATALOG.items():
        payload[name] = {
            "type": kind, "unit": unit, "description": description,
            "value": 0.0,
            **({"count": 0.0, "sum": 0.0, "min": 0.0, "max": 0.0,
                "p50": 0.0, "p95": 0.0, "p99": 0.0} if kind == "histogram" else {}),
            "samples": [{"labels": {}, "value": 0.0,
                         **({"count": 0.0, "sum": 0.0, "min": 0.0, "max": 0.0,
                             "p50": 0.0, "p95": 0.0, "p99": 0.0, "avg": 0.0,
                             "buckets": []} if kind == "histogram" else {})}],
        }
    return MetricsSnapshot(payload)


def emit_request_metrics(
    request_id: str | None = None,
    logger: logging.Logger | None = None,
    extra: Mapping[str, Any] | None = None,
    flush: bool = False,
    only: str | None = None,
) -> str:
    """请求结束时调用：把当前注册表快照写成 METRIC JSON 行。

    :param request_id: 显式 request_id；缺省从 observability 当前上下文取；
    :param extra: 追加的自定义字段（例如 ``{"ttft_seconds": 1.2}``）；
    :param flush: True 表示导出后清空注册表（适合 CLI 一次性任务）。
    :param only: 只导出名字以该前缀开头的指标（避免每个请求都吐全量指标）。
    """
    snap = snapshot()
    if only:
        snap = MetricsSnapshot(
            {k: v for k, v in snap.metrics.items() if k.startswith(only)},
            snap.timestamp,
        )
    if request_id is None:
        try:
            from .observability import current_request_id  # 延迟导入避免循环
            request_id = current_request_id()
        except Exception:  # pragma: no cover
            request_id = None

    record = snap.to_dict()
    if request_id:
        record["request_id"] = request_id
    if extra:
        record["extra"] = dict(extra)
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
    log = logger or _LOGGER
    if metrics_enabled():
        try:
            log.info("METRIC %s", payload)
            if snap.metrics:
                log.info("METRIC_SUMMARY %s", _human_summary(snap))
        except Exception:  # pragma: no cover
            _LOGGER.warning("请求指标写出失败", exc_info=True)
    if flush:
        reset_metrics()
    return payload


def summarize_metrics(snap: MetricsSnapshot | None = None) -> str:
    return _human_summary(snap or snapshot())


def flush_metrics(logger: logging.Logger | None = None, only: str | None = None) -> str:
    """兼容写法：导出并清空（等价 ``emit_request_metrics(flush=True)``）。"""
    return emit_request_metrics(logger=logger, flush=True, only=only)


# ---------------- 输出 2：Prometheus 文本 ----------------

_TYPE_MAP = {"counter": "counter", "gauge": "gauge", "histogram": "histogram"}


def render_prometheus(snap: MetricsSnapshot | None = None) -> str:
    """渲染成 Prometheus 文本格式（供 ``GET /metrics`` 使用）。

    保证：``# HELP`` / ``# TYPE`` 里声明的名字与紧随其后的**真实样本名完全一致**。
    Counter 走 :func:`_prometheus_counter_name` 归一化（已带 ``_total`` 不重复追加）。
    """
    snap = snap or snapshot()
    lines: list[str] = []
    for name in sorted(snap.metrics):
        payload = snap.metrics[name]
        kind = payload.get("type", "gauge")
        ptype = _TYPE_MAP.get(kind, "gauge")
        description = (payload.get("description") or "").replace("\n", " ")
        unit = payload.get("unit") or ""
        help_text = f"{description} (unit: {unit})" if unit else description or name

        # 声明名 = 样本名前缀：counter 归一化后作为 HELP/TYPE 与样本的共同名字，
        # 三者（HELP / TYPE / 样本）永远同名，不再出现「TYPE 名无同名样本」。
        sample_name = _prometheus_counter_name(name) if ptype == "counter" else name
        if ptype == "counter":
            # 累计口径显式写进 HELP：与同屏 Gauge 存量（如 vector_store_rows）一眼可分。
            help_text = f"[{COUNTER_CUMULATIVE_NOTE}] {help_text}"
        lines.append(f"# HELP {sample_name} {help_text}")
        lines.append(f"# TYPE {sample_name} {ptype}")

        samples = payload.get("samples") or [{}]
        for sample in samples:
            labels = sample.get("labels") or {}
            if kind == "histogram":
                count = int(sample.get("count", 0) or 0)
                total = float(sample.get("sum", 0.0) or 0.0)
                values_count = count
                for bound in _bucket_bounds(name):
                    lines.append(
                        f'{name}_bucket{_with_label(labels, "le", _fmt(bound))} '
                        f"{_bucket_le(sample, bound, values_count)}"
                    )
                lines.append(f'{name}_bucket{_with_label(labels, "le", "+Inf")} {values_count}')
                lines.append(f"{name}_count{_format_labels(labels)} {values_count}")
                lines.append(f"{name}_sum{_format_labels(labels)} {_fmt(total)}")
            else:
                metric_name = sample_name
                lines.append(f"{metric_name}{_format_labels(labels)} {_fmt(sample.get('value', 0))}")
    return "\n".join(lines) + ("\n" if lines else "")


def _prometheus_counter_name(name: str) -> str:
    """Counter 的 Prometheus 样本名：**已带 ``_total`` 就原样返回，否则补一次**。

    这是「注册名 vs 样本名」的**唯一**归一化点。历史 bug：早期实现无条件追加 ``_total``，
    而 ``METRIC_CATALOG`` 里的 counter 名**本身已带** ``_total``，于是声明与样本错位 ——
    ``# TYPE docs_total counter`` 挂在真实样本 ``docs_total_total`` 上，
    Prometheus 抓到的序列退化为 untyped。回归测试见 ``tests/test_observability.py``。
    """
    return name if name.endswith("_total") else f"{name}_total"


#: 所有 counter 的 HELP 前缀：显式标注「累计」口径，避免与同屏存量 Gauge 混淆。
COUNTER_CUMULATIVE_NOTE = "本次进程累计，重启归零"


def _bucket_bounds(name: str) -> tuple[float, ...]:
    metric = registry().get(name)
    if isinstance(metric, Histogram):
        return metric.buckets
    return DEFAULT_BUCKETS


def _bucket_le(sample: Mapping[str, Any], bound: float, total: int) -> int:
    """用快照里的分位/最值信息近似还原累计桶（快照不含原始样本时的兜底）。"""
    buckets = sample.get("buckets")
    if isinstance(buckets, Sequence) and buckets:
        for bound_value, cumulative in buckets:
            if float(bound_value) == float(bound):
                return int(cumulative)
    maximum = float(sample.get("max", 0.0) or 0.0)
    return total if maximum <= bound else 0


def _with_label(labels: Mapping[str, str], key: str, value: str) -> str:
    merged = {str(k): str(v) for k, v in labels.items()}
    merged[key] = value
    return _format_labels(merged)


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "0"
    if number != number or number in (float("inf"), float("-inf")):
        return "0"
    if number == int(number) and abs(number) < 1e15:
        return str(int(number))
    return repr(round(number, 9))
