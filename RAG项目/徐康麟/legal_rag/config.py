# -*- coding: utf-8 -*-
"""全局配置中心。

所有可切换的后端都集中在这里：默认全部走「零重依赖兜底实现」，
装好真实依赖后只改配置即可切到 BGE-m3 / Ollama / Milvus / Redis，
后续再换算法云 Qwen3.8-27B（vLLM/SGLang）也不需要改调用代码。

**约定：所有新配置一律「环境变量优先 + 代码内默认值」。业务模块只读 RagConfig，
不得自行读 os.environ。**

⚠️ 这条约定有**已知例外**（如实记录，不假装不存在）。例外只允许两类：
「初始化期读一次并缓存」与「密钥/日志引导项必须绕开 RagConfig」，实际清单：

* ``logging_setup.py`` —— ``LOG_LEVEL`` / ``LOG_DIR`` / ``LOG_FILE`` / ``LOG_TO_FILE``
  / ``LOG_NOISY_LEVEL``：日志引导必须早于配置中心可用（鸡生蛋问题）；
* ``metrics.py`` —— ``METRICS_LOG_ENABLED`` / ``METRICS_BUCKETS``：指标是配置中心的
  底座依赖，不能反向依赖它；
* ``observability.py`` —— ``TRACE_SAMPLE``：同上（埋点底座）；
* ``store/base.py`` —— ``MILVUS_FALLBACK_TO_MEMORY``：store 层不反向依赖 config
  （它与 ``DEFAULT_MILVUS_FALLBACK_TO_MEMORY`` 是同值镜像，已有测试断言两处一致）；
* ``embedding/bge_m3.py`` —— ``EMBEDDING_DEVICE``：构造 embedder 时读一次（初始化期）；
* ``generate/router.py`` / ``generate/deepseek.py`` / ``generate/openai_compat.py`` ——
  ``DEEPSEEK_API_KEY`` / ``OPENAI_API_KEY``：**密钥刻意不进 RagConfig**
  （避免配置对象被打印/序列化时泄露），客户端构造或调用时读；
* ``system_metrics.py`` —— ``SystemDrive`` / ``SystemRoot``：Windows 平台探测，
  不是本项目的配置项。

除上述之外，**新代码一律走 RagConfig**；新增例外请在同一处更新这份清单。

默认值：本地优先（127.0.0.1）—— ⚠️ 这是一次**行为变更**
--------------------------------------------------
Milvus / Redis 的代码内默认值一律指向**本机 `127.0.0.1`**：
WSL（或 Docker Desktop）里用 Docker 起的 Milvus / Redis 都走这个地址，
从 Windows 访问 WSL 内的服务也有 localhost 转发，因此两边默认值一致。

⚠️ **行为变更（自本版起）**：旧版本不设 `MILVUS_HOST` / `REDIS_URL` 时，
默认连的是 Ubuntu VM `192.168.188.128`（一台可能已经不存在的机器）；
现在默认连**本机**。要连**远端**（保留旧 VM 或另一台机器）必须**显式**设置，
不再有任何隐式远端默认值。

===========  ===============================  ==========================
配置项         Windows 宿主                     WSL / Linux（本机 Docker）
===========  ===============================  ==========================
OLLAMA_BASE_URL  http://127.0.0.1:11434        http://127.0.0.1:11434
MILVUS_HOST      127.0.0.1（本机）              127.0.0.1（本机）
REDIS_URL        redis://127.0.0.1:6379/0      redis://127.0.0.1:6379/0
===========  ===============================  ==========================

连远端时的覆盖示例（三处都要按实际机器显式设）：
* `MILVUS_HOST=<远端机器>`、`REDIS_URL=redis://<远端机器>:6379/0`；
* WSL 里连 **Windows 宿主**上的 Ollama：`OLLAMA_BASE_URL=http://<宿主IP>:11434`
  （WSL 内的 `127.0.0.1` 指 WSL 自己，连不上宿主）。

安全红线（写死在默认值里）
--------------------------
* Redis：只用 ``legal_rag:`` 前缀做隔离，**绝不 flush 已有用户 key**；
* Milvus：默认 collection 是新建的 ``legal_rag_chunks_bge_m3_1024``，
  绝不触碰 ``user_long_term_memory`` / ``user_long_term_memory_bge`` /
  ``ragtest`` 等既有 collection。
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
KNOWLEDGE_DIR: Path = PROJECT_ROOT / "knowledge"
INDEX_DIR: Path = PROJECT_ROOT / "index"

#: 本地优先默认值：Milvus / Redis 都默认连**本机** 127.0.0.1
#: （WSL 或 Docker Desktop 里起的 Milvus/Redis 都能命中；Windows 也有到 WSL 的 localhost 转发）。
#: ⚠️ 行为变更：旧版本这里写的是 Ubuntu VM 的 192.168.188.128，不设 MILVUS_HOST /
#: REDIS_URL 就会默认连那台 VM；现在默认连本机。**连远端必须显式设置**，
#: 例如 MILVUS_HOST=<远端机器>、REDIS_URL=redis://<远端机器>:6379/0。
DEFAULT_MILVUS_HOST = "127.0.0.1"
DEFAULT_REDIS_URL = "redis://127.0.0.1:6379/0"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MILVUS_COLLECTION = "legal_rag_chunks_bge_m3_1024"

#: Redis key 统一前缀（隔离用户已有 19 个 key）
REDIS_PREFIX = "legal_rag:"

#: Milvus 连不上/维度不匹配时是否降级为内存向量库的**唯一默认值**（True = 保持既有行为）。
#: 注意：``legal_rag.store.base.DEFAULT_MILVUS_FALLBACK_TO_MEMORY`` 是本常量的同值镜像
#: （store 层不反向依赖 config，故各自持有一份；**两处必须保持一致**，已有测试断言）。
DEFAULT_MILVUS_FALLBACK_TO_MEMORY = True

_TRUE_VALUES = {"1", "true", "yes", "on", "y", "t"}
_FALSE_VALUES = {"0", "false", "no", "off", "n", "f", ""}

_LEVEL_NAMES = {
    "CRITICAL": 50, "FATAL": 50, "ERROR": 40, "WARNING": 30, "WARN": 30,
    "INFO": 20, "DEBUG": 10, "NOTSET": 0,
}


#: Milvus 本地文件式 URI 的环境变量正式名（pymilvus 撞名保护的产物，
#: 见 :func:`release_local_milvus_uri_env`）。
MILVUS_URI_LOCAL_ENV = "LEGAL_RAG_MILVUS_URI"

#: 兼容读取的旧名字。注意 pymilvus 自己也读 ``MILVUS_URI``，且只认 http(s)://。
MILVUS_URI_LEGACY_ENVS = ("MILVUS_URI",)


def _is_local_milvus_uri(uri: str) -> bool:
    """判定是否「本地文件式」URI（Milvus Lite 的 ``xxx.db`` 或普通路径）。

    Milvus Lite 拿文件路径当 URI（``/root/autodl-tmp/data/legal_rag_lite.db``），
    而 pymilvus 的 ORM 只认 ``http(s)://host:port``。两者共用 ``MILVUS_URI``
    这个变量名会互相打脸，所以必须先分类再决定要不要从环境里摘掉。
    """
    text = (uri or "").strip()
    if not text:
        return False
    if text.lower().startswith(("http://", "https://", "unix:", "tcp:")):
        return False
    return "://" not in text


def release_local_milvus_uri_env() -> str:
    """把「本地文件式」的 MILVUS_URI 从 ``os.environ`` 摘出来并返回其值（幂等）。

    背景（真机踩坑，cloud 2026-09-21）：``pymilvus`` 自己也会读环境变量
    ``MILVUS_URI``——``pymilvus/orm/connections.py`` 的模块级 ``Connections()``
    单例在 import ``pymilvus.orm.collection`` 时就解析它（``Config.MILVUS_URI``），
    且只接受 ``http[s]://host:port``。我们为 Milvus Lite 设的
    ``MILVUS_URI=/xxx/legal_rag_lite.db`` 会让 pymilvus 在 **import 阶段**直接抛
    ``ConnectionConfigException: Illegal uri``，表现得像「连不上 Milvus」，
    实际是环境变量撞名（旧代码因此静默降级到内存库，131k 块白嵌）。

    处理：本地文件式的值读完就摘掉，并归一化写入 ``LEGAL_RAG_MILVUS_URI``；
    ``http(s)://`` 形式保持原样（pymilvus 与我们的语义一致，不冲突）。
    """
    for name in MILVUS_URI_LEGACY_ENVS:
        raw = (os.environ.get(name) or "").strip()
        if raw and _is_local_milvus_uri(raw):
            os.environ.pop(name, None)
            os.environ[MILVUS_URI_LOCAL_ENV] = raw
            return raw
    return ""


#: 模块导入时立刻执行：必须早于任何 pymilvus import，否则撞名保护失效。
LOCAL_MILVUS_URI: str = release_local_milvus_uri_env()


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    """布尔环境变量：1/true/yes/on 为真，0/false/no/off/空为假。"""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    text = raw.strip().lower()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    return default


def _env_list(name: str, default: list[str] | None = None) -> list[str]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return list(default or [])
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_bool_with_warning(name: str, default: bool) -> bool:
    """严格布尔解析：非法值**保守取 default 并记 warning**（供一等配置项使用）。"""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    text = raw.strip().lower()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    logging.getLogger(__name__).warning(
        "环境变量 %s=%r 不是合法布尔值（可用 1/true/yes/on 或 0/false/no/off），"
        "已保守采用默认值 %s", name, raw, default)
    return default


def env_log_level(default: str = "INFO") -> str:
    """读取 LOG_LEVEL（日志引导用，返回值可直接喂 logging.setLevel 的解析函数）。"""
    return _env("LOG_LEVEL", default)


def env_log_level_int(default: int = 20) -> int:
    """读取 LOG_LEVEL 并解析成 logging 级别常量。"""
    raw = env_log_level(str(_level_name(default)))
    if raw.isdigit():
        return int(raw)
    return _LEVEL_NAMES.get(raw.upper(), default)


def _level_name(level: int) -> str:
    for name, value in _LEVEL_NAMES.items():
        if value == level:
            return name
    return "INFO"


@dataclass
class ChunkConfig:
    """分块参数。"""
    # fixed | sentence | paragraph | article
    # 默认用「法条感知」（article）：法律文本按「条」组织，固定窗口会把条切碎——
    # 离线探针实测 paragraph/300 下 88.7% 的子块横跨 ≥2 条、39.8% 从半句起头、
    # 最大块 1693 字 → 检索命中的片段缺条号、引用无法落到具体条文。
    # 非法律文本（不含「第X条」）会自动退化为 paragraph，不报错也不丢内容。
    strategy: str = "article"
    chunk_size: int = 300         # 子块目标字符数
    overlap: int = 50             # 子块之间的重叠，避免语义断裂
    parent_size: int = 1200       # 父块目标字符数（保上下文）


@dataclass
class RetrievalConfig:
    """检索与重排参数。"""
    vector_top_k: int = 10
    keyword_top_k: int = 10
    final_top_k: int = 5
    #: 法名定向召回：问题点名了哪部法（"商标侵权"/"正当防卫"…），就额外去该法的文件里
    #: 补一遍候选（**只加不删**，不做硬过滤）。评测 v1/v2 证明「该法条没进候选池」是
    #: 失分主因，重排救不了缺失的候选 —— 见 ``retrieve/law_scope.py``。
    law_scope_enabled: bool = True
    law_scope_top_k: int = 10
    #: 法名定向召回里"配套/司法解释"那一组取多少条（法典本体单列一组，用 ``law_scope_top_k``）。
    #: 真机踩过：两组混在一起取时司法解释把名额挤掉，讲"什么行为构成商标侵权"的
    #: 《商标法》第五十七条进不了候选。
    law_scope_related_top_k: int = 5
    #: 法名命中时，最终上下文里**至少**几条来自该法（0 = 不保留，**默认关**）。
    #: ⚠️ 实测（2026-09-22，78 题条号级）：把它设成 2 **没有改善**，且余弦基线的
    #: Recall@5 从 0.359 掉到 0.282（该法家族本已占 ≥2 个名额时"保留"换来的只是该法**别的**
    #: 条文）。根因是**条文级语义**：问题问"商标侵权"，《商标法》第五十七条写的是
    #: "侵犯注册商标专用权"（词面对不上），而库里**别处** 19 次"商标侵权"反而分更高。
    #: ⇒ 需要的是"条文级语义选择器"（LLM 列表式重排 / 法律域重排器），不是名额分配。
    law_scope_reserve: int = 0
    #: **法内检索的候选宽度**：识别到法名后，在该法文件集合内取这么多条参与"法内重排"。
    #: 为什么需要它（P8 实测教训）：法内候选若只有 `law_scope_top_k`(10)+配套(5) 条，
    #: 期望条号在法内同样进不来 ⇒ "名额保留"只能换成该法**别的**条文、甚至把原本对的那条挤掉
    #: （实测 cosine 的条号级 Recall@5 从 0.359 掉到 0.282）。一部法通常几十到几百条，
    #: 所以法内候选要**尽量全**（200 足够覆盖绝大多数法典），再在法内重排取前 k。
    law_scope_internal_top_k: int = 200
    #: 口语→立法术语映射（用在**关键词通道**：只加词、不引模型）。实测依据：问"出资不实"
    #: 而法里写"未履行出资义务"；问"商标侵权"而《商标法》第五十七条写"侵犯注册商标专用权"。
    #: ⚠️ **实测（2026-09-22，78 题条号级）没有改善**：cosine 0.3590→0.3077（-5.1%）、
    #: m3 0.3846→0.3718（-1.3%）、large 不变、base 0.3462→0.3590（+1.3%）—— 净效果在噪声内
    #: 偏负 ⇒ **默认关**，代码与测试保留（`TERM_MAP_ENABLED=1` 可开，术语表在 `retrieve/term_map.py`）。
    term_map_enabled: bool = False
    term_map_limit: int = 6
    #: **LLM 列表式选择器**（P8 的第五条杠杆）：识别到法名后，把该法的"法内宽候选"
    #: （最多 `law_selector_max_candidates` 条，每条只给 条号+前若干字）交给大模型挑
    #: "最能直接回答该问题的条"，选中的条文**置顶且豁免词面证据闸门**。
    #: 与前面四条杠杆（池放大 / 换重排器 / 法内检索+名额保留 / 术语映射）的区别：前四条都在
    #: **改分数**（词面或通用向量），而实测根因是"问题问的是**哪一条**"这层语义抓不住
    #: （"商标侵权" ↔《商标法》第五十七条"侵犯注册商标专用权"；"出资不实" ↔"未履行出资义务"）
    #: ⇒ 需要一个读得懂条文的判官。它是**选择器不是生成器**（只回序号，原文来自语料，无编造风险）。
    #: 默认**开**（用户 2026-09-23 定调：「速度与质量取中间值，速度也不能太慢」）：
    #: 便宜口径下（24 候选 × 60 字）实测 **1.43 s/次**，只在识别到法名时调 ⇒ 平均每题
    #: **+1.0 s**，换来的增益是 cosine Recall@5 0.4103→0.5513（+0.1410）、
    #: m3 0.5000→0.5385（+0.0385，MRR +0.0776）。见 `docs/RERANK-EXPERIMENT.md`。
    law_selector_enabled: bool = True
    #: 选择器调用的**超时**（秒）。**必须显著小于** `LLM_TIMEOUT`：这一步是锦上添花，
    #: 大模型端点抖动时绝不能让每题都卡满 60 s；配合"连续 3 次失败即熔断"，最坏情况是
    #: 前 3 题各多等这么久，之后不再调用。
    law_selector_timeout: float = 8.0
    #: 选择器最多挑几条（挑中的会插到最终结果最前面）
    law_selector_top_k: int = 3
    #: 送进 prompt 的候选上限（控成本：prompt 长度≈候选数×每条字数 ⇒ 直接决定 TTFT）
    #: 实测：60 条 × 120 字 ≈ 6k token ≈ **5.3 秒/题**（27B FP8）。用户要求「速度与质量取中间值」
    #: ⇒ 改用 24 条 × 60 字（≈1.5k token）并把调用时机收紧（见下）。
    law_selector_max_candidates: int = 24
    #: 每条候选截断的字符数
    law_selector_snippet_chars: int = 60
    #: **什么时候才叫大模型挑条**（速度/质量折中的主开关）：
    #:   ``always``         —— 只要识别到法名就调（最准、最慢）
    #:   ``low_confidence`` —— 只在"便宜的信号显示没把握"时调：
    #:       ① 最终 top-k 里**该法家族的条文不足** `law_selector_min_law_hits` 条
    #:          （用户问的正是这部法，却几乎没有它的条文 ⇒ 大概率没找对）；
    #:       ② 过闸门后剩下的条数**少于** `law_selector_min_hits`（召回本来就薄）。
    #:     两个判据都**尺度无关**（不依赖重排分数阈值，换重排器/换后端不用重标定）。
    law_selector_trigger: str = "low_confidence"
    #: 触发判据①：top-k 里该法家族条文少于这么多条就要挑条
    law_selector_min_law_hits: int = 2
    #: 触发判据②：过闸门后剩这么少条（或更少）就要挑条
    law_selector_min_hits: int = 5
    #: 给选择器的候选**加版本标记**（「现行有效」/「尚未施行(YYYY-MM-DD)」）。
    #: ⚠️ **实测后默认关**（2026-09-24，78 题条号级）：开标记后 m3 0.5385→0.5256、
    #: cosine 0.5513→0.5385（各 −1 题）。而且当初要改的理由（"选择器挑了未生效新版的条号"）
    #: 是**过期挑条缓存**造成的假象 —— 候选池过版本过滤后重新挑条时，现行有效版的条本来就在第一。
    #: 开关留给"新旧版条号差异更大"的语料（换场景必须重新实测）。
    law_selector_version_labels: bool = False
    #: 同一来源文件最多保留几条（0 = 不限制）。真机踩过：L23 的 top-3 是**同一份批复的
    #: 3 个拷贝**，把名额吃光；上限 2 能腾出位置给别的文件。
    max_per_source: int = 2
    #: BM25 索引落盘缓存（真机踩坑：每次进程重启都要流式取 11.9 万块重建，约 160s；
    #: 这期间关键词通道为空、用户会看到「资料还在预热」）。缓存按"签名"（语料条数 +
    #: 过滤范围）分文件存放，签名对不上自动重建，**绝不会**拿旧索引糊弄。
    bm25_cache_enabled: bool = True
    #: 版本过滤：语料里同一文件有多个版本时，只让现行版进候选（读
    #: ``index/version_inventory.json``；缺失即不过滤，见 ``retrieve/version_filter.py``）。
    version_filter_enabled: bool = True
    vector_weight: float = 1.0
    keyword_weight: float = 1.0
    fusion: str = "weighted"      # weighted | rrf
    score_threshold: float = 0.0  # 低分过滤
    #: 相关性闸门的**融合分阈值**（``0`` = 关闭自动判定）。注意 t115 起改口径：
    #: **门槛是词面证据（尺度无关，见 ``retrieve/hybrid.py::_MIN_CONTENT_TERMS`` = 3），
    #: 这个分数阈值只在门槛内做「补充保留」** —— 它不是每轮的硬门槛。
    #: 标定依据（真实链路 Milvus + bge-m3 向量通道 + BM25 关键词通道；逐题读数表见
    #: ``.pytmp/t115/diag-before.json`` / ``diag-after.json``、``.pytmp/t109/probe-score.json``）：
    #:   * 法律题 top1 融合分 = 2.0000 / 1.9628 / 1.9723 / 1.8600 / 1.8544 / 1.9329（两通道都命中）；
    #:   * 非法律题 top1 = 1.0000（关键词通道零命中，只有向量通道被归一化命中，恒为 1.0），
    #:     但「推荐几部好看的科幻电影」被**单字**命中的关键词通道抬到 1.8977 ⇒ 分数不能单独
    #:     当门槛，它只能是被词面证据门槛兜住的「补充保留」；
    #:   * ``劳动争议仲裁的时效是多久？`` 的候选融合分只有 0.9577~1.0000，词面证据却有 6~8 个
    #:     ⇒ 把分数当硬门槛必然误杀（t115 修的正是这条）。
    #: 默认 1.5 只决定「门槛内还能多留哪几条」：留 1.5 让 L2 那类第 2 条候选（1.8157）继续成为引用，
    #: 引用条数因而不低于 t109 基线。
    relevance_gate: float = 1.5
    #: 闸门只在**已标定后端**上生效（默认仅真实 Milvus）：阈值取自真实语料上的分数分布，
    #: 内存/离线后端（含 Milvus 不可用时的降级兜底）没有可比的分数尺度，在那里按分数判定
    #: 等于用没有意义的分数误杀。跳过一次一笔记入
    #: ``retrieval_gate_skipped_total{reason=uncalibrated_backend}``，不静默。
    #: ⚠️ 与「关键词通道未就绪」不是一回事：后者（真实后端 + 索引进后台建）**不再整条跳过**，
    #: 而是退到尺度无关的词面证据判据（``retrieval_gate_mode_total{mode=lexical_only}``）。
    relevance_gate_backends: tuple[str, ...] = ("milvus",)
    time_decay: float = 0.0       # >0 时按 created_at 做时间衰减加权
    time_decay_half_life_days: float = 180.0


@dataclass
class MilvusConfig:
    """Milvus 向量库（gRPC）。默认连本机 127.0.0.1；连远端须显式设 MILVUS_HOST。"""
    host: str = DEFAULT_MILVUS_HOST
    port: int = 19530
    collection: str = DEFAULT_MILVUS_COLLECTION
    #: 直接用 URI 覆盖 host/port（例如 "http://<远端主机>:19530"）
    uri: str = ""
    #: 免 token 部署留空；填了才传 token
    token: str = ""
    user: str = ""
    password: str = ""
    db_name: str = "default"
    connect_timeout: float = 10.0
    timeout: float = 30.0
    #: 维度不匹配时是否允许自动重建 collection（默认 false，避免误删数据）
    recreate_on_dim_mismatch: bool = False
    #: Milvus 构建/连接失败时，是否**降级为内存向量库**（默认 true = 与既有行为一致）。
    #:
    #: ⚠️ 风险语义：降级到 ``MemoryVectorStore`` 意味着**数据不持久** ——
    #: 向量只存在进程内存中，**进程一重启就全部丢失**，多进程/多 worker 之间也互不可见；
    #: 检索接口仍会正常返回结果，于是「看起来一切正常」但实际已经脱离 Milvus。
    #: 因此它**只适合「本机没有 Milvus 的本地调试」**（例如 Windows 未启动 Docker 时）。
    #: 生产 / VM 试运行请显式置 false（``MILVUS_FALLBACK_TO_MEMORY=0``），
    #: 让连不上 Milvus 直接抛 ``MilvusUnavailableError``，而不是静默降级。
    #: 降级一旦发生，``RagEngine.health()`` 会报 ``status=degraded`` /
    #: ``store_actual=memory`` / ``degraded_reason=...``，便于按日志定位。
    #: 环境变量 ``MILVUS_FALLBACK_TO_MEMORY``：1/true/yes/on 为开，0/false/no/off 为关（大小写不敏感）。
    fallback_to_memory: bool = DEFAULT_MILVUS_FALLBACK_TO_MEMORY
    #: 建索引参数（HNSW / IP / COSINE）
    index_type: str = "HNSW"
    metric_type: str = "COSINE"
    index_m: int = 16
    index_ef_construction: int = 200
    index_ef_search: int = 64


@dataclass
class OllamaConfig:
    """本地 Ollama 服务：嵌入 bge-m3 + 生成 qwen2.5:3b（免密钥、免 torch）。"""
    base_url: str = DEFAULT_OLLAMA_BASE_URL
    embedding_model: str = "bge-m3"
    llm_model: str = "qwen2.5:3b"
    #: 免密钥部署留空；Open WebUI 等转发场景可填
    api_key: str = ""
    embed_timeout: float = 60.0
    generate_timeout: float = 180.0
    #: 模型常驻时长（keep_alive），CPU 场景避免频繁重载
    keep_alive: str = "10m"
    #: None 表示交给 Ollama 自动选择（有 GPU 用 GPU，VM 自动回落 CPU）
    num_gpu: int | None = None
    num_thread: int | None = None
    #: 生成采样参数
    temperature: float = 0.3
    top_p: float = 0.9
    max_tokens: int = 512
    #: 调用失败重试次数与退避秒数
    max_retries: int = 2
    retry_backoff: float = 1.0

    @property
    def embed_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/api/embed"

    @property
    def generate_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/api/generate"

    @property
    def chat_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/api/chat"

    @property
    def tags_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/api/tags"


@dataclass
class ConcurrencyConfig:
    """并发与线程池：VM 无 GPU，默认保守值，避免 CPU 推理被打爆。"""
    max_concurrent_requests: int = 8
    thread_pool_size: int = 8
    ingest_workers: int = 4
    queue_max_size: int = 100


@dataclass
class UploadConfig:
    """PDF 上传与入库限制。"""
    upload_dir: str = "uploads"
    max_upload_mb: int = 50
    max_upload_files: int = 10
    allowed_extensions: list[str] = field(default_factory=lambda: [".pdf"])


@dataclass
class CacheConfig:
    """嵌入缓存（Redis，前缀隔离，绝不 flush 用户既有 key）。"""
    enabled: bool = True
    backend: str = "redis"           # redis | memory | none
    prefix: str = REDIS_PREFIX
    embed_ttl: int = 7 * 24 * 3600   # 7 天
    memory_max_items: int = 2048
    connect_timeout: float = 3.0
    socket_timeout: float = 5.0


@dataclass
class LoggingConfig:
    """日志与指标。"""
    level: str = "INFO"
    dir: str = "logs"
    file_name: str = "app.log"
    to_file: bool = True
    max_bytes: int = 20 * 1024 * 1024
    backup_count: int = 5
    noisy_level: str = "WARNING"
    metrics_log_enabled: bool = True
    metrics_buckets: str = ""
    trace_sample: float = 1.0
    # ---- P0.5 耗时评估 ----
    #: L3「全量函数前后日志」的级别。**空串 = 回落 LOG_LEVEL**。
    #: 存在意义：``@traced`` 默认 DEBUG 而 ``LOG_LEVEL`` 默认 INFO，导致
    #: 「耗时 X.Xms」被静默过滤（见 docs/REFACTOR-PLAN.md §3.7 的 B1）。
    trace_level: str = ""
    #: L2「慢调用清单」阈值（毫秒）；``<=0`` 关闭。只影响日志行，不影响 L1 直方图。
    slow_call_ms: float = 500.0
    #: L2 日志行采样率（0~1）。默认 1.0：慢调用一条都不丢。
    slow_call_sample: float = 1.0
    sys_metrics_enabled: bool = True
    sys_metrics_interval: float = 15.0
    sys_gpu_enabled: bool = True


def _auth_required_flag(default: bool) -> bool:
    """读「是否强制鉴权」：**主名 ``AUTH_REQUIRED``，别名 ``REQUIRE_AUTH``**。

    * 只设了一个 → 用那个；
    * 两个都设 → **以主名 ``AUTH_REQUIRED`` 为准**并记 warning（避免"我明明设了"
      却因为另一个残留变量而不生效，那种问题排查起来很费时间）；
    * 都没设 → 默认 ``False``（不强制，既有调用方零影响；对外发布前必须显式打开）。
    """
    canonical = os.environ.get("AUTH_REQUIRED")
    alias = os.environ.get("REQUIRE_AUTH")
    has_canonical = canonical is not None and canonical.strip() != ""
    has_alias = alias is not None and alias.strip() != ""
    if has_canonical and has_alias:
        logging.getLogger(__name__).warning(
            "同时设置了 AUTH_REQUIRED=%r 与 REQUIRE_AUTH=%r；两者都生效时"
            "以主名 AUTH_REQUIRED 为准（REQUIRE_AUTH 只是别名）", canonical, alias)
    if has_canonical:
        return _env_bool_with_warning("AUTH_REQUIRED", default)
    if has_alias:
        return _env_bool_with_warning("REQUIRE_AUTH", default)
    return default


@dataclass
class AuthConfig:
    """账号体系（注册/登录）开关与令牌参数。

    ⚠️ **默认「不强制鉴权」（require_auth=False）**，原因有二：

    1. 既有 345 项测试与内网调试脚本（直接传 ``user_id=u1``）必须零改动全绿
       （`AC-AU-37` 的硬要求正是"关闭后既有测试全通过"）；
    2. 强制鉴权一旦默认打开，所有既有调用方会在升级瞬间收到 401 —— 那是把
       "补一个登录页"变成"服务对外不可用"。

    **但这意味着默认部署是「谁都能冒充任何 user_id」的**。对外发布前必须置
    ``AUTH_REQUIRED=true``（别名 ``REQUIRE_AUTH``，两个名字都能生效，主名是
    ``AUTH_REQUIRED``）—— 启动横幅与 docs/API.md §9 都会显著提示这一点。
    """

    #: 是否强制鉴权（主名 ``AUTH_REQUIRED``，别名 ``REQUIRE_AUTH``）。默认关闭。
    require_auth: bool = False
    #: 会话令牌的 Cookie 名
    cookie_name: str = "lr_session"
    #: 令牌有效期（秒），默认 7 天（规范 §3.7）
    token_ttl_seconds: int = 7 * 24 * 3600
    #: Cookie 是否带 ``Secure``。本机是 http，默认 **False**，否则本地登录直接失效；
    #: 生产（HTTPS）必须置 ``AUTH_COOKIE_SECURE=true``。
    cookie_secure: bool = False
    #: 用户名查重（`/auth/username-available`）限速：窗口秒数 + 窗口内最大请求数。
    #: 规范给了"阈值可在 10–60 之间选择"的余地，取 30/60s（`AC-AU-36`）。
    rate_limit_window_seconds: int = 60
    rate_limit_max_requests: int = 30
    #: 恢复码熵（字节，``secrets.token_urlsafe`` 的入参）。一次性恢复码是**用户亲定**的
    #: 找回方式（不收集手机号/邮箱、不发验证码），长度要足够抗猜：默认 24 字节 ≈ 192 bit。
    recovery_code_bytes: int = 24
    #: 恢复码哈希的 scrypt 参数（与登录密码**不同盐**、同一套算法与参数）。
    #: 单独列出是为了让"恢复码被拖库"与"密码被拖库"两件事各自独立可调。
    #: 默认值与 `api/auth.py::SCRYPT_N`（2**15）保持一致；**不在 config 里 import
    #: api 层**，避免 config ← api 的反向依赖。
    recovery_hash_n: int = 2 ** 15


@dataclass
class RagConfig:
    # ---- 路径 ----
    project_root: Path = PROJECT_ROOT
    knowledge_dir: Path = KNOWLEDGE_DIR
    index_dir: Path = INDEX_DIR

    # ---- 向量库（对外兼容字段）----
    vector_store: str = "memory"        # memory | chroma | milvus
    collection: str = "legal_kb"

    # ---- 向量化 ----
    embedding_provider: str = "offline"  # offline | bge_m3 | ollama
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 256

    # ---- 重排 ----
    rerank_provider: str = "cosine"      # cosine | bge_rerank
    rerank_model: str = "BAAI/bge-reranker-v2-m3"

    # ---- 生成 ----
    llm_provider: str = "mock"           # mock | deepseek | openai_compat | ollama
    llm_fallbacks: list[str] = field(default_factory=list)  # 容灾降级链，例如 ["deepseek"]
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 512
    llm_timeout: float = 180.0
    deepseek_base_url: str = "https://api.deepseek.com"
    openai_compat_base_url: str = ""

    #: 知识库**删除**权限开关（用户裁决：「用户没有删除的权力即可」）。
    #: **默认关闭** = 普通用户不能删资料（`DELETE /documents/{doc_id}` 403 + 明确原因）；
    #: 打开（``ALLOW_DOCUMENT_DELETE=true``）是**本地维护者路径**：维护者自己要能清理。
    #: 上传与三档判定不受这个开关影响；资料列表始终共享可见（不按用户过滤）。
    allow_document_delete: bool = False

    # ---- 会话与记忆 ----
    session_window: int = 20             # 短期记忆滑动窗口条数
    redis_url: str = DEFAULT_REDIS_URL
    redis_prefix: str = REDIS_PREFIX     # 隔离前缀；服务侧绝不 flush 全库
    mysql_dsn: str = ""
    sqlite_path: str = ""                # 为空则用 index/legal_rag.db
    longterm_enabled: bool = False
    longterm_threshold: int = 20
    # ---- B-9 长期记忆**召回**（与"写入"分开的开关，默认关）----
    # 口径：长期记忆**写进去**（longterm_enabled）不等于**想得起来**。
    # 开启召回后，`/chat` 会把该用户长期记忆里语义最相近的若干条**作为提示词的一部分**
    # 注入本轮（独立分区，**不是法律依据**、不进引用）。默认关，因为注入历史会改变
    # 答案分布，必须先测量再开（`docs/`/`handoff/B-9-VERIFICATION.md` §7）。
    longterm_recall_enabled: bool = False
    longterm_recall_top_k: int = 3
    #: 召回注入的最少字符数（碎片太短就不注入，避免噪声）
    longterm_recall_min_chars: int = 8

    # ---- 服务 ----
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    stream: bool = True

    # ---- 检索性能（CPU 场景保守）----
    retrieval_final_top_k: int = 3
    context_max_chars: int = 1800

    # ---- 上传 ----
    upload: UploadConfig = field(default_factory=UploadConfig)

    # ---- 账号体系（注册/登录）----
    auth: AuthConfig = field(default_factory=AuthConfig)

    # ---- 子配置 ----
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # ---- 兼容旧代码的扁平别名（读起来短一点）----
    @property
    def milvus_host(self) -> str:
        return self.milvus.host

    @property
    def milvus_port(self) -> int:
        return self.milvus.port

    @property
    def milvus_collection(self) -> str:
        return self.milvus.collection

    @property
    def milvus_timeout(self) -> float:
        return self.milvus.timeout

    @property
    def ollama_base_url(self) -> str:
        return self.ollama.base_url

    @property
    def embed_timeout(self) -> float:
        return self.ollama.embed_timeout

    @property
    def upload_dir(self) -> str:
        return self.upload.upload_dir

    @property
    def log_level(self) -> str:
        return self.logging.level

    @property
    def log_dir(self) -> str:
        return self.logging.dir

    @classmethod
    def from_env(cls) -> "RagConfig":
        """从环境变量构造配置；任何一项都可覆盖，缺省用「各机自用」默认值。"""
        cfg = cls()

        # ---- 向量库（旧字段保留）----
        cfg.vector_store = _env("VECTOR_STORE", cfg.vector_store)
        cfg.collection = _env("VECTOR_COLLECTION", cfg.collection)

        # ---- 向量化 ----
        cfg.embedding_provider = _env("EMBEDDING_PROVIDER", cfg.embedding_provider)
        cfg.embedding_model = _env("EMBEDDING_MODEL", cfg.embedding_model)
        cfg.embedding_dim = _env_int("EMBEDDING_DIM", cfg.embedding_dim)
        cfg.rerank_provider = _env("RERANK_PROVIDER", cfg.rerank_provider)
        cfg.rerank_model = _env("RERANK_MODEL", cfg.rerank_model)

        # ---- 生成 ----
        cfg.llm_provider = _env("LLM_PROVIDER", cfg.llm_provider)
        cfg.llm_fallbacks = _env_list("LLM_FALLBACKS", cfg.llm_fallbacks)
        cfg.llm_model = _env("LLM_MODEL", cfg.llm_model)
        cfg.llm_temperature = _env_float("LLM_TEMPERATURE", cfg.llm_temperature)
        cfg.llm_max_tokens = _env_int("LLM_MAX_TOKENS", cfg.llm_max_tokens)
        cfg.llm_timeout = _env_float("LLM_TIMEOUT", cfg.llm_timeout)
        cfg.deepseek_base_url = _env("DEEPSEEK_BASE_URL", cfg.deepseek_base_url)
        cfg.openai_compat_base_url = _env("OPENAI_COMPAT_BASE_URL", cfg.openai_compat_base_url)

        # ---- Milvus ----
        cfg.milvus.host = _env("MILVUS_HOST", cfg.milvus.host)
        cfg.milvus.port = _env_int("MILVUS_PORT", cfg.milvus.port)
        cfg.milvus.collection = _env("MILVUS_COLLECTION", cfg.milvus.collection)
        # LEGAL_RAG_MILVUS_URI 优先（本地文件式配置的正式名字，见上方
        # release_local_milvus_uri_env）；MILVUS_URI 继续兼容（http(s):// 远端照旧可用）。
        cfg.milvus.uri = (
            _env(MILVUS_URI_LOCAL_ENV, "")
            or _env("MILVUS_URI", "")
            or LOCAL_MILVUS_URI
            or cfg.milvus.uri
        )
        cfg.milvus.token = _env("MILVUS_TOKEN", cfg.milvus.token)
        cfg.milvus.user = _env("MILVUS_USER", cfg.milvus.user)
        cfg.milvus.password = _env("MILVUS_PASSWORD", cfg.milvus.password)
        cfg.milvus.db_name = _env("MILVUS_DB_NAME", cfg.milvus.db_name)
        cfg.milvus.connect_timeout = _env_float("MILVUS_CONNECT_TIMEOUT", cfg.milvus.connect_timeout)
        cfg.milvus.timeout = _env_float("MILVUS_TIMEOUT", cfg.milvus.timeout)
        cfg.milvus.recreate_on_dim_mismatch = _env_bool(
            "MILVUS_RECREATE_ON_DIM_MISMATCH", cfg.milvus.recreate_on_dim_mismatch)
        # 降级开关（一等配置项）：未设置/空白 → 保持默认（= 常量默认，行为不变）；
        # 设了 → 按 1/true/yes/on 或 0/false/no/off 解析；非法值保守取默认并记 warning。
        cfg.milvus.fallback_to_memory = _env_bool_with_warning(
            "MILVUS_FALLBACK_TO_MEMORY", DEFAULT_MILVUS_FALLBACK_TO_MEMORY)
        cfg.milvus.index_type = _env("MILVUS_INDEX_TYPE", cfg.milvus.index_type)
        cfg.milvus.metric_type = _env("MILVUS_METRIC_TYPE", cfg.milvus.metric_type)

        # ---- Ollama ----
        cfg.ollama.base_url = _env("OLLAMA_BASE_URL", cfg.ollama.base_url)
        cfg.ollama.embedding_model = _env("EMBEDDING_MODEL", cfg.ollama.embedding_model)
        cfg.ollama.llm_model = _env("LLM_MODEL", cfg.ollama.llm_model)
        cfg.ollama.api_key = _env("OLLAMA_API_KEY", cfg.ollama.api_key)
        cfg.ollama.embed_timeout = _env_float("EMBED_TIMEOUT", cfg.ollama.embed_timeout)
        cfg.ollama.generate_timeout = _env_float("LLM_TIMEOUT", cfg.ollama.generate_timeout)
        cfg.ollama.keep_alive = _env("OLLAMA_KEEP_ALIVE", cfg.ollama.keep_alive)
        _num_gpu = _env("OLLAMA_NUM_GPU", "")
        if _num_gpu:
            cfg.ollama.num_gpu = _env_int("OLLAMA_NUM_GPU", 0)
        _num_thread = _env("OLLAMA_NUM_THREAD", "")
        if _num_thread:
            cfg.ollama.num_thread = _env_int("OLLAMA_NUM_THREAD", 0)
        cfg.ollama.temperature = _env_float("LLM_TEMPERATURE", cfg.ollama.temperature)
        cfg.ollama.top_p = _env_float("LLM_TOP_P", cfg.ollama.top_p)
        cfg.ollama.max_tokens = _env_int("LLM_MAX_TOKENS", cfg.ollama.max_tokens)
        cfg.ollama.max_retries = _env_int("LLM_MAX_RETRIES", cfg.ollama.max_retries)

        # ---- 并发 ----
        cfg.concurrency.max_concurrent_requests = _env_int(
            "MAX_CONCURRENT_REQUESTS", cfg.concurrency.max_concurrent_requests)
        cfg.concurrency.thread_pool_size = _env_int(
            "THREAD_POOL_SIZE", cfg.concurrency.thread_pool_size)
        cfg.concurrency.ingest_workers = _env_int("INGEST_WORKERS", cfg.concurrency.ingest_workers)
        cfg.concurrency.queue_max_size = _env_int("QUEUE_MAX_SIZE", cfg.concurrency.queue_max_size)

        # ---- 上传 ----
        cfg.upload.upload_dir = _env("UPLOAD_DIR", cfg.upload.upload_dir)
        cfg.upload.max_upload_mb = _env_int("MAX_UPLOAD_MB", cfg.upload.max_upload_mb)
        cfg.upload.max_upload_files = _env_int("MAX_UPLOAD_FILES", cfg.upload.max_upload_files)
        cfg.upload.allowed_extensions = [
            ext if ext.startswith(".") else f".{ext}"
            for ext in _env_list("UPLOAD_ALLOWED_EXT", cfg.upload.allowed_extensions)
        ]

        # ---- 记忆与缓存 ----
        cfg.redis_url = _env("REDIS_URL", cfg.redis_url)
        cfg.redis_prefix = _env("REDIS_PREFIX", cfg.redis_prefix)
        cfg.cache.enabled = _env_bool("EMBED_CACHE_ENABLED", cfg.cache.enabled)
        cfg.cache.backend = _env("EMBED_CACHE_BACKEND", cfg.cache.backend)
        cfg.cache.prefix = _env("CACHE_PREFIX", cfg.cache.prefix)
        cfg.cache.embed_ttl = _env_int("EMBED_CACHE_TTL", cfg.cache.embed_ttl)
        cfg.cache.memory_max_items = _env_int("EMBED_CACHE_MAX_ITEMS", cfg.cache.memory_max_items)
        cfg.mysql_dsn = _env("MYSQL_DSN", cfg.mysql_dsn)
        cfg.sqlite_path = _env("SQLITE_PATH", cfg.sqlite_path)
        cfg.session_window = _env_int("SESSION_WINDOW", cfg.session_window)
        cfg.longterm_enabled = _env_bool("LONGTERM_ENABLED", cfg.longterm_enabled)
        cfg.longterm_threshold = _env_int("LONGTERM_THRESHOLD", cfg.longterm_threshold)
        cfg.longterm_recall_enabled = _env_bool("LONGTERM_RECALL_ENABLED",
                                                cfg.longterm_recall_enabled)
        cfg.longterm_recall_top_k = _env_int("LONGTERM_RECALL_TOP_K", cfg.longterm_recall_top_k)
        cfg.longterm_recall_min_chars = _env_int("LONGTERM_RECALL_MIN_CHARS",
                                                 cfg.longterm_recall_min_chars)

        # ---- 服务 ----
        cfg.api_host = _env("API_HOST", cfg.api_host)
        cfg.api_port = _env_int("API_PORT", cfg.api_port)
        cfg.stream = _env_bool("STREAM", cfg.stream)
        cfg.retrieval_final_top_k = _env_int("RETRIEVAL_FINAL_TOP_K", cfg.retrieval_final_top_k)
        cfg.retrieval.final_top_k = cfg.retrieval_final_top_k
        # 候选池宽度（t-评测 v1/v2 结论：失分主因是"该法条没进候选池"，重排救不了
        # 缺失的候选 ⇒ 这两个必须可调，否则线上没法做召回实验）。
        cfg.retrieval.vector_top_k = _env_int("VECTOR_TOP_K", cfg.retrieval.vector_top_k)
        cfg.retrieval.keyword_top_k = _env_int("KEYWORD_TOP_K", cfg.retrieval.keyword_top_k)
        cfg.retrieval.law_scope_enabled = _env_bool("LAW_SCOPE_ENABLED",
                                                    cfg.retrieval.law_scope_enabled)
        cfg.retrieval.law_scope_top_k = _env_int("LAW_SCOPE_TOP_K", cfg.retrieval.law_scope_top_k)
        cfg.retrieval.law_scope_related_top_k = _env_int(
            "LAW_SCOPE_RELATED_TOP_K", cfg.retrieval.law_scope_related_top_k)
        cfg.retrieval.law_scope_reserve = _env_int("LAW_SCOPE_RESERVE",
                                                   cfg.retrieval.law_scope_reserve)
        cfg.retrieval.law_scope_internal_top_k = _env_int(
            "LAW_SCOPE_INTERNAL_TOP_K", cfg.retrieval.law_scope_internal_top_k)
        cfg.retrieval.term_map_enabled = _env_bool("TERM_MAP_ENABLED",
                                                   cfg.retrieval.term_map_enabled)
        cfg.retrieval.term_map_limit = _env_int("TERM_MAP_LIMIT", cfg.retrieval.term_map_limit)
        cfg.retrieval.law_selector_enabled = _env_bool("LAW_SELECTOR_ENABLED",
                                                        cfg.retrieval.law_selector_enabled)
        cfg.retrieval.law_selector_timeout = _env_float(
            "LAW_SELECTOR_TIMEOUT", cfg.retrieval.law_selector_timeout)
        cfg.retrieval.law_selector_top_k = _env_int("LAW_SELECTOR_TOP_K",
                                                    cfg.retrieval.law_selector_top_k)
        cfg.retrieval.law_selector_max_candidates = _env_int(
            "LAW_SELECTOR_MAX_CANDIDATES", cfg.retrieval.law_selector_max_candidates)
        cfg.retrieval.law_selector_snippet_chars = _env_int(
            "LAW_SELECTOR_SNIPPET_CHARS", cfg.retrieval.law_selector_snippet_chars)
        cfg.retrieval.law_selector_trigger = (
            _env("LAW_SELECTOR_TRIGGER", cfg.retrieval.law_selector_trigger) or "always").strip()
        cfg.retrieval.law_selector_min_law_hits = _env_int(
            "LAW_SELECTOR_MIN_LAW_HITS", cfg.retrieval.law_selector_min_law_hits)
        cfg.retrieval.law_selector_min_hits = _env_int(
            "LAW_SELECTOR_MIN_HITS", cfg.retrieval.law_selector_min_hits)
        cfg.retrieval.law_selector_version_labels = _env_bool(
            "LAW_SELECTOR_VERSION_LABELS", cfg.retrieval.law_selector_version_labels)
        cfg.retrieval.max_per_source = _env_int("RETRIEVAL_MAX_PER_SOURCE",
                                                cfg.retrieval.max_per_source)
        cfg.retrieval.bm25_cache_enabled = _env_bool("BM25_CACHE_ENABLED",
                                                     cfg.retrieval.bm25_cache_enabled)
        cfg.retrieval.version_filter_enabled = _env_bool("VERSION_FILTER_ENABLED",
                                                         cfg.retrieval.version_filter_enabled)
        cfg.retrieval.relevance_gate = _env_float("RETRIEVAL_RELEVANCE_GATE",
                                                  cfg.retrieval.relevance_gate)
        gate_backends = _env_list("RETRIEVAL_RELEVANCE_GATE_BACKENDS",
                                  list(cfg.retrieval.relevance_gate_backends))
        cfg.retrieval.relevance_gate_backends = tuple(gate_backends)
        cfg.context_max_chars = _env_int("CONTEXT_MAX_CHARS", cfg.context_max_chars)

        # ---- 账号体系（注册/登录）----
        # 主名 AUTH_REQUIRED（与 UX 规范 §3 同名）；REQUIRE_AUTH 是**别名**（任务书里的名字）。
        # 两个名字都读；同时设置时以主名 AUTH_REQUIRED 为准（并记 warning，见 _auth_required_flag）。
        cfg.auth.require_auth = _auth_required_flag(cfg.auth.require_auth)
        cfg.auth.cookie_name = _env("AUTH_COOKIE_NAME", cfg.auth.cookie_name)
        cfg.auth.token_ttl_seconds = _env_int("AUTH_TOKEN_TTL_SECONDS",
                                              cfg.auth.token_ttl_seconds)
        cfg.auth.cookie_secure = _env_bool("AUTH_COOKIE_SECURE", cfg.auth.cookie_secure)
        cfg.auth.rate_limit_window_seconds = _env_int("AUTH_RATE_LIMIT_WINDOW_SECONDS",
                                                      cfg.auth.rate_limit_window_seconds)
        cfg.auth.rate_limit_max_requests = _env_int("AUTH_RATE_LIMIT_MAX_REQUESTS",
                                                    cfg.auth.rate_limit_max_requests)
        cfg.auth.recovery_code_bytes = _env_int("RECOVERY_CODE_BYTES",
                                                cfg.auth.recovery_code_bytes)

        # ---- 知识库删除权限（默认关闭 = 普通用户不能删资料）----
        cfg.allow_document_delete = _env_bool("ALLOW_DOCUMENT_DELETE",
                                              cfg.allow_document_delete)

        # ---- 日志 / 指标 ----
        cfg.logging.level = _env("LOG_LEVEL", cfg.logging.level)
        cfg.logging.dir = _env("LOG_DIR", cfg.logging.dir)
        cfg.logging.file_name = _env("LOG_FILE", cfg.logging.file_name)
        cfg.logging.to_file = _env_bool("LOG_TO_FILE", cfg.logging.to_file)
        cfg.logging.max_bytes = _env_int("LOG_MAX_BYTES", cfg.logging.max_bytes)
        cfg.logging.backup_count = _env_int("LOG_BACKUP_COUNT", cfg.logging.backup_count)
        cfg.logging.noisy_level = _env("LOG_NOISY_LEVEL", cfg.logging.noisy_level)
        cfg.logging.metrics_log_enabled = _env_bool("METRICS_LOG_ENABLED",
                                                    cfg.logging.metrics_log_enabled)
        cfg.logging.metrics_buckets = _env("METRICS_BUCKETS", cfg.logging.metrics_buckets)
        cfg.logging.trace_sample = _env_float("TRACE_SAMPLE", cfg.logging.trace_sample)
        # ---- P0.5 耗时评估（详见 docs/REFACTOR-PLAN.md §3.7）----
        cfg.logging.trace_level = _env("TRACE_LEVEL", cfg.logging.trace_level)
        cfg.logging.slow_call_ms = _env_float("SLOW_CALL_MS", cfg.logging.slow_call_ms)
        cfg.logging.slow_call_sample = _env_float("SLOW_CALL_SAMPLE",
                                                  cfg.logging.slow_call_sample)

        # ---- 系统指标（t6 用）----
        cfg.logging.sys_metrics_enabled = _env_bool("SYS_METRICS_ENABLED",
                                                    cfg.logging.sys_metrics_enabled)
        cfg.logging.sys_metrics_interval = _env_float("SYS_METRICS_INTERVAL",
                                                      cfg.logging.sys_metrics_interval)
        cfg.logging.sys_gpu_enabled = _env_bool("SYS_GPU_ENABLED", cfg.logging.sys_gpu_enabled)

        if _env("RAG_OFFLINE", "") == "1":
            # 离线模式：强制全部走兜底实现，不下载模型、不外呼
            cfg.vector_store = "memory"
            cfg.embedding_provider = "offline"
            cfg.rerank_provider = "cosine"
            cfg.llm_provider = "mock"
        return cfg

    # ---------------- 便捷派生值 ----------------

    @property
    def milvus_uri(self) -> str:
        """pymilvus 连接 URI：显式 MILVUS_URI 优先，否则由 host/port 拼。"""
        if self.milvus.uri:
            return self.milvus.uri
        return f"http://{self.milvus.host}:{self.milvus.port}"

    @property
    def upload_path(self) -> Path:
        path = Path(self.upload.upload_dir)
        return path if path.is_absolute() else self.project_root / path

    @property
    def log_path(self) -> Path:
        path = Path(self.logging.dir)
        base = path if path.is_absolute() else self.project_root / path
        return base / self.logging.file_name

    def with_overrides(self, **kwargs) -> "RagConfig":
        """派生一份改过个别字段的配置（测试/脚本用，不透传环境变量）。"""
        return replace(self, **kwargs)

    def ensure_dirs(self) -> None:
        """创建运行期需要的目录（索引、上传、日志）。"""
        for path in (self.index_dir, self.upload_path, self.log_path.parent):
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError:  # pragma: no cover - 只读环境不阻塞启动
                pass
