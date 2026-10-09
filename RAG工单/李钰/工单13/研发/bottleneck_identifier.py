# -*- coding: utf-8 -*-
"""
瓶颈识别器 + 优化策略
工单编号: 人工智能 NLP-RAG 项目-RAG 性能瓶颈识别与优化

6 大瓶颈 + 对应优化:
  A. Query 处理慢    → 跳过 rewrite / 规则 rewrite / 缓存
  B. 向量检索慢      → FAISS / IVF / 缓存 / Top-K 剪枝
  C. 上下文组装慢    → 预拼接模板 / 去重哈希 / 剪枝
  D. LLM 生成慢      → 小模型 / 流式 / 缓存 / 减少 tokens
  E. 后处理慢        → 异步 / 规则 / 缓存
  X. 全流程串行      → asyncio / 多线程并行
"""
import os, sys, time, json, hashlib, threading, logging, asyncio
from typing import List, Dict, Any, Callable, Optional
from functools import wraps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_v13 as config

logger = logging.getLogger(__name__)


# ============ 瓶颈诊断 ============

BOTTLENECK_DIAGNOSIS = {
    # 阶段名 → (阈值 ms, 瓶颈类型, 推荐优化)
    "query_process": (200, "A", ["query_skip", "query_cache"]),
    "retrieve": (500, "B", ["index_cache", "topk_prune"]),
    "context_assemble": (200, "C", ["context_prune", "dedup_hash"]),
    "llm_generate": (1000, "D", ["small_model", "llm_cache", "reduce_tokens"]),
    "post_process": (100, "E", ["async_post", "post_cache"]),
}


def diagnose_bottlenecks(metrics_summary: Dict) -> List[Dict]:
    """根据性能指标诊断瓶颈"""
    bottlenecks = []
    for stage_name, (threshold, btype, optims) in BOTTLENECK_DIAGNOSIS.items():
        metric_key = f"stage.{stage_name}"
        stats = metrics_summary.get(metric_key)
        if not stats:
            continue
        mean_ms = stats["mean"]
        if mean_ms > threshold:
            severity = "🔴 严重" if mean_ms > threshold * 3 else (
                "🟡 中等" if mean_ms > threshold * 1.5 else "🟢 轻微"
            )
            bottlenecks.append({
                "stage": stage_name,
                "type": btype,
                "severity": severity,
                "mean_ms": mean_ms,
                "threshold_ms": threshold,
                "recommendation": optims,
                "improvement_potential": round((mean_ms - threshold) / mean_ms * 100, 1),
            })

    # 按严重度排序
    bottlenecks.sort(key=lambda x: x["mean_ms"], reverse=True)
    return bottlenecks


# ============ 优化策略 ============

class SimpleCache:
    """线程安全简单缓存 (带 TTL)"""
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.Lock()
        self.max_size = max_size
        self.ttl = ttl
        self.hits = 0
        self.misses = 0

    def _make_key(self, *args, **kwargs) -> str:
        raw = json.dumps({"a": [str(a) for a in args], "k": {k: str(v) for k, v in kwargs.items()}},
                         sort_keys=True, default=str)
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                if time.time() - self._timestamps[key] < self.ttl:
                    self.hits += 1
                    return self._cache[key]
                else:
                    del self._cache[key]
                    del self._timestamps[key]
            self.misses += 1
            return None

    def set(self, key: str, value: Any):
        with self._lock:
            if len(self._cache) >= self.max_size:
                # LRU: 删最老的
                oldest = min(self._timestamps, key=self._timestamps.get)
                del self._cache[oldest]
                del self._timestamps[oldest]
            self._cache[key] = value
            self._timestamps[key] = time.time()

    def cached(self, func: Callable):
        """装饰器: 自动缓存"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = self._make_key(func.__name__, *args, **kwargs)
            val = self.get(key)
            if val is not None:
                logger.info(f"[Cache HIT] {func.__name__}")
                return val
            result = func(*args, **kwargs)
            self.set(key, result)
            return result
        return wrapper

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# ============ 各种优化器 ============

class Optimizer:
    """优化器基类"""
    name = "base"
    description = ""

    def apply(self, data: Any) -> Any:
        raise NotImplementedError


class CacheOptimizer(Optimizer):
    """优化 1: 结果缓存 (Query 增强 / 检索结果 / LLM 回答)"""
    name = "result_cache"
    description = "缓存 query rewrite + 检索结果 + LLM 回答, 相同 query 秒返回"

    def __init__(self):
        self.cache = SimpleCache(max_size=config.CACHE_MAX_SIZE, ttl=config.CACHE_TTL)

    def apply(self, query: str) -> Optional[str]:
        """尝试命中缓存"""
        return self.cache.get(query)

    def store(self, query: str, result: Any):
        self.cache.set(query, result)


class TopKPruningOptimizer(Optimizer):
    """优化 2: Top-K 剪枝"""
    name = "topk_prune"
    description = "检索后只留 Top-K (默认 5, 基线是 10), 减少上下文长度"

    def __init__(self, k: int = None):
        self.k = k or config.OPT_TOP_K

    def apply(self, chunks: List[Dict]) -> List[Dict]:
        if len(chunks) <= self.k:
            return chunks
        return chunks[:self.k]


class ContextPruningOptimizer(Optimizer):
    """优化 3: 上下文剪枝 (去冗余 + 长度限制)"""
    name = "context_prune"
    description = "去除重复/相似 chunk, 限制总 token 数, 避免 LLM 处理超长上下文"

    def __init__(self, max_chars: int = 4000):
        self.max_chars = max_chars

    def apply(self, chunks: List[Dict]) -> List[Dict]:
        result = []
        seen_hash = set()
        total = 0
        for c in chunks:
            text = c.get("text", str(c))
            # 去重哈希
            h = hashlib.md5(text.encode()).hexdigest()[:8]
            if h in seen_hash:
                continue
            seen_hash.add(h)

            if total + len(text) > self.max_chars:
                break
            result.append(c)
            total += len(text)
        return result


class ParallelOptimizer(Optimizer):
    """优化 4: 并行处理 (asyncio / ThreadPool)"""
    name = "parallel"
    description = "Query 处理 + 检索 并行; 后处理异步"

    def apply(self, funcs: List[Callable]) -> List[Any]:
        """并行执行多个函数"""
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(funcs), 4)) as pool:
            return list(pool.map(lambda f: f(), funcs))


class SmallModelOptimizer(Optimizer):
    """优化 5: 小模型替代"""
    name = "small_model"
    description = "用小模型 (或规则) 做 Query rewrite, 跳过大型 LLM"

    def apply(self, query: str) -> List[str]:
        """规则式 query 扩展, 0ms"""
        expansions = [query]
        # 简单同义词扩展 (规则, 无 LLM)
        synonyms = {
            "注册资本": ["股本", "实缴资本"],
            "控股股东": ["控制方", "最大股东"],
            "募集资金": ["募资", "融资"],
        }
        for kw, alts in synonyms.items():
            if kw in query:
                for alt in alts:
                    expansions.append(query.replace(kw, alt))
        return expansions


class AsyncPostOptimizer(Optimizer):
    """优化 6: 异步后处理"""
    name = "async_post"
    description = "后处理 (格式化/引用) 异步执行, 不阻塞主流程"

    def apply(self, answer: str, chunks: List[Dict]) -> str:
        """轻量规则后处理"""
        return answer.strip() + "\n\n[检索来源: {} 段文本]".format(len(chunks))


# ============ 优化方案组合 ============

def build_optimizers() -> Dict[str, Optimizer]:
    """根据 config 构建优化器集合"""
    opts = {}
    if config.USE_CACHE:
        opts["cache"] = CacheOptimizer()
    if config.USE_CONTEXT_PRUNING:
        opts["context_prune"] = ContextPruningOptimizer()
    if config.USE_PARALLEL:
        opts["parallel"] = ParallelOptimizer()
    if config.USE_SMALL_MODEL:
        opts["small_model"] = SmallModelOptimizer()
    if config.USE_QUERY_SKIP:
        # skip = 直接返回, 不做任何处理
        opts["query_skip"] = None  # 标记
    opts["async_post"] = AsyncPostOptimizer()
    return opts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 缓存测试
    c = CacheOptimizer()
    print(f"Query 扩展: {SmallModelOptimizer().apply('武汉力源的注册资本是多少?')}")

    # 剪枝测试
    chunks = [{"text": f"chunk_{i}", "score": 10 - i} for i in range(10)]
    pruned = TopKPruningOptimizer(k=3).apply(chunks)
    print(f"Top-K 剪枝: {len(chunks)} → {len(pruned)}")

    # 诊断测试
    from performance_monitor import record_metric, get_metrics_summary, reset_metrics
    reset_metrics()
    record_metric("stage.query_process", 100)
    record_metric("stage.retrieve", 800)
    record_metric("stage.llm_generate", 2500)
    summary = get_metrics_summary()
    bottlenecks = diagnose_bottlenecks(summary)
    print("\n=== 瓶颈诊断 ===")
    for b in bottlenecks:
        print(f"  {b['severity']} {b['stage']}: mean={b['mean_ms']:.0f}ms "
              f"(阈值{b['threshold_ms']}ms) → {b['recommendation']}")
