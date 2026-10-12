# 工单编号：人工智能NLP-RAG-混合检索任务
# 文件说明：接口冒烟测试。覆盖 PDF 解析入库、向量/全文/混合检索、三种融合、三种重排、布尔/短语/模糊、
#           中英文问答、多轮指代、缓存、反馈。用标准库 urllib，无需额外依赖；用例同时兼容 Windows。
import json, sys, time, urllib.request                                             # 标准库
from urllib.error import HTTPError                                                  # 错误处理

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8010"                # 允许命令行指定服务地址
PASS, FAIL = [], []                                                                 # 记录通过/失败用例

def call(path: str, payload=None, timeout=120):                                     # 统一调用
    """GET/POST 统一入口；返回 (状态码, 响应体)。异常不抛出，交由用例断言。"""
    url = BASE + path                                                               # 完整地址
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None  # 中文不转义
    req = urllib.request.Request(url, data=data,                                  # 构造请求
                                 headers={"Content-Type": "application/json; charset=utf-8"})  # UTF-8 头
    try:                                                                            # 网络异常也要转成可断言的返回值
        with urllib.request.urlopen(req, timeout=timeout) as r:                     # 发起请求
            return r.status, json.loads(r.read().decode("utf-8"))                   # 解析 JSON
    except HTTPError as e:                                                          # 4xx/5xx
        return e.code, json.loads(e.read().decode("utf-8") or "{}")                 # 返回错误体
    except Exception as e:                                                          # 连接失败等
        return 0, {"error": f"{type(e).__name__}: {e}"}                             # 返回 0 表示不可达

def case(name: str, ok: bool, detail: str = ""):                                    # 断言记录
    """记录用例结果并即时打印，避免中途异常导致结果丢失。"""
    (PASS if ok else FAIL).append(name)                                             # 归类
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)              # 即时输出

def main():                                                                         # 测试主流程
    print("=" * 78)                                                                 # 分隔
    print("阶段三 · 接口冒烟测试")                                                     # 标题
    print("=" * 78)                                                                 # 分隔
    # ---------- 1. 健康检查与配置 ----------
    st, h = call("/health")                                                         # 探活
    case("GET /health 可达且知识库非空", st == 200 and h.get("chunks", 0) > 0, f"chunks={h.get('chunks')}")
    case("Milvus 与 Redis 均连通", h.get("milvus") is True and h.get("redis") is True)
    st, cfg = call("/config")                                                       # 配置
    case("GET /config 返回 3 模式 3 融合 3 重排",
         st == 200 and len(cfg.get("modes", [])) == 3 and len(cfg.get("fusions", [])) == 3
         and len(cfg.get("rerankers", [])) == 3)
    # ---------- 2. 三种检索模式 ----------
    q = "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"           # 主测试问句
    for mode in ("vector", "bm25", "hybrid"):                                       # 三种模式
        st, r = call("/search", {"question": q, "mode": mode, "top_k": 8})           # 检索
        case(f"POST /search mode={mode}", st == 200 and r.get("count", 0) > 0, f"命中 {r.get('count')} 条 {r.get('degraded')}")
    # ---------- 3. 三种融合算法 ----------
    for fu in ("weighted", "rrf", "vote"):                                          # 三种融合
        st, r = call("/search", {"question": q, "mode": "hybrid", "fusion": fu, "top_k": 8})  # 检索
        ok = st == 200 and r.get("count", 0) > 0 and r["results"][0].get("rank_detail")      # 必须回带两路名次
        case(f"混合检索 fusion={fu}", ok, f"top1={r['results'][0]['chunk_id'] if r.get('results') else None}")
    # ---------- 4. 三种重排器（含级联顺序）----------
    for rk in (["bge"], ["tfidf"], ["adaptive"], ["tfidf", "bge"]):                  # 单用与级联
        st, r = call("/search", {"question": q, "mode": "hybrid", "rerankers": rk, "top_k": 8})  # 检索
        case(f"重排器 {'+'.join(rk)}", st == 200 and r.get("count", 0) > 0, f"degraded={r.get('degraded')}")
    # ---------- 5. 布尔 / 短语 / 模糊 ----------
    st, r1 = call("/search", {"question": "程家明 AND 法定代表人", "mode": "bm25", "top_k": 5})      # 布尔
    case("BM25 布尔查询 AND", st == 200 and r1.get("count", 0) > 0, f"命中 {r1.get('count')}")
    st, r2 = call("/search", {"question": '"本次股票发行后拟在科创板上市"', "mode": "bm25", "top_k": 5})  # 短语
    case("BM25 短语匹配", st == 200 and r2.get("count", 0) > 0, f"命中 {r2.get('count')}")
    st, r3 = call("/search", {"question": "程加明~", "mode": "bm25", "top_k": 5})                    # 模糊
    hit = any("程家明" in x.get("snippet", "") for x in r3.get("results", []))       # 模糊应能纠错召回
    case("BM25 模糊匹配（程加明~ → 程家明）", st == 200 and hit, f"命中 {r3.get('count')}")
    # ---------- 6. 问答：中文 / 英文 / 多轮 ----------
    st, c1 = call("/chat", {"question": q, "session_id": "test-zh", "top_k": 8})     # 中文
    case("中文问答命中关键数字", st == 200 and "6,464.51" in c1.get("answer", ""), f"{c1.get('latency_ms')}ms")
    st, c2 = call("/chat", {"question": "Who is the legal representative of Wuhan P&S Information Technology Co., Ltd.?",
                            "session_id": "test-en"})                                # 英文
    case("英文问答命中关键人物", st == 200 and "赵马克" in c2.get("answer", ""), f"{c2.get('latency_ms')}ms")
    call("/chat", {"question": "武汉兴图新科电子股份有限公司的法定代表人是谁？", "session_id": "test-mt"})  # 多轮第一问
    st, c3 = call("/chat", {"question": "那武汉力源信息技术股份有限公司呢？它的法定代表人是谁？", "session_id": "test-mt"})  # 指代
    case("多轮指代消解（切换到力源信息）", st == 200 and "赵马克" in c3.get("answer", ""),
         f"消解后问句={c3.get('retrieval_query', '')[:34]}")
    # ---------- 7. 缓存 ----------
    # 注意：缓存键里含"最近几轮历史"。因此**同一会话里连问两遍同一个问题不会命中**——
    # 第一遍的回答已经进入上下文，第二遍的历史不同、键就不同。这是设计使然，不是失效。
    # 要验证缓存本身，得让两次请求的上下文一致：用两个全新的空会话问同一个问题。
    try:                                                                            # 把两个会话的历史清空
        import redis as _r                                                          # 延迟导入
        _rc = _r.Redis.from_url("redis://localhost:6379/0", protocol=2,             # 与生产同参数：Redis 5.x 必须 RESP2
                                decode_responses=True, socket_timeout=3)            # 短超时
        _rc.delete("gd6:hist:test-cache-a", "gd6:hist:test-cache-b")                # 清掉残留历史
    except Exception:                                                               # Redis 不可用
        pass                                                                        # 忽略，退化为不校验
    qn = f"{q}（缓存自检 {int(time.time())}）"                                        # 加随机串，保证首次一定是未命中
    body_a = {"question": qn, "session_id": "test-cache-a", "top_k": 8}              # 会话 A
    body_b = {"question": qn, "session_id": "test-cache-b", "top_k": 8}              # 会话 B（同问句、同空历史 → 同键）
    t0 = time.time(); call("/chat", body_a); first = (time.time() - t0) * 1000        # 首次：真跑
    t0 = time.time(); st, c4 = call("/chat", body_b); second = (time.time() - t0) * 1000  # 二次：应命中
    case("Redis 缓存命中（同上下文同问句）", c4.get("cached") is True, f"首次 {first:.0f}ms → 二次 {second:.0f}ms")
    # ---------- 8. 反馈 ----------
    st, f1 = call("/feedback", {"chunk_id": "招股说明书1-00418", "doc": "招股说明书1", "rating": 1})  # 点赞
    case("POST /feedback 写入成功", st == 200 and f1.get("ok") is True)
    st, f2 = call("/feedback", {"chunk_id": "x", "doc": "y", "rating": 9})            # 非法值
    case("POST /feedback 非法 rating 被拒", st == 400, f"HTTP {st}")
    # ---------- 9. 结果片段结构 ----------
    st, r = call("/search", {"question": "军用领域收入", "mode": "hybrid", "top_k": 5})  # 结构校验
    item = (r.get("results") or [{}])[0]                                             # 首条
    case("检索结果含出处/页码/通道/分数",
         all(k in item for k in ("chunk_id", "doc", "page", "channel", "score")),
         f"字段={sorted(item.keys())[:7]}")
    # ---------- 汇总 ----------
    print("=" * 78)                                                                 # 分隔
    print(f"通过 {len(PASS)} / {len(PASS) + len(FAIL)}")                              # 统计
    if FAIL:                                                                        # 有失败则列出
        print("失败用例：")                                                           # 标题
        for f in FAIL:                                                              # 逐条
            print(f"  - {f}")                                                       # 打印
    return 1 if FAIL else 0                                                         # 返回退出码

if __name__ == "__main__":                                                          # 允许直接运行
    sys.exit(main())                                                                # 退出码供 CI 使用
