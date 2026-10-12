# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
# 文件说明：容错注入测试（工单"容错机制"验收项）。**真的把依赖打断**再观察行为，而不是只读代码判断有没有 try。
# 用法：python tests/test_faults.py
import sys                                                                   # 退出码
from pathlib import Path                                                     # 路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))               # 允许 import 项目根模块
import config as C                                                           # 集中配置


def fresh():                                                                 # 清空进程内状态，模拟冷启动
    from backend import rag as R                                             # 延迟导入
    R.STATE.update(bm25=None, chunks=[], client=None, tried=0.0)              # 复位索引状态
    return R                                                                 # 返回模块


def run(name: str, fn) -> bool:                                              # 单个场景执行器
    """执行一个注入场景，捕获任何异常——**抛异常即为失败**：容错的意义就是不让异常冒到接口层。"""
    try:                                                                     # 场景执行
        detail = fn()                                                        # 返回说明字符串
        print(f"  ✓ {name}\n      {detail}")                                  # 通过
        return True                                                          # 记通过
    except Exception as e:                                                   # 未兜住的异常
        print(f"  ✗ {name}\n      抛出未处理异常 {type(e).__name__}: {e}")       # 失败
        return False                                                         # 记失败


def scenario_milvus_down() -> str:                                           # 场景 1：Milvus 不可用
    """把地址改到不存在的端口，验证能退回 BM25 单通道作答，而不是整个问答失败。"""
    old = C.MILVUS_URI                                                        # 保存原配置
    from backend import store as V                                           # 延迟导入
    old_client = V.milvus_client                                             # 保存原函数
    V.milvus_client = lambda: (_ for _ in ()).throw(ConnectionError("模拟 Milvus 宕机"))  # 强制构造即抛异常
    C.MILVUS_URI = "http://127.0.0.1:1"                                      # 指向死端口
    try:                                                                     # 执行
        R = fresh()                                                          # 冷启动
        # 关键点：MilvusClient() 在**构造阶段**就抛异常，若不在 load 里兜住，异常会直接冒到接口层，
        # 向量检索内部的兜底分支根本走不到，结果是整个问答失败而非退回 BM25。这条以前真的漏过。
        r = R.answer("武汉力源信息技术股份有限公司注册资本是多少？", "zh", use_cache=False)  # 提问
        assert r.get("answer") and not str(r.get("answer")).startswith("系统繁忙"), f"答案异常：{r.get('answer')}"  # 必须有可用答案
        assert r.get("retrieval_mode") in ("bm25_only", None), f"检索通道异常：{r.get('retrieval_mode')}"  # 只能是 BM25 通道
        return f"检索通道={r.get('retrieval_mode')}，降级提示={r.get('degraded_reason') or '（无）'}，仍返回答案 {len(r['answer'])} 字"  # 说明
    finally:                                                                 # 还原
        V.milvus_client, C.MILVUS_URI = old_client, old                      # 恢复配置


def scenario_redis_down() -> str:                                            # 场景 2：Redis 不可用
    """缓存全挂也不能影响问答：读缓存恒未命中、写缓存静默失败。"""
    from backend import cache as K                                           # 延迟导入
    old_url = C.REDIS_URL                                                    # 保存配置
    C.REDIS_URL, K._CLIENT = "redis://127.0.0.1:1/0", None                    # 指向死端口并清客户端
    try:                                                                     # 执行
        assert K.ping().startswith("error"), "探测应返回 error 而不是抛异常"      # 探测不抛栈
        R = fresh()                                                          # 冷启动
        r = R.answer("武汉兴图新科电子股份有限公司法定代表人是谁？", "zh")            # 提问（含缓存读写）
        assert r.get("answer"), "Redis 挂掉后仍应给出答案"                       # 必须有答案
        return f"探测={K.ping()}，问答正常返回 {len(r['answer'])} 字，缓存键未生效但不阻断主流程"  # 说明
    finally: C.REDIS_URL, K._CLIENT = old_url, None                          # 还原


def scenario_llm_down() -> str:                                              # 场景 3：大模型不可用
    """Key 失效 / 限流 / 余额不足（402）/ 超时：应把检索原文交回用户，而不是只回一句报错。"""
    from backend import rag as R                                             # 延迟导入
    old = R.generate                                                         # 保存原函数
    R.generate = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("模拟 DeepSeek 402 余额不足"))  # 强制失败
    try:                                                                     # 执行
        r = R.answer("武汉力源信息技术股份有限公司本次发行股数是多少？", "zh", use_cache=False)  # 提问
        assert r.get("contexts"), "生成失败时仍应返回检索到的上下文"                 # 上下文必须保留
        assert r.get("degraded") and "生成服务不可用" in (r.get("degraded_reason") or ""), "应显式标注生成降级"  # 降级必须可见
        return f"降级原因={r['degraded_reason']}；同时返回 {len(r['contexts'])} 条原文供用户自查"  # 说明
    finally: R.generate = old                                                # 还原


def scenario_embed_down() -> str:                                            # 场景 4：向量模型不可用
    """显存不足或模型路径写错时，向量通道应空手而归，由 BM25 撑住。"""
    from backend import store as V                                           # 延迟导入
    old = V.embed                                                            # 保存原函数
    V.embed = lambda *a, **k: (_ for _ in ()).throw(OSError("模拟向量模型路径不存在"))  # 强制失败
    try:                                                                     # 执行
        R = fresh()                                                          # 冷启动（避免命中已加载的模型）
        r = R.answer("武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？", "zh", use_cache=False)  # 提问
        assert r.get("answer"), "向量模型不可用时仍应有答案"                       # 必须有答案
        return f"检索通道={r.get('retrieval_mode')}（向量通道已空），降级说明={r.get('degraded_reason') or '（无）'}"  # 说明
    finally: V.embed = old                                                   # 还原


def scenario_rerank_down() -> str:                                           # 场景 5：重排模型不可用
    """重排挂了必须按融合分兜底排序，而不是返回空上下文。"""
    from backend import store as V                                           # 延迟导入
    old_path = C.RERANK_MODEL_PATH                                           # 保存配置
    C.RERANK_MODEL_PATH, V._RERANK = "D:/不存在的模型路径", None               # 指向错误路径并清单例
    try:                                                                     # 执行
        docs = [dict(text=f"候选{i}", chunk_id=f"c{i}", score=1.0 - i * 0.1, page_no=1) for i in range(5)]  # 假候选
        out = V.rerank("测试问题", docs, 3)                                     # 重排
        assert len(out) == 3, f"应兜底返回 3 条，实得 {len(out)}"                  # 必须仍有输出
        assert all("rerank_score" in d for d in out), "应回填分数"                # 分数必须存在
        return f"模型缺失时按融合分兜底排序，返回 {len(out)} 条，首条分数 {out[0]['rerank_score']}"  # 说明
    finally: C.RERANK_MODEL_PATH, V._RERANK = old_path, None                  # 还原


def scenario_index_missing() -> str:                                         # 场景 6：索引文件缺失
    """未入库时应提示"请先执行 scripts/ingest.py"，而不是丢一个 FileNotFoundError 栈。"""
    from backend import rag as R                                             # 延迟导入
    old = C.INDEX_FILE                                                       # 保存配置
    C.INDEX_FILE = Path("D:/不存在的索引.json")                                 # 指向不存在的索引
    try:                                                                     # 执行
        r = fresh().answer("任意问题", "zh", use_cache=False)                   # 提问
        assert r.get("error") == "index_missing", f"错误码异常：{r.get('error')}"   # 必须识别为索引缺失
        return f"错误码={r['error']}，提示：{r['answer']}"                        # 说明
    finally: C.INDEX_FILE = old                                              # 还原


def scenario_bad_input() -> str:                                             # 场景 7：用户输入错误
    """空问题、超长问题、纯符号等异常输入都要给出中文友好提示。"""
    R = fresh()                                                              # 延迟导入
    cases = [("", "empty_question"), ("   ", "empty_question"), ("啊" * 600, "query_too_long")]  # 三类坏输入
    got = []                                                                 # 结果
    for q, want in cases:                                                    # 逐个
        r = R.answer(q, "zh", use_cache=False)                                # 提问
        assert r.get("error") == want, f"{q[:8]!r} 期望 {want}，实得 {r.get('error')}"  # 必须命中预期错误码
        assert r["answer"] and not r["answer"].startswith("系统繁忙"), "应给出具体中文提示"  # 提示要具体
        got.append(f"{want}→{r['answer'][:22]}")                              # 记录
    return "；".join(got)                                                    # 说明


def main() -> int:                                                           # 入口
    print("容错注入测试（真打断依赖，观察是否兜住）\n")                              # 标题
    cases = [("Milvus 不可用 → 退回 BM25", scenario_milvus_down),               # 各场景
             ("Redis 不可用 → 缓存静默失效", scenario_redis_down),               # 缓存
             ("DeepSeek 不可用（402/超时）→ 返回检索原文", scenario_llm_down),     # 生成
             ("向量模型不可用 → BM25 撑住", scenario_embed_down),                 # 嵌入
             ("重排模型不可用 → 融合分兜底", scenario_rerank_down),                # 重排
             ("索引缺失 → 提示重建", scenario_index_missing),                    # 索引
             ("输入错误（空/超长）→ 友好提示", scenario_bad_input)]               # 输入
    oks = [run(n, f) for n, f in cases]                                      # 执行
    ok = sum(oks)                                                            # 通过数
    print(f"\n结果：{ok}/{len(cases)} 个容错场景通过")                             # 汇总
    if ok < len(cases): print("未通过项必须修复后，才可声明「高可用、容错」达标。")       # 提醒
    return 0 if ok == len(cases) else 1                                      # 退出码


if __name__ == "__main__":                                                    # 入口
    sys.exit(main())                                                         # 执行
