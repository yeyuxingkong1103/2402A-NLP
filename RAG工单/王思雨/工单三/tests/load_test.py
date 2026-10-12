# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 文件说明：并发压测脚本。验证工单 P8「高并发环境下稳定运行」，记录 P50/P95/错误率。
#   用法：python tests/load_test.py [并发数] [缓存模式秒数] [不缓存请求数]
#   注意：用 UTF-8 客户端发中文，不要用 Git Bash 的 curl（会乱码，见 docs/05_deploy.md）
import http.client                                                                # 用 http.client 而非 urllib：urllib 每次请求都新建连接
import json                                                                       # 构造请求体与解析响应
import statistics as st                                                           # 分位数统计
import sys                                                                        # 读取命令行参数
import threading                                                                  # 多线程模拟并发
import time                                                                       # 计时与控制压测时长

HOST, PORT = "127.0.0.1", 8000                                                    # 后端地址
QUESTIONS = [                                                                     # 覆盖两篇文档与表格/正文两类来源
    "武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？",
    "武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目？",
    "与武汉力源信息技术股份有限公司存在控制关系的关联方是谁，持股比例和本公司关系是什么？",
    "与武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些？",
    "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？",
    "武汉兴图新科电子股份有限公司法定代表人是谁？",
    "武汉兴图新科电子股份有限公司注册资本是多少？",
    "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？",
]


class Client:                                                                     # 每个并发工人一条长连接
    """用 keep-alive 长连接发请求。

    坑：一开始用 urllib.urlopen，它**每次请求都新建 TCP 连接**，10 并发跑 60 秒开了 4.2 万个连接，
    直接把 Windows 的临时端口耗尽，报 `WinError 10048 通常每个套接字地址只允许使用一次`，
    错误率显示 60%——**那是客户端自己造成的，与服务端无关**。10 个并发工人就是 10 条连接。
    """
    def __init__(self):                                                            # 建长连接
        self.conn = http.client.HTTPConnection(HOST, PORT, timeout=120)            # 每工人一条，不每次新建
        self.retrieval_ms = self.generate_ms = 0                                   # 最近一次请求的分段耗时

    def ask(self, question: str, use_cache: bool) -> tuple:                        # 发一次提问
        """返回 (耗时秒, 错误信息)；错误信息为 None 表示成功。"""
        body = json.dumps(dict(question=question, use_cache=use_cache)).encode("utf-8")  # UTF-8 编码，避免中文乱码
        t0 = time.time()                                                           # 计时起点
        try:                                                                       # 连接被服务端断开时重连一次再试
            self.conn.request("POST", "/ask", body, {"Content-Type": "application/json"})  # 复用连接发请求
            data = json.loads(self.conn.getresponse().read().decode("utf-8"))        # 读响应并解析
            self.retrieval_ms = data.get("retrieval_ms") or 0                          # 记录检索段耗时，用于区分瓶颈在 GPU 还是大模型
            self.generate_ms = data.get("generate_ms") or 0                            # 记录生成段耗时
            if data.get("error") or not (data.get("answer") or "").strip():            # 业务层失败（HTTP 仍是 200）
                return time.time() - t0, f"业务错误: {data.get('error')}"              # 记为失败
            return time.time() - t0, None                                             # 成功
        except Exception as exc:                                                    # 连接中断或超时
            try: self.conn.close()                                                   # 丢掉坏连接
            except Exception: pass                                                   # 关闭失败不掩盖原错误
            self.conn = http.client.HTTPConnection(HOST, PORT, timeout=120)           # 重连，下一轮继续可用
            return time.time() - t0, f"{type(exc).__name__}: {exc}"                    # 记为失败


def run(workers: int, seconds: int, use_cache: bool, limit: int = 0) -> dict:        # 压测一轮
    """按并发数起线程持续打请求，到时长或达到请求上限为止，返回耗时样本与错误。"""
    times, errors, lock = [], [], threading.Lock()                                  # 共享结果与锁
    ret, gen = [], []                                                               # 检索段与生成段耗时样本
    stop = threading.Event()                                                        # 停止信号
    counter = [0]                                                                   # 已发请求计数

    def worker(idx: int) -> None:                                                   # 单个并发工人
        cli, i = Client(), idx                                                      # 从不同问题起步，避免所有线程打同一题
        while not stop.is_set():                                                    # 未到时间就继续
            with lock:                                                              # 计数与上限判断要互斥
                if limit and counter[0] >= limit: stop.set(); return                 # 达到请求上限就停
                counter[0] += 1                                                     # 计数 +1
            dt, err = cli.ask(QUESTIONS[i % len(QUESTIONS)], use_cache)              # 发一次请求
            with lock:                                                              # 汇总结果
                times.append(dt)                                                     # 记录耗时
                if err: errors.append(err)                                           # 记录错误
                if cli.retrieval_ms: ret.append(cli.retrieval_ms)                    # 记录检索段（GPU 侧）
                if cli.generate_ms: gen.append(cli.generate_ms)                      # 记录生成段（在线 API 侧）
            i += 1                                                                  # 换下一题
        cli.conn.close()                                                            # 收工关连接

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(workers)]  # 建线程
    t0 = time.time()                                                                # 起点
    for t in threads: t.start()                                                      # 全部启动
    while time.time() - t0 < seconds and not stop.is_set(): time.sleep(0.2)           # 等到时长或提前停止
    stop.set()                                                                       # 通知收工
    for t in threads: t.join(timeout=600)                                            # 等线程退出（含在途请求）
    wall = time.time() - t0                                                          # 实际墙钟时长

    if not times: return dict(workers=workers, requests=0, error=len(errors))         # 极端情况保护
    s = sorted(times)                                                                # 排序后取分位
    def pct(p): return round(s[min(len(s) - 1, int(len(s) * p))], 3)                  # 分位数取值
    def stage(v):                                                                   # 分段耗时的分位数
        if not v: return {}                                                         # 缓存命中时没有分段数据
        x = sorted(v)                                                               # 排序
        return dict(p50=round(x[len(x) // 2], 1), p95=round(x[min(len(x) - 1, int(len(x) * 0.95))], 1))  # 中位与 95 分位
    return dict(workers=workers, use_cache=use_cache, requests=len(times),            # 基本规模
                wall_s=round(wall, 1), rps=round(len(times) / wall, 1),               # 墙钟与吞吐
                p50=pct(0.50), p95=pct(0.95), max=round(s[-1], 3), mean=round(st.mean(s), 3),  # 端到端耗时分布
                retrieval_ms=stage(ret), generate_ms=stage(gen),                       # 分段：分别对应 GPU 侧与在线大模型侧
                error=len(errors), error_rate=round(len(errors) / len(times), 4),      # 错误率
                samples=errors[:3])                                                   # 错误样例，便于定位


if __name__ == "__main__":                                                           # 命令行入口
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 10                               # 并发数，默认 10
    sec = int(sys.argv[2]) if len(sys.argv) > 2 else 60                             # 压测秒数，默认 60
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 60                               # 不缓存模式的请求上限
    print("=== 第 1 轮：缓存开启（测系统在稳态下的并发承载）===")                          # 说明这一轮测什么
    print(json.dumps(run(w, sec, True), ensure_ascii=False))                          # 缓存轮，不受大模型拖累
    print("=== 第 2 轮：缓存关闭（测真实链路，受在线大模型吞吐约束）===")                     # 说明这一轮测什么
    print(json.dumps(run(w, 600, False, limit=n), ensure_ascii=False))                # 不缓存轮，用请求数封顶控制成本
