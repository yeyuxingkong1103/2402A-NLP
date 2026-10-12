# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
# 文件说明：工单 16 题批量测试。逐题调用问答接口，输出答案/来源/检索精确度/耗时，汇总准确率与响应时间。
# 用法：python scripts/test_questions.py            （走 HTTP /api/ask，需先启动服务）
#       python scripts/test_questions.py --direct   （直连流水线，免起服务，便于快速回归）
#       python scripts/test_questions.py --selfcheck （用已知错误答案回验判分器本身）
import argparse, json, re, statistics, sys, time                             # 参数、序列化、判分、统计、计时
from datetime import datetime                                                # 结果文件时间戳
from pathlib import Path                                                     # 路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))               # 允许 import 项目根模块
import config as C                                                           # 集中配置

RESULT_DIR = C.ROOT / "results"                                              # 结果目录
API = "http://localhost:8000/api/ask"                                        # 接口地址


def load_items() -> list:                                                    # 读取题库
    return json.loads(C.QUESTIONS.read_text(encoding="utf-8"))                # 含问题、标准答案、关键词与负样本


def hit(rate: float, text: str, keys: list) -> tuple:                        # 关键词命中判定
    """返回 (是否通过, 命中数, 缺失词)。大小写与全半角差异先归一，避免"１，６７０"这类因格式判错。"""
    t = re.sub(r"[\s，,]", "", (text or "").lower()).replace("％", "%")        # 归一化：去空格与逗号、统一百分号
    miss = [k for k in keys if re.sub(r"[\s，,]", "", k.lower()).replace("％", "%") not in t]  # 缺失关键词
    return (len(keys) - len(miss)) / len(keys) >= rate, len(keys) - len(miss), miss  # 是否达到命中率、命中数与缺失项


_FRAG = re.compile(r"[一-鿿]([a-z]{2,8})(?=[一-鿿]|[\W_]|$)")               # 紧贴中文的短小写拉丁串


def dirty(text: str) -> list:                                                # ★实体污染检测
    """答案里出现紧贴中文的小写拉丁碎片（如"赵马克gzh"）即判为被水印污染。

    为什么单列一条判据：这类错**关键词判分完全抓不到**——"赵马克"照样命中关键词，判分给过，
    但实体名已经错了。实测工单四 id=3 就是这样：答案写成"赵马克gzh"，三重判据全过。
    判据取"紧贴中文 + 全小写"：正规术语（IC / ERP / GDP / BOM）是全大写，不会被误判。
    """
    return [m.group(1) for m in _FRAG.finditer(text or "")]                   # 返回污染碎片


def judge(item: dict, answer: str, sources: list) -> dict:                    # 单题判分
    """四重判据：必含关键词命中率、负样本不得出现、枚举题须数够个数、答案实体名不得被污染。

    只查"该有的有没有"会漏掉**多答无关内容**——实测工单三自动判分 14/14 满分，
    人工复核才发现答案里混进了别家公司；故此处必须叠加 expect_not 负样本判据。
    """
    keys = item.get("keywords") or []                                        # 必含关键词
    need = item.get("keywords_min_rate", 0.6)                                # 命中率门槛
    ok_kw, n_hit, miss = hit(need, answer, keys) if keys else (True, 0, [])   # 关键词判据
    bad = [b for b in (item.get("expect_not") or []) if b and b.lower() in (answer or "").lower()]  # 负样本判据
    # 枚举题额外核对个数：模型常把 6 个销售处写成 5 个，关键词全中也会错
    cnt, cnt_ok = item.get("expect_count"), True                              # 期望条目数
    if cnt:                                                                  # 只有配置了才检查
        for pat, want in cnt.items():                                        # pat=正则，want=期望出现次数
            cnt_ok = cnt_ok and len(re.findall(pat, answer or "")) >= want     # 出现次数不足即判错
    frag = dirty(answer)                                                     # ★实体名污染判据
    wrong = bool(bad) or not ok_kw or not cnt_ok or bool(frag)                # 任一判据不过即判错
    return dict(correct=not wrong, keyword_hit=n_hit, keyword_total=len(keys),  # 判分结果与依据
                keyword_miss=miss, negative_hit=bad,                        # 缺失词与命中的负样本
                count_ok=cnt_ok, dirty_frag=frag, source_hit=bool(sources))   # 枚举个数、污染碎片与是否有来源


def selfcheck() -> int:                                                      # 判分器自检
    """用**已知错误的答案**回验判分器：若错答也被判为正确，说明判据失灵，必须先修判分器再谈准确率。"""
    items = load_items()                                                     # 题库
    samples = [("销售部下设 4 个部门，大客户销售部下设 5 个销售处", False),          # 数量错（已知错答，必须判错）
               ("资料中未找到相关信息", False),                                    # 拒答（必须判错）
               ("销售部下设渠道销售部、电话及网络销售部、大客户销售部和国际贸易部，共 4 个部门；"
                "大客户销售部下设珠海销售处、深圳销售处、北京销售处、武汉销售处、广州销售处、成都销售处，"
                "共 6 个销售处", True),                                            # 全对（必须判对）
               ("销售部由 4 个部门构成，大客户销售部有 6 个销售处；另外武汉兴图新科的注册资本是 5,520 万元", False),  # 多答无关企业（工单三实测漏判的形态）
               ("与武汉力源信息技术股份有限公司存在控制关系的关联方是赵马克gzh，持股比例为42.35%", False)]  # 实体名被水印污染：关键词照样命中，三重判据全过，必须由污染判据拦下
    bad = 0                                                                  # 失误计数
    for text, want in samples:                                               # 逐样本
        got = judge(items[0], text, [{"doc": "x"}])["correct"]                # 判分
        flag = "✓" if got == want else "✗ 判分器失灵"                          # 是否与预期一致
        print(f"  自检 {flag}  期望={want} 实得={got}  «{text[:34]}…»")          # 输出
        bad += got != want                                                   # 统计失误
    print(f"判分器自检：{'通过' if not bad else f'{bad} 例失灵，请先修判分逻辑'}")     # 结论
    return 1 if bad else 0                                                   # 失灵则返回非零


def direct(question: str, lang: str, no_cache: bool = False) -> dict:        # 直连流水线
    from backend import rag as R                                             # 延迟导入，避免 --selfcheck 时加载重依赖
    r = R.answer(question, lang, use_cache=not no_cache)                     # 走完整问答
    return dict(answer=r["answer"], sources=r["contexts"],                    # 对齐接口字段
                retrieval_precision=r.get("retrieval_precision", 0.0),        # 检索精确度
                elapsed_ms=round(r.get("elapsed", 0) * 1000),                 # 端到端耗时
                retrieval_mode=r.get("retrieval_mode"), degraded=r.get("degraded"),  # 通道与降级
                cached=r.get("cached", False))                               # 是否命中缓存


def ask(question: str, lang: str, use_http: bool, no_cache: bool = False) -> dict:  # 调用问答
    """默认走 HTTP /api/ask（与工单"调用接口"口径一致）；--direct 时直连流水线。

    缓存旁路必须显式传给**服务端**：以前 `--no-cache` 只在测试进程里给 K.get/K.set 打补丁，
    而 HTTP 模式下缓存跑在服务端，于是 16 题全部命中缓存、耗时 3 ms，性能数据失真却毫无提示。
    """
    if not use_http: return direct(question, lang, no_cache)                   # 直连
    import urllib.request                                                     # 标准库，免额外依赖
    payload = dict(question=question, lang=lang, no_cache=no_cache)           # JSON 入参（含缓存旁路）
    req = urllib.request.Request(API,                                        # 构造请求
                                 data=json.dumps(payload).encode("utf-8"),     # 入参
                                 headers={"Content-Type": "application/json"})  # 头部
    with urllib.request.urlopen(req, timeout=120) as r:                       # 发起
        return json.loads(r.read().decode("utf-8"))                           # 解析响应


def main() -> None:                                                          # 批量测试主流程
    ap = argparse.ArgumentParser()                                           # 命令行
    ap.add_argument("--direct", action="store_true", help="直连流水线，不走 HTTP")  # 免起服务
    ap.add_argument("--selfcheck", action="store_true", help="只回验判分器")        # 判分器自检
    ap.add_argument("--no-cache", action="store_true", help="绕过缓存，测真实端到端耗时")  # 压测耗时用
    ap.add_argument("--lang", default="auto", help="强制语言：auto/zh/en")         # 多语言测试
    ap.add_argument("--tag", default="", help="结果文件名后缀，便于对照优化前后")       # 对照组标记
    a = ap.parse_args()                                                      # 解析
    if a.selfcheck: sys.exit(selfcheck())                                    # 只做自检
    if a.direct:                                                             # 直连模式下先预热
        # 不预热的话，第一题会把 bge-m3 与重排模型的加载时间算进响应时间（实测把 P95 抬到 20 秒），
        # 那是冷启动成本，不是这道题的耗时，混在一起会让性能数据失真。
        from backend import rag as R                                         # 延迟导入
        print("预热中…", R.warmup())                                          # 加载索引与本地模型
    items = load_items()                                                     # 题库
    print(f"共 {len(items)} 题，{'直连流水线' if a.direct else '调用 ' + API}\n")     # 起始提示
    rows, t_all = [], []                                                     # 逐题结果与耗时序列
    for i, it in enumerate(items, 1):                                        # 逐题
        t0 = time.time()                                                     # 计时
        try: r = ask(it["question"], a.lang, not a.direct, a.no_cache)         # 提问（缓存旁路透传到服务端）
        except Exception as e:                                               # 网络或服务异常
            r = dict(answer=f"调用失败：{type(e).__name__}: {e}", sources=[], retrieval_precision=0.0,  # 记为错答
                     elapsed_ms=round((time.time() - t0) * 1000))            # 耗时
        el = r.get("elapsed_ms") or round((time.time() - t0) * 1000)          # 端到端耗时
        j = judge(it, r.get("answer", ""), r.get("sources", []))              # 判分
        t_all.append(el)                                                     # 收集耗时
        rows.append(dict(id=it["id"], question=it["question"], answer=r.get("answer", ""),  # 逐题记录
                         correct=j["correct"], judge=j,                        # 判分
                         retrieval_precision=r.get("retrieval_precision"),     # 检索精确度
                         elapsed_ms=el, cached=r.get("cached"),                 # 耗时与缓存
                         retrieval_mode=r.get("retrieval_mode"), degraded=r.get("degraded"),  # 降级信息
                         sources=[dict(doc=s.get("doc"), page_no=s.get("page_no"),  # 来源
                                       block_type=s.get("block_type"), image_type=s.get("image_type"),  # 块与图像类型
                                       score=s.get("score")) for s in r.get("sources", [])]))  # 相关度
        mark = "✓" if j["correct"] else "✗"                                    # 对错标记
        print(f"{mark} [{i:2d}/{len(items)}] id={it['id']:<4} {el:>6} ms  "
              f"精确度 {r.get('retrieval_precision', 0):.2f}  {str(r.get('answer'))[:52]}…")  # 逐题输出
    n = len(rows); ok = sum(1 for x in rows if x["correct"])                  # 统计
    acc = ok / n if n else 0                                                 # 准确率
    srt = sorted(t_all)                                                      # 排序后算分位数
    p = lambda q: srt[min(len(srt) - 1, int(len(srt) * q))] if srt else 0      # 分位数
    precs = [x["retrieval_precision"] for x in rows if x["retrieval_precision"] is not None]  # 检索精确度序列
    summary = dict(questions=n, correct=ok, accuracy=round(acc, 4),           # 汇总结论
                   target_accuracy=0.90, accuracy_pass=acc >= 0.90,           # 工单指标：≥90%
                   avg_ms=round(statistics.mean(t_all), 1) if t_all else 0,   # 平均响应时间
                   p50_ms=p(0.5), p95_ms=p(0.95), max_ms=max(t_all or [0]),   # 分位数与最大值
                   target_ms=3000, p50_pass=p(0.5) <= 3000,                   # 工单指标：≤3 秒
                   avg_retrieval_precision=round(statistics.mean(precs), 4) if precs else 0.0,  # 平均检索精确度
                   cached_hits=sum(1 for x in rows if x["cached"]),           # 命中缓存数
                   degraded=sum(1 for x in rows if x["degraded"]))            # 降级次数
    print("\n================ 汇总 ================")                          # 分隔
    print(f"准确率      {ok}/{n} = {acc:.1%}   （目标 ≥90%：{'达标' if acc >= 0.9 else '未达标'}）")  # 准确率
    print(f"响应时间    平均 {summary['avg_ms']} ms  P50 {summary['p50_ms']} ms  "  # 耗时
          f"P95 {summary['p95_ms']} ms  最大 {summary['max_ms']} ms  （目标 P50 ≤3000："
          f"{'达标' if summary['p50_pass'] else '未达标'}）")                 # 是否达标
    print(f"检索精确度  平均 {summary['avg_retrieval_precision']:.3f}")         # 检索精确度
    print(f"缓存命中 {summary['cached_hits']} 题，降级 {summary['degraded']} 题")  # 运行状态
    RESULT_DIR.mkdir(exist_ok=True)                                          # 确保目录存在
    f = RESULT_DIR / f"test_{datetime.now():%Y%m%d_%H%M%S}{('_' + a.tag) if a.tag else ''}.json"  # 结果文件
    f.write_text(json.dumps(dict(summary=summary, rows=rows), ensure_ascii=False, indent=1), encoding="utf-8")  # 落盘
    print(f"结果已写入 {f}")                                                   # 提示


if __name__ == "__main__":                                                    # 入口
    main()                                                                    # 执行
