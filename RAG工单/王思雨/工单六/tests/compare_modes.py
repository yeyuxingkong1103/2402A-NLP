# 工单编号：人工智能NLP-RAG-混合检索任务
# 文件说明：检索层精度对比。对同一套题分别用「纯向量」「纯全文」「混合」检索，统计 top-k 片段里
#           是否包含标准答案的关键字段。全程不调用大模型，只反映**检索层**能力，不掺生成层的偶然性。
#
#           指标拆成两个，因为混在一起会互相掩盖：
#             recall  = 必需字段是否全部被召回（衡量召回能力）
#             clean   = 同时要求负样本字段一个都不出现（衡量"有没有把另一家公司的内容也捞进来"）
#           注意 clean 会随 k 增大而天然变差——上下文越长，越容易混进无关内容。
#           这正是"不能只用一个笼统命中率下结论"的原因。
import json, re, sys                                                               # 标准库
from pathlib import Path                                                           # 路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))                    # 允许 import app 包
from app import config as C                                                        # 配置
from app import rag_core as R                                                      # 检索核心

FLAT = re.compile(r"[,\s，]")                                                      # 千分位逗号与空白


def flat(s: str) -> str:                                                           # 归一化
    """去掉千分位与空白，使 6,464.51 与 6464.51 视为同一结果。"""
    return FLAT.sub("", s or "")                                                   # 返回


def judge(ctx: str, q: dict) -> tuple:                                             # 命中判定
    """返回 (recall, clean)：必需字段全含 = recall；再要求负样本不出现 = clean。"""
    c = flat(ctx)                                                                  # 归一化上下文
    must = (all(flat(e) in c for e in q.get("expect", []))                         # 必需字段全含
            and all(any(flat(x) in c for x in g) for g in q.get("expect_any", [])))  # 同义组至少一项
    clean = must and not any(flat(n) in c for n in q.get("expect_not", []))        # 且不含负样本
    return must, clean                                                             # 返回两项


def main() -> int:                                                                 # 返回退出码
    """遍历 模式 × 融合算法 × 是否重排 × top_k，打印两个指标。"""
    data = json.loads(C.EVAL_FILE.read_text(encoding="utf-8"))                     # 读题集
    rows = data.get("questions", data) if isinstance(data, dict) else data         # 兼容两种结构
    R.bm25_load(); R.embed(["预热"]); R.rerank_model()                             # 预热，避免首查询承担加载开销
    combos = []                                                                    # 待测组合
    for mode in ("vector", "bm25", "hybrid"):                                      # 三种检索模式
        for fu in (["weighted", "rrf", "vote"] if mode == "hybrid" else ["-"]):    # 混合模式才谈融合算法
            for rk, rtag in ((["bge"], "bge"), ([], "无")):                         # 重排开 / 关
                combos.append((mode, fu, rk, rtag))                                # 收集
    print("=" * 92)                                                                # 分隔
    print(f"阶段三 · 检索层精度对比（题数 {len(rows)}）")                              # 标题
    print("  recall = 必需字段全部召回；clean = 同时不含负样本（负样本判据会随上下文变长而变差）")  # 口径
    print("=" * 92)                                                                # 分隔
    print(f"{'模式':<9}{'融合':<10}{'重排':<6}{'k=4 recall':<13}{'k=4 clean':<12}{'k=8 recall':<13}{'k=8 clean':<12}")  # 表头
    out = {}                                                                       # 汇总
    for mode, fu, rk, rtag in combos:                                              # 逐组合
        cells = []                                                                 # 该组合的四个数
        for k in (4, 8):                                                           # 两个 top_k
            r_ok = c_ok = 0                                                        # 计数
            for q in rows:                                                         # 逐题
                fw = 0.7 if fu == "-" else C.VEC_WEIGHT                            # 非混合模式下权重无意义
                bw = 0.3 if fu == "-" else C.BM25_WEIGHT                           # 同上
                cands, _ = R.retrieve(q["question"], mode, k, fw, bw,             # 检索
                                      "weighted" if fu == "-" else fu, rk)         # 融合与重排
                m, c = judge(" ".join(x.get("text") or "" for x in cands), q)       # 判定
                r_ok += int(m); c_ok += int(c)                                     # 累加
            cells += [round(r_ok / len(rows), 4), round(c_ok / len(rows), 4)]       # 记录
        key = f"{mode}/{fu}/{rtag}"                                                # 组合名
        out[key] = {"k4": {"recall": cells[0], "clean": cells[1]},                 # 结果
                    "k8": {"recall": cells[2], "clean": cells[3]}}                 # 同上
        print(f"{mode:<9}{fu:<10}{rtag:<6}{cells[0]:<13}{cells[1]:<12}{cells[2]:<13}{cells[3]:<12}")  # 打印
    print("-" * 92)                                                                # 分隔
    print("原始数据:", json.dumps(out, ensure_ascii=False))                          # 便于粘贴进报告
    return 0                                                                       # 退出


if __name__ == "__main__":                                                         # 允许直接运行
    sys.exit(main())                                                               # 退出码
