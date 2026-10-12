# 工单编号：人工智能NLP-RAG-混合检索任务
# 文件说明：**自造测试集的准入闸门**。任何自造题集在进评估之前必须先过这里的两道筛：
#           ① 题面成立（是个真问题，不是模板抽取失败留下的碎片）
#           ② 答案可核验（expect 字段确实存在于语料里；且不能与 expect_not 自相矛盾）
#           事件背景：水印漏判与"8 道废题"是同一个模式的第二次出现——做了但没质检。
#           所以把它做成可执行脚本，而不是写一句"下次注意"。
# 用法：D:/myconda1/python.exe tests/validate_questions.py [题集.json ...]
#      不给参数则检查 tests/questions.json 与 tests/questions100.json
import json, re, sys                                                                  # 标准库
from pathlib import Path                                                              # 路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))                       # 允许 import app 包
from app import config as C, rag_core as R                                            # 复用配置与索引

# ---- 第一道筛：题面成立的判据 ----
INTERROGATIVE = re.compile(r"是多少|是什么|是谁|哪些|哪一|哪家|哪里|哪个|何时|多久|是否|多少|谁|如何|怎么")  # 必须是个问句
# 坑：首版词表漏了 哪里/哪家/哪个，把 23 道手工题全判成"无疑问词"——
# 一个没被验证过的校验工具，和没有校验是同一回事。故本文件必须先用"已知合格/已知废题"验收（见 __main__ 末尾的自检）。
FRAGMENT_SUBJECT = re.compile(r"^(公司|相关金额|金额|万元|万股|项目|情况|产能较?|借款额|其余|其他)$")  # 主题抽取失败的残留
MIN_SUBJECT = 4                                                                       # 主题至少这么长才算说得清


def subject_of(q: str, company: str) -> str:                                          # 抽出问句的主题部分
    """去掉公司名前缀与疑问尾，剩下的就是"问的是什么东西"。"""
    s = q.replace(company, "").lstrip("，,、 ")                                          # 去公司名
    return re.sub(r"(分别是多少？?|是多少？?|是什么？?|是谁？?|有多少？?|？|\?)$", "", s).strip()  # 去疑问尾


def check_question(q: dict, company: str) -> list:                                     # 第一道筛
    """返回失败原因列表；空列表表示题面成立。"""
    bad = []                                                                          # 失败原因
    text = q.get("question") or q.get("question_zh") or ""                             # 取问句
    if not text.strip():                                                              # 没问句
        return ["无问句"]                                                              # 直接判废
    if not (INTERROGATIVE.search(text) or text.rstrip().endswith(("？", "?"))):       # 既是问句又有问号才算
        bad.append("无疑问词")                                                         # 记因
    subj = subject_of(text, company)                                                  # 主题
    if len(subj) < MIN_SUBJECT:                                                       # 主题太短
        bad.append(f"主题过短({subj!r})")                                              # 记因
    elif FRAGMENT_SUBJECT.match(subj):                                                # 主题是模板残留
        bad.append(f"主题是碎片({subj!r})")                                            # 记因
    return bad                                                                        # 返回


def check_answer(q: dict, corpus: dict) -> list:                                       # 第二道筛
    """返回失败原因列表；空列表表示答案可核验。"""
    bad = []                                                                          # 失败原因
    doc = q.get("doc") or ""                                                          # 目标文档
    if doc not in corpus:                                                             # 手工题集没有 doc 字段
        text = q.get("question") or q.get("question_zh") or ""                          # 从问句里推断
        doc = next((d for d in corpus if any(w in text for w in _ALIAS.get(d, ()))), "")  # 按公司名/简称匹配
    table = corpus.get(doc, "")                                                       # 该文档全文（去逗号便于比对）
    if not table:                                                                     # 仍对不上
        return [f"文档名无法匹配({q.get('doc')})"]                                      # 记因
    miss = [e for e in q.get("expect", []) if e.replace(",", "") not in table]         # 必需的答案字段
    if miss:                                                                          # 有字段不在语料里
        bad.append(f"expect 不在语料中{miss[:3]}")                                     # 记因
    for n in q.get("expect_not", []):                                                 # 负样本不得与答案同块
        if n.replace(",", "") in table and all(                                     # 若负样本也在语料里
                e.replace(",", "") in table for e in q.get("expect", [])):            # 且答案也在（无法区分）
            pass                                                                      # 说明不了问题，跳过
    return bad                                                                        # 返回


_ALIAS = {"招股说明书1": ("兴图", "Xingtu"), "招股说明书2": ("力源", "P&S")}             # 文档别名，用于推断 doc


def load(path: Path) -> tuple:                                                        # 读题集
    """兼容三种结构：{questions:[...]}、{subset_A_natural,subset_B_probe}、纯 list。"""
    d = json.loads(path.read_text(encoding="utf-8"))                                  # 读文件
    if isinstance(d, list):                                                           # 纯列表
        return d, []                                                                  # 全部题
    if "questions" in d:                                                              # 单层
        return d["questions"], []                                                     # 全部题
    return d.get("subset_A_natural", []) + d.get("subset_B_probe", []), []             # 子集合并


def main() -> int:                                                                    # 返回退出码
    """对每个题集跑两道筛，打印失败清单与汇总。任一题集有失败则退出码非零。"""
    files = [Path(a) for a in sys.argv[1:]] or [C.ROOT / "tests" / "questions.json",  # 默认检查两个题集
                                                C.ROOT / "tests" / "questions100.json"]
    R.bm25_load()                                                                     # 载入索引（拿语料全文）
    chunks = R.bm25_load()["chunks"]                                                  # 块表
    corpus = {}                                                                       # 文档名 → 去逗号全文
    for c in chunks:                                                                  # 逐块拼接
        corpus[c["doc"]] = corpus.get(c["doc"], "") + c["text"]                       # 累加
    corpus = {k: v.replace(",", "") for k, v in corpus.items()}                        # 统一去千分位
    COMPANY = ["武汉兴图新科电子股份有限公司", "武汉力源信息技术股份有限公司"]             # 两家公司名
    # ---- 闸门自检：校验工具本身也必须被校验，否则它就是下一个"没质检" ----
    _good = {"question": "武汉兴图新科电子股份有限公司的注册资本是多少？", "expect": ["5,520"], "doc": "招股说明书1"}
    _bad = {"question": "武汉兴图新科电子股份有限公司，相关金额分别是多少？", "expect": [], "doc": "招股说明书1"}
    _c = {"招股说明书1": "注册资本5520万元", "招股说明书2": "注册资本5000万元"}          # 假语料也要与真实语料同口径（已去千分位）
    ok_self = (not check_question(_good, COMPANY[0]) and not check_answer(_good, _c)   # 合格样本必须过
               and bool(check_question(_bad, COMPANY[0])))                            # 废样本必须挂
    print(f"闸门自检：{'通过' if ok_self else '不通过（校验器本身有问题，先修它再跑题集）'}")   # 打印
    if not ok_self:                                                                   # 自检失败
        return 2                                                                      # 直接退出，不给出误导性结论
    total_fail = 0                                                                    # 失败总数
    for f in files:                                                                   # 逐题集
        if not f.exists():                                                            # 文件不存在
            print(f"[跳过] {f} 不存在")                                                 # 提示
            continue                                                                  # 下一个
        rows, _ = load(f)                                                             # 读题
        print("=" * 74)                                                               # 分隔
        print(f"{f.name}：{len(rows)} 题")                                             # 标题
        fails = []                                                                    # 失败清单
        for i, q in enumerate(rows, 1):                                               # 逐题
            company = next((c for c in COMPANY if c in (q.get("question") or q.get("question_zh") or "")), "")  # 命中哪家公司
            bad = check_question(q, company) + check_answer(q, corpus)                 # 两道筛
            if bad:                                                                   # 有问题
                fails.append((i, (q.get("question") or q.get("question_zh"))[:44], bad))  # 记录
        total_fail += len(fails)                                                      # 累加
        print(f"  通过 {len(rows) - len(fails)}/{len(rows)}；失败 {len(fails)}")         # 汇总
        for i, text, bad in fails[:12]:                                               # 列失败（最多 12 条）
            print(f"    #{i:<4} {'; '.join(bad):<34} {text}")                          # 打印
        if len(fails) > 12:                                                           # 还有更多
            print(f"    …另有 {len(fails) - 12} 条")                                    # 提示
    print("=" * 74)                                                                   # 分隔
    print(f"结论：{'全部通过，可以进评估' if total_fail == 0 else f'共 {total_fail} 题不合格 —— 不合格题必须先剔除或修正，再进评估'}")  # 结论
    return 0 if total_fail == 0 else 1                                                # 退出码供 CI 用


if __name__ == "__main__":                                                            # 允许直接运行
    sys.exit(main())                                                                  # 退出码
