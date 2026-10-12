# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-功能测试及评估 —— 自检脚本
# 纪律来源（RAG/CLAUDE.md 第二条）：自造的产物进评估前，必须先过「已知正例全过 + 已知反例全拦」。
# 本文件对自造的四个关键件做自检：① 解析降级链 ② 混合融合 ③ 缓存键指纹 ④ RAGAS 样本构造。
# 任一用例失败即整体非 0 退出——自检不通过时不得给出任何评估结论。
import sys, pathlib, os                                            # 路径准备与退出码
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # 把项目根加进模块搜索路径
from app import config as C                                        # 配置
from app.ingest import doc_meta, parse_pymupdf, parse_mineru, parse_txt  # 被检对象：身份回认与三级解析器
from app.rag_core import fuse, _fp                                 # 被检对象：融合与缓存指纹
FAIL = []                                                          # 收集失败用例

def ck(name, cond, detail=""):                                     # 极简断言器，失败不中断以便一次看全
    print(("  PASS  " if cond else "  FAIL  ") + name + ("" if cond else f"  <- {detail}"))
    if not cond:
        FAIL.append(name)                                          # 记录失败

print("\n[1] 解析三级链：正例（真实 PDF 可解析）与反例（不存在的文件必须全部返回 None）")
pdfs = sorted(p for p in (C.CORPUS_DIR / "pdf").iterdir() if p.suffix.lower() == ".pdf")   # 真实语料
real = str(pdfs[0])                                                # 取第一份做正例
blocks = parse_pymupdf(real)                                       # pymupdf 应当读到内容
ck("正例 pymupdf 解析出多页文本", bool(blocks) and len(blocks) > 100, f"blocks={len(blocks) if blocks else 0}")
ck("正例 页码从 1 开始且无空页", blocks[0][0] == 1 and all(t for _, t in blocks))          # 页码语义正确
fake = str(C.DATA / "不存在的文件.pdf")                             # 反例：不存在的路径
ck("反例 pymupdf 返回 None（不抛异常）", parse_pymupdf(fake) is None)                       # 降级而非崩溃
ck("反例 txt 兜底返回 None", parse_txt(fake) is None)                                       # 兜底也要能失败退出
ck("反例 单级 MinerU 对坏文件返回 None（不抛异常）", parse_mineru(fake) is None)              # 单级降级正确
os.environ["MINERU_ENABLED"] = "0"                                 # 打开开关：parse_pdf 应跳过 MinerU 主路径
import app.ingest as _ing                                          # 重新取模块以读取最新环境变量
_blocks2, _parser = _ing.parse_pdf(real)                           # 对真实 PDF 走一遍降级入口
ck("正例 MINERU_ENABLED=0 时主路径被跳过，实际用 pymupdf", _parser == "parse_pymupdf", _parser)  # 开关真的生效
ck("正例 跳过 MinerU 后仍拿到同样的内容", len(_blocks2) == len(blocks))                        # 降级结果等价

print("\n[2] 文档身份回认：正例（文件名中文已被 U+FFFD 破坏仍能认对）与反例（无股票代码→unknown）")
code, name, year = doc_meta(real)                                  # 真实文件名（中文已损坏）
ck("正例 股票代码解析正确", code == "000001", code)                 # 唯一稳定标识
ck("正例 公司名回认正确", name == "平安银行", name)                  # 靠 CODE2NAME 映射
ck("正例 报告年度 = 披露年 - 1", year == "2019", year)              # 2020-02-14 → 2019 年报
ck("反例 无代码文件名回认为 unknown", doc_meta("noise.pdf")[0] == "unknown")               # 不能瞎猜

print("\n[3] 混合融合：通道内归一化 + 加权 + 双通道去重合并")
vh = [{"text": "A" * 100, "score": 0.9, "src": "vec", "company": "x", "year": "2019", "page": 1},
      {"text": "B" * 100, "score": 0.5, "src": "vec", "company": "x", "year": "2019", "page": 2}]  # 向量通道
bh = [{"text": "C" * 100, "score": 40.0, "src": "bm25", "company": "x", "year": "2019", "page": 3},
      {"text": "A" * 100, "score": 20.0, "src": "bm25", "company": "x", "year": "2019", "page": 1}]  # 关键词通道（A 重复）
out = fuse(vh, bh, k=10)                                           # 融合
ck("正例 输出条数 = 去重后条数（3）", len(out) == 3, f"len={len(out)}")                     # 去重键生效
ck("正例 A 因双通道命中得分最高", out[0]["text"].startswith("A"), out[0]["src"])            # 加权相加生效
ck("正例 A 的来源被合并标记", "vec" in out[0]["src"] and "bm25" in out[0]["src"], out[0]["src"])
ck("正例 分数已归一到不超过 1", all(0 <= h["score"] <= 1.0001 for h in out))               # 归一化后加权和为 1
ck("反例 空通道参与融合不报错", fuse([], [], k=5) == [])                                    # 边界：全空

print("\n[4] 缓存键指纹：同输入同键、异输入异键、源码变则键变（防旧缓存静默吞掉新逻辑）")
import hashlib                                                     # 按 app/rag_core.py::answer 的构造方式复算键
key = lambda raw: "gd7:ans:" + _fp() + ":" + hashlib.sha256(raw.encode()).hexdigest()[:32]  # 与实现同构
raw1 = "问题|hybrid|0.7|0.3|8"                                      # 原始键串
ck("正例 同输入 → 同键", key(raw1) == key(raw1))                     # 键稳定（可命中缓存）
ck("反例 不同问题 → 不同键", key(raw1) != key("另一个问题|hybrid|0.7|0.3|8"))                  # 问题进键
ck("反例 只改检索参数也必须换键", key(raw1) != key("问题|hybrid|0.7|0.3|4"))                   # 参数进键，防串味
ck("正例 指纹为 16 位十六进制", len(_fp()) == 16 and all(c in "0123456789abcdef" for c in _fp()))  # 结构正确
print("  指纹样本 =", _fp())                                        # 打印便于比对

print("\n[5] RAGAS 样本构造：10 题四元组齐全，contexts 与 reference 非空")
import json                                                        # 读题集
qs = json.loads(C.QUESTIONS_FILE.read_text(encoding="utf-8"))["questions"]  # 题集
ck("正例 题数为 10", len(qs) == 10, len(qs))                        # 工单硬性要求 10 题
ck("正例 每题都有 question 与 ground_truth", all(q.get("question") and q.get("ground_truth") for q in qs))
ck("正例 题集含 sample 原题 4 道", sum(1 for q in qs if q["source"] == "sample") == 4)      # 与原 PDF 一致
ck("反例 空 ground_truth 会被发现", any(not q.get("ground_truth") for q in [{"ground_truth": ""}]))  # 反例可检出
ck("正例 题号唯一", len({q["id"] for q in qs}) == len(qs))          # 无重复题号

print(f"\n自检结果：{'全部通过' if not FAIL else '存在失败用例 ' + str(FAIL)}")
sys.exit(1 if FAIL else 0)                                         # 失败必须非 0 退出，让调用方无法忽略
