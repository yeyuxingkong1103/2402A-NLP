# -*- coding: utf-8 -*-
"""t22 补丁 7 后英文逐题复测（用 language.english_ratio 正式口径，排除逐字原文片段）。

能力说明：本脚本为 t22「英文作答语言适配」的定稿复测脚本，与
`优化/评估结果/过程日志/_t22_patch*.py` 同属可审计过程产物（工单3）。
"""
import sys
sys.path.insert(0, r"E:\gao6gongdan\工单3\研发")
from app.core import citation as citation_mod
from app.core import text_utils
from app.core.config import get_config
from app.core.generator import ENGLISH_RATIO_MIN
from app.core.language import english_ratio
from app.core.logging_conf import get_logger, setup_logging
from app.core.qa_engine import build_engine

cfg = get_config(); setup_logging(cfg, force=True); log = get_logger("t22_after7")

EN = [
    ("EN1 registered capital 兴图", "What is the registered capital of Wuhan Xingtu Xinke Electronics Co., Ltd.?", None, "field"),
    ("EN2 legal representative 兴图", "Who is the legal representative of Wuhan Xingtu Xinke Electronics Co., Ltd.?", None, "field"),
    ("EN3 领域 兴图", "In which field has Wuhan Xingtu Xinke Electronics Co., Ltd. become an important supplier?", None, "narrative"),
    ("EN4 registered capital 力源", "What is the registered capital of Wuhan Liyuan Information Technology Co., Ltd.?", ["招股说明书2.pdf"], "field"),
    ("EN5 shares 力源", "How many shares will Wuhan Liyuan Information Technology Co., Ltd. issue?", ["招股说明书2.pdf"], "field"),
    ("EN6 registered address 兴图", "What is the registered address of Wuhan Xingtu Xinke Electronics Co., Ltd.?", None, "field"),
    ("EN7 main business 兴图", "What is the main business of Wuhan Xingtu Xinke Electronics Co., Ltd.?", None, "narrative"),
]
NEG = [
    ("N-Tesla", "What is the registered capital of Tesla, Inc.?", None, "negative"),
    ("N-2099", "What are the plans of Wuhan Xingtu Xinke Electronics Co., Ltd. for the 2099 lunar base project?", None, "negative"),
]
ECHO = [
    ("ECHO-en1", "What is the registered capital of Wuhan Xingtu Xinke Electronics Co., Ltd.?", None, "echo"),
    ("ECHO-en5", "How many shares will Wuhan Liyuan Information Technology Co., Ltd. issue?", ["招股说明书2.pdf"], "echo"),
]
ZH = [
    ("ZH1 中文对照 注册资本", "武汉兴图新科电子股份有限公司的注册资本是多少？", None, "zh-control"),
]


def dump(name, q, files, kind):
    a = engine.ask(q, top_k=5, file_names=files)
    body = citation_mod.answer_body(a.text)
    sq_body = text_utils.squash_text(body)
    sq_q = text_utils.squash_text(q)
    echo = bool(sq_body) and (sq_body in sq_q) and len(sq_body) >= 0.6 * len(sq_q)
    cites = [c.render() for c in a.citations]
    print(f"  {name} [{kind}] metric_ratio={english_ratio(body)} raw_ratio={english_ratio(body, keep_verbatim=True)} "
          f"unknown={a.is_unknown} echoes_q={echo} cites={cites}")
    print(f"     BODY {body[:160]}")
    return {"name": name, "kind": kind, "ratio": english_ratio(body),
            "raw": english_ratio(body, keep_verbatim=True), "unknown": a.is_unknown,
            "echo": echo, "body": body, "cites": cites}


with log.enter("t22_after7", {"min_ratio": ENGLISH_RATIO_MIN}) as span:
    engine = build_engine(cfg=cfg, warmup=True, logger=log)
    print("=== 英文逐题（补丁 7 后；阈值 %.2f）===" % ENGLISH_RATIO_MIN)
    rows = [dump(*c) for c in EN]
    print("=== 英文负例（须拒答且零引用）===")
    negs = [dump(*c) for c in NEG]
    print("=== 回声专项（正文不得是问句回声）===")
    ech = [dump(*c) for c in ECHO]
    print("=== 中文对照（不得退化）===")
    zh = [dump(*c) for c in ZH]
    ok = [r for r in rows if not r["unknown"] and r["ratio"] >= ENGLISH_RATIO_MIN]
    deg = [r for r in rows if not r["unknown"] and r["ratio"] < ENGLISH_RATIO_MIN]
    print(f"\n汇总：达标 {len(ok)} / 降级 {len(deg)} / 英文正例被拒答 "
          f"{len([r for r in rows if r['unknown']])}")
    print("达标:", [r["name"] for r in ok])
    print("降级:", [r["name"] for r in deg])
    print("负例拒答:", [(r["name"], r["unknown"], len(r["cites"])) for r in negs])
    print("回声命中:", [r["name"] for r in ech if r["echo"]])
    print("中文对照:", [(r["name"], r["unknown"], r["ratio"]) for r in zh])
    span.set_output({"ok": len(ok), "degraded": len(deg)})
