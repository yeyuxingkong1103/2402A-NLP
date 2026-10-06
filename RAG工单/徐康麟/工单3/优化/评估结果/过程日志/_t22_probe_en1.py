# -*- coding: utf-8 -*-
"""t22 英文单题探针：打印正文/占比/是否拒答/引用，并回显最近的语言闸门事件。

用法：run_py.ps1 优化/评估结果/过程日志/_t22_probe_en1.py [EN1|EN2|EN4|EN5|EN7]
"""
import sys
sys.path.insert(0, r"E:\gao6gongdan\工单3\研发")
from app.core import citation as citation_mod
from app.core.config import get_config
from app.core.generator import ENGLISH_RATIO_MIN
from app.core.language import english_ratio
from app.core.logging_conf import get_logger, setup_logging
from app.core.qa_engine import build_engine

CASES = {
    "EN1": ("What is the registered capital of Wuhan Xingtu Xinke Electronics Co., Ltd.?", None),
    "EN2": ("Who is the legal representative of Wuhan Xingtu Xinke Electronics Co., Ltd.?", None),
    "EN4": ("What is the registered capital of Wuhan Liyuan Information Technology Co., Ltd.?",
            ["招股说明书2.pdf"]),
    "EN5": ("How many shares will Wuhan Liyuan Information Technology Co., Ltd. issue?", ["招股说明书2.pdf"]),
    "EN6": ("What is the registered address of Wuhan Xingtu Xinke Electronics Co., Ltd.?", None),
    "EN7": ("What is the main business of Wuhan Xingtu Xinke Electronics Co., Ltd.?", None),
    "EN3": ("In which field has Wuhan Xingtu Xinke Electronics Co., Ltd. become an important supplier?", None),
}
NAME = (sys.argv[1] if len(sys.argv) > 1 else "EN1").upper()
REPEAT = int(sys.argv[2]) if len(sys.argv) > 2 else 1
q, files = CASES[NAME]

cfg = get_config(); setup_logging(cfg, force=True); log = get_logger("t22_probe")
print(f"backend_llm = {cfg.llm.backend} | 用例 {NAME} × {REPEAT} | 阈值 {ENGLISH_RATIO_MIN}")
with log.enter("t22_probe", {"case": NAME, "repeat": REPEAT}) as span:
    engine = build_engine(cfg=cfg, warmup=True, logger=log)
    for i in range(REPEAT):
        a = engine.ask(q, top_k=5, file_names=files)
        body = citation_mod.answer_body(a.text)
        print(f"[{NAME}#{i + 1}] unknown={a.is_unknown} metric_ratio={english_ratio(body)} "
              f"cites={[c.render() for c in a.citations]}")
        print(f"    BODY {body[:220]!r}")
    span.set_output({"case": NAME, "repeat": REPEAT})
