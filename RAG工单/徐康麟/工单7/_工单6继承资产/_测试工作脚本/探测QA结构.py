# -*- coding: utf-8 -*-
"""临时探测脚本：确认 QAEngine.answer / retrieve_only 的真实返回字段结构。
工单：人工智能NLP-RAG-功能测试及评估
"""
import json
import sys
import time

sys.path.insert(0, r'E:\gao6gongdan\工单7\研发')

from app.core.config import get_config  # noqa: E402
from app.core.qa_engine import QAEngine  # noqa: E402

cfg = get_config()
eng = QAEngine(cfg=cfg)

t0 = time.perf_counter()
eng.warmup(with_generate=True)
print('WARMUP_MS=%.1f' % ((time.perf_counter() - t0) * 1000), flush=True)

Q = '平安银行2019年年度报告中，该行2019年实现营业收入和净利润分别是多少？'
t0 = time.perf_counter()
res = eng.answer(Q, lang='auto', top_k=10, session_id='probe', return_evidence=True)
print('ANSWER_MS=%.1f' % ((time.perf_counter() - t0) * 1000), flush=True)

print('RES_TYPE=', type(res).__name__)
attrs = [a for a in dir(res) if not a.startswith('_')]
print('RES_ATTRS=', attrs)
for a in attrs:
    try:
        v = getattr(res, a)
    except Exception as e:  # noqa: BLE001
        print('  %s -> ERR %s' % (a, e))
        continue
    if callable(v):
        continue
    s = repr(v)
    print('  %s = %s' % (a, s[:600]))

print('=== citations ===')
print(json.dumps(getattr(res, 'citations', None), ensure_ascii=False, default=str)[:1500])
print('=== cited_pages ===')
print(json.dumps(getattr(res, 'cited_pages', None), ensure_ascii=False, default=str)[:1500])
print('=== latency_ms ===')
print(json.dumps(getattr(res, 'latency_ms', None), ensure_ascii=False, default=str)[:1500])

ev = getattr(res, 'evidence', None)
print('=== evidence len ===', None if ev is None else len(ev))
if ev:
    print('EVIDENCE_KEYS=', list(ev[0].keys()))
    print(json.dumps(ev[0], ensure_ascii=False, default=str)[:2500])

ch = getattr(res, 'chunks', None)
print('=== chunks len ===', None if ch is None else len(ch))
if ch:
    print('CHUNK_TYPE=', type(ch[0]).__name__)
    if isinstance(ch[0], dict):
        print('CHUNK_KEYS=', list(ch[0].keys()))
        print(json.dumps(ch[0], ensure_ascii=False, default=str)[:2000])
    else:
        print('CHUNK_ATTRS=', [a for a in dir(ch[0]) if not a.startswith('_')])

print('=== retrieve_only ===')
try:
    hits = eng.retrieve_only('平安银行2019年营业收入', top_k=10)
    print('HITS_TYPE=', type(hits).__name__)
    print('HITS_LEN=', len(hits) if hasattr(hits, '__len__') else 'n/a')
    if hasattr(hits, '__len__') and len(hits):
        h0 = hits[0]
        if isinstance(h0, dict):
            print('HIT_KEYS=', list(h0.keys()))
            print(json.dumps(h0, ensure_ascii=False, default=str)[:2000])
        else:
            print('HIT_ATTRS=', [a for a in dir(h0) if not a.startswith('_')])
            for a in [a for a in dir(h0) if not a.startswith('_')]:
                try:
                    v = getattr(h0, a)
                except Exception:  # noqa: BLE001
                    continue
                if callable(v):
                    continue
                print('  HIT.%s = %s' % (a, repr(v)[:300]))
except Exception as e:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    print('retrieve_only ERROR:', e)

print('PROBE_DONE')
