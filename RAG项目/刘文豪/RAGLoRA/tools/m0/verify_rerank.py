# -*- coding: utf-8 -*-
"""M0：bge-reranker-v2-m3 上 GPU(fp16) 打分，测显存与耗时，确认 8G 红线可行。"""
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = r"D:\桌面\模型\精排模型\bge-reranker-v2-m3"

def log(m): print(m, flush=True)

pairs = [
    ("高血压患者能吃腌菜吗？", "高血压患者应限制钠盐摄入，每日食盐量不超过5克，少吃腌制食品。"),
    ("高血压患者能吃腌菜吗？", "劳动合同应当以书面形式订立，并具备以下条款。"),
    ("民法典规定违约责任有哪些？", "当事人一方不履行合同义务或者履行合同义务不符合约定的，应当承担继续履行、采取补救措施或者赔偿损失等违约责任。"),
    ("民法典规定违约责任有哪些？", "高血压的诊断标准为收缩压≥140mmHg和/或舒张压≥90mmHg。"),
]

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR, torch_dtype=torch.float16
).cuda().eval()
log(f"[1] 加载到 GPU 完成 {time.time()-t0:.1f}s | dtype=fp16")
log(f"    GPU = {torch.cuda.get_device_name(0)}")
log(f"    模型参数显存 = {torch.cuda.memory_allocated()/1024**3:.2f} GB")

queries = [p[0] for p in pairs]
docs    = [p[1] for p in pairs]
enc = tok(queries, docs, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")

with torch.no_grad():
    t1 = time.time()
    logits = model(**enc, return_dict=True).logits.view(-1).float()
    torch.cuda.synchronize()
    dt = time.time() - t1

log(f"[2] 打分完成 {dt*1000:.0f}ms / {len(pairs)} 对")
log(f"    峰值显存 = {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
log("")
log("[3] 相关性打分（分数越高越相关）：")
for (q, d), s in zip(pairs, logits.tolist()):
    tag = "相关  " if s > 0 else "不相关"
    log(f"    {s:+.4f}  {tag}  Q={q[:14]:<16} D={d[:28]}")

# 排序正确性检查
ok1 = logits[0] > logits[1]
ok2 = logits[2] > logits[3]
log("")
log(f"[4] 排序正确性：医疗对 {ok1} | 法律对 {ok2} -> {'通过' if ok1 and ok2 else '失败'}")
log(f"[done] 总耗时 {time.time()-t0:.1f}s")
