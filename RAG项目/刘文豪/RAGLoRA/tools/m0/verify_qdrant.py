# -*- coding: utf-8 -*-
"""M0：Qdrant 嵌入式模式 + dense/sparse named vectors 混合检索（RRF 融合）端到端验证。"""
import shutil, time
from pathlib import Path
import torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient, models

MODEL_DIR = r"D:\桌面\模型\嵌入模型\bge-m3"
STORE = Path(r"D:\桌面\RAGLoRA\tools\m0\_qdrant_probe")

def log(m): print(m, flush=True)

# ---------- 1. 编码器（复用已验证的路径） ----------
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModel.from_pretrained(MODEL_DIR, torch_dtype=torch.float32).eval()
lin = nn.Linear(1024, 1)
lin.load_state_dict(torch.load(Path(MODEL_DIR)/"sparse_linear.pt", map_location="cpu", weights_only=False))
lin.eval()

SPECIAL = set(tok.all_special_ids)

def encode(texts):
    enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        h = model(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1)
    dense = torch.nn.functional.normalize((h*mask).sum(1)/mask.sum(1).clamp(min=1e-9), p=2, dim=1)
    with torch.no_grad():
        w = torch.relu(lin(h)).squeeze(-1) * enc["attention_mask"]
    sps = []
    for b in range(len(texts)):
        agg = {}
        for tid, val in zip(enc["input_ids"][b].tolist(), w[b].tolist()):
            if val > 0 and tid not in SPECIAL:
                agg[tid] = max(agg.get(tid, 0.0), val)
        sps.append(models.SparseVector(indices=list(agg.keys()), values=list(agg.values())))
    return dense.tolist(), sps

# ---------- 2. 建库 ----------
if STORE.exists(): shutil.rmtree(STORE, ignore_errors=True)
client = QdrantClient(path=str(STORE))
COLL = "probe_kb"
client.create_collection(
    collection_name=COLL,
    vectors_config={"dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)},
    sparse_vectors_config={"sparse": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))},
)
log(f"[1] collection 建立 OK | path={STORE}")

docs = [
    ("高血压患者应限制钠盐摄入，每日食盐量不超过5克，少吃腌制食品。", {"src":"指南","law":None}),
    ("高血压的诊断标准为收缩压≥140mmHg和/或舒张压≥90mmHg。", {"src":"指南","law":None}),
    ("常用的降压药物包括钙通道阻滞剂、ACEI、ARB、利尿剂等。", {"src":"指南","law":None}),
    ("当事人一方不履行合同义务的，应当承担继续履行、赔偿损失等违约责任。", {"src":"民法典","law":"民法典"}),
    ("劳动合同应当以书面形式订立，并具备劳动合同期限等条款。", {"src":"劳动法","law":"劳动法"}),
    ("消费者因购买商品受到人身损害的，享有依法获得赔偿的权利。", {"src":"消保法","law":"消保法"}),
]
dense, sparse = encode([d[0] for d in docs])
client.upsert(COLL, points=[
    models.PointStruct(id=i, vector={"dense": dense[i], "sparse": sparse[i]}, payload=docs[i][1])
    for i in range(len(docs))
])
log(f"[2] 写入 {len(docs)} 条 (dense 1024维 + sparse 词权重) OK | count={client.count(COLL).count}")

# ---------- 3. 混合检索 ----------
def hyde(q, use_filter=None, topk=3):
    qd, qs = encode([q])
    flt = models.Filter(must=[models.FieldCondition(key="law", match=models.MatchValue(value=use_filter))]) if use_filter else None
    return client.query_points(
        COLL, prefetch=[
            models.Prefetch(query=qd[0], using="dense", limit=10),
            models.Prefetch(query=qs[0], using="sparse", limit=10),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=flt, limit=topk, with_payload=True,
    ).points

log("")
log("[3] 混合检索（dense ∥ sparse → RRF 融合）")
for q in ["高血压吃什么盐比较好？", "违约了要承担什么责任？", "腌菜能不能吃"]:
    log(f"  Q: {q}")
    for r in hyde(q):
        log(f"     {r.score:.6f}  [{r.payload['src']}] {docs[r.id][0][:34]}…")
    log("")

log("[4] 带 payload 过滤（只在法律库内检索）")
for r in hyde("赔偿", use_filter="消保法"):
    log(f"     {r.score:.6f}  [{r.payload['law']}] {docs[r.id][0][:34]}…")

log(f"\n[done] 混合检索链路端到端可用")
