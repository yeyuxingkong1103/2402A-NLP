# -*- coding: utf-8 -*-
"""M0 胜负手：不装 FlagEmbedding，用 rag_env 现有 transformers+torch 直接产出 bge-m3 的 dense + learned sparse。"""
import time, sys
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

MODEL_DIR = r"D:\桌面\模型\嵌入模型\bge-m3"

def log(msg): print(msg, flush=True)

def main():
    t0 = time.time()
    log("=" * 70)
    log(f"torch {torch.__version__} | cuda={torch.cuda.is_available()}")
    log(f"模型目录 {MODEL_DIR}")

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModel.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
    model.eval()
    log(f"[1] 模型加载 OK  {time.time()-t0:.1f}s | hidden={model.config.hidden_size} | layers={model.config.num_hidden_layers} | max_pos={model.config.max_position_embeddings}")

    # ---- sparse_linear.pt ----
    sp_path = Path(MODEL_DIR) / "sparse_linear.pt"
    log(f"[2] 载入 {sp_path.name} ({sp_path.stat().st_size} bytes)")
    try:
        sp = torch.load(sp_path, map_location="cpu", weights_only=False)
        log(f"    类型 = {type(sp).__name__}")
        if isinstance(sp, nn.Module):
            sd = sp.state_dict()
            log(f"    state_dict keys = {list(sd.keys())}")
            for k, v in sd.items():
                log(f"      {k}: shape={tuple(v.shape)} dtype={v.dtype}")
            sparse_linear = sp
        elif isinstance(sp, dict):
            log(f"    dict keys = {list(sp.keys())}")
            sparse_linear = None
        else:
            log(f"    其他对象，属性 = {[a for a in dir(sp) if not a.startswith('_')][:20]}")
            sparse_linear = None
    except Exception as e:
        log(f"    !! 载入失败: {type(e).__name__}: {e}")
        sparse_linear = None

    # ---- 若 sparse_linear 是 Dataset 包装（FlagEmbedding 存法），尝试取出内部 Linear ----
    if sparse_linear is None and isinstance(sp, dict):
        lin = nn.Linear(model.config.hidden_size, 1)
        lin.load_state_dict(sp)
        lin.eval()
        sparse_linear = lin
        log("    已从 state_dict 重建 nn.Linear(1024,1)")

    # ---- 前向 ----
    texts = [
        "高血压患者应当限制钠盐摄入，每日食盐量不超过5克。",
        "Hypertension patients should limit sodium intake.",
    ]
    enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    log(f"[3] tokenize OK | input_ids={tuple(enc['input_ids'].shape)}")

    with torch.no_grad():
        out = model(**enc)
    h = out.last_hidden_state                      # [B, L, H]
    mask = enc["attention_mask"].unsqueeze(-1)     # [B, L, 1]
    log(f"[4] 前向 OK | last_hidden_state={tuple(h.shape)}")

    # ---- dense: mean pooling + L2 归一化 ----
    dense = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    dense = torch.nn.functional.normalize(dense, p=2, dim=1)
    log(f"[5] dense OK | shape={tuple(dense.shape)} | L2范数={dense.norm(dim=1).tolist()}")
    log(f"    前5维 = {[round(x,5) for x in dense[0,:5].tolist()]}")

    # ---- sparse: relu(Linear(h)) * mask, 按 token id 取 max ----
    if sparse_linear is not None:
        with torch.no_grad():
            w = torch.relu(sparse_linear(h)).squeeze(-1)   # [B, L]
        w = w * enc["attention_mask"]                      # 屏蔽 padding
        log(f"[6] sparse 权重 OK | shape={tuple(w.shape)} | max={w.max().item():.4f}")

        ids = enc["input_ids"][0].tolist()
        wv  = w[0].tolist()
        agg = {}
        for tid, val in zip(ids, wv):
            if val > 0:
                agg[tid] = max(agg.get(tid, 0.0), val)
        top = sorted(agg.items(), key=lambda kv: -kv[1])[:12]
        log(f"    token 数={len(agg)}（非零稀疏维度）")
        log("    top12 稀疏词权重：")
        for tid, val in top:
            log(f"      {tok.decode([tid])!r:<16} {val:.4f}")
        nonzero = sum(1 for v in wv if v > 0)
        log(f"    有效 token 数={nonzero} / 总长={len(wv)}")
    else:
        log("[6] !! sparse_linear 不可用，无法产出稀疏向量")

    log(f"[done] 总耗时 {time.time()-t0:.1f}s")
    log("=" * 70)

if __name__ == "__main__":
    main()
