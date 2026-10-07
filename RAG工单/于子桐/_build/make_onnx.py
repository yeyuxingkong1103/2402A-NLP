# -*- coding: utf-8 -*-
"""
把 bge-base-zh-v1.5 导出为 ONNX 并做 int8 动态量化, 供 CPU 快速嵌入。

背景: 本机无 GPU (torch 为 CPU 版), PyTorch eager 前向约 2.6 块/秒,
      9 份金融年报(5876 块)需要 40 分钟以上, 且多任务并发时更慢。
      ONNX Runtime + int8 量化可把 CPU 推理提速数倍。

用法: python make_onnx.py
输出: D:\专高三资料\bge-base-zh-v1.5\onnx\model.onnx  /  model_int8.onnx
"""
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL_DIR = r"D:\专高三资料\bge-base-zh-v1.5"
# 注意: onnxruntime 量化会在输出目录建临时文件, 中文路径会乱码失败,
#       因此 ONNX 产物统一放在纯 ASCII 路径下
ONNX_DIR = r"D:\onnx_bge"
FP32 = os.path.join(ONNX_DIR, "model.onnx")
INT8 = os.path.join(ONNX_DIR, "model_int8.onnx")
MAX_LEN = 512


def export():
    import torch
    from transformers import AutoTokenizer, AutoModel
    os.makedirs(ONNX_DIR, exist_ok=True)
    if os.path.exists(FP32):
        print("ONNX fp32 已存在")
        return
    print("导出 ONNX ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    base = AutoModel.from_pretrained(MODEL_DIR).eval()

    class Enc(torch.nn.Module):
        """包装成只接受 3 个输入的最小前向, 供 ONNX 导出使用"""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.m(input_ids=input_ids, attention_mask=attention_mask,
                          token_type_ids=token_type_ids).last_hidden_state

    model = Enc(base).eval()
    dummy = tok(["测试文本"], return_tensors="pt", padding="max_length",
                truncation=True, max_length=32)
    inputs = (dummy["input_ids"], dummy["attention_mask"],
              dummy["token_type_ids"])
    torch.onnx.export(
        model, inputs, FP32,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "token_type_ids": {0: "batch", 1: "seq"},
            "last_hidden_state": {0: "batch", 1: "seq"},
        },
        opset_version=14, do_constant_folding=True,
        dynamo=False,          # 新版 dynamo 导出会把 batch 维写死, 用旧版导出器
    )
    tok.save_pretrained(ONNX_DIR)
    print("  已导出", FP32)


def quantize():
    from onnxruntime.quantization import quantize_dynamic, QuantType
    if os.path.exists(INT8):
        print("ONNX int8 已存在")
        return
    print("int8 动态量化 ...", flush=True)
    quantize_dynamic(FP32, INT8, weight_type=QuantType.QInt8,
                     extra_options={"MatMulConstBOnly": False})
    print("  已生成", INT8)


def main():
    export()
    quantize()
    import numpy as np
    from transformers import AutoTokenizer
    import onnxruntime as ort

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    texts = ["武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"
             "报告期内公司主营业务收入构成情况如下表所示，公司主要客户为军方。"] * 64
    texts = [t[:500] for t in texts]

    enc = tok(texts, return_tensors="np", padding="max_length",
              truncation=True, max_length=MAX_LEN)
    feeds = {k: v.astype(np.int64) for k, v in enc.items()}

    for tag, path in [("onnx-fp32", FP32), ("onnx-int8", INT8)]:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 20
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        names = [i.name for i in sess.get_inputs()]
        ins = {k: feeds[k] for k in names if k in feeds}
        sess.run(None, ins)                              # 预热
        t = time.time()
        out = sess.run(None, ins)[0]
        d = time.time() - t
        vec = out[:, 0]                                  # CLS pooling
        n = np.linalg.norm(vec, axis=1, keepdims=True)
        print(f"{tag}: 64 块 {d:.1f}s -> {64/d:.1f} 块/s   shape={vec.shape}  "
              f"norm={float(n[0]):.3f}")


if __name__ == "__main__":
    main()
