# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 本脚本职责：把 bge-reranker-v2-m3 导出为 ONNX，并做动态 int8 量化 —— 一次性离线构建步骤，
#             产物供 retrieval.py 的 ONNX 重排旁路加载。不参与在线问答链路，也不改任何线上默认配置。
# 用法：python scripts/export_reranker_onnx.py --out "D:/模型/bge-reranker-v2-m3-onnx"

import argparse                        # 命令行参数解析
import os                              # 改写临时目录环境变量
import shutil                          # 拷贝分词器等小文件
import tempfile                        # 让 onnxruntime 的量化中转文件也落在 ASCII 目录
import time                            # 计时
from pathlib import Path               # 路径处理

import numpy as np                     # 校验导出结果时用到
import torch                           # 导出用（CPU fp32）
from transformers import AutoModelForSequenceClassification, AutoTokenizer   # 本地模型与分词器


def build_dummy(tok, device="cpu"):                                    # 造一组固定输入用于导出与校验
    """用真实 tokenizer 造 dummy 输入：导出时形状固定 batch=2，序列维标为动态。"""
    enc = tok(["武汉兴图新科电子股份有限公司法定代表人是谁？", "程家明"],
              ["法定代表人是程家明。", "本招股意向书由武汉兴图新科电子股份有限公司出具。"],
              padding=True, truncation=True, max_length=512, return_tensors="pt")   # 一对正例一对负例
    return enc["input_ids"], enc["attention_mask"]                     # XLM-R 不使用 token_type_ids


def main() -> int:                                                     # 脚本主函数
    ap = argparse.ArgumentParser(description="导出并量化 bge-reranker-v2-m3 为 ONNX int8")   # 参数解析器
    ap.add_argument("--model", default="D:/模型/bge-reranker-v2-m3", help="原始 HF 模型目录")   # 输入模型
    ap.add_argument("--out", default="D:/模型/bge-reranker-v2-m3-onnx", help="ONNX 产物输出目录")  # 输出目录
    ap.add_argument("--workdir", default="D:/onnx_build", help="导出中转目录，**必须纯 ASCII**")   # 中转目录
    ap.add_argument("--opset", type=int, default=17, help="ONNX opset 版本")   # opset
    ap.add_argument("--skip-quant", action="store_true", help="只导出 fp32，不做 int8 量化")   # 跳过量化
    args = ap.parse_args()                                             # 解析命令行

    src, out = Path(args.model), Path(args.out)                        # 转 Path
    if not src.exists():                                               # 模型目录不存在
        print(f"[错误] 找不到模型目录：{src}")                            # 提示
        return 2                                                       # 退出码 2
    # 必须先把 ONNX 写到纯 ASCII 目录：实测 torch 的 TorchScript 导出器在中文路径下建临时文件会失败
    # （报 `Could not open file or directory: D:/模型/.../Constant_586_attr__value`）。
    # 本机用户名与项目路径都含中文，连系统临时目录也是，所以要显式指定一个 ASCII 中转目录。
    work = Path(args.workdir)                                          # 导出中转目录
    if not work.drive or any(ord(ch) > 127 for ch in str(work)):       # 含非 ASCII 字符就提前拦下
        print(f"[错误] --workdir 必须是纯 ASCII 路径，当前为：{work}")      # 提示
        return 2                                                       # 退出码 2
    work.mkdir(parents=True, exist_ok=True)                            # 建中转目录
    out.mkdir(parents=True, exist_ok=True)                             # 建输出目录
    # 同一个坑的第二处：onnxruntime 的量化会在系统临时目录下建 ort.quant.* 中转目录，
    # 而本机 %TEMP% 同样含中文（…\雨子\AppData\Local\Temp），会报
    # `FileNotFoundError: …/ort.quant.xxx/model-inferred.onnx`。把临时目录也改到 ASCII 路径。
    os.environ["TMP"] = os.environ["TEMP"] = os.environ["TMPDIR"] = str(work)   # 改环境变量
    tempfile.tempdir = str(work)                                       # 覆盖 tempfile 已缓存的值（关键）
    print(f"      临时目录已切到：{tempfile.gettempdir()}")                # 确认生效

    print(f"[1/4] 加载模型（CPU fp32）：{src}")                           # 进度
    t0 = time.time()                                                   # 计时
    tok = AutoTokenizer.from_pretrained(str(src))                      # 分词器
    model = AutoModelForSequenceClassification.from_pretrained(str(src), torch_dtype=torch.float32)  # 模型
    model.config.use_cache = False                                     # XLM-R 的 past_key_values 会干扰导出，关掉
    model.eval()                                                       # 评估模式（关 dropout）
    print(f"      完成，耗时 {time.time() - t0:.1f}s，参数量 {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    ids, mask = build_dummy(tok)                                       # dummy 输入

    fp32_path = work / "model.onnx"                                    # fp32 ONNX 路径（先落 ASCII 中转目录）
    print(f"[2/4] 导出 ONNX（opset={args.opset}，序列维动态）→ {fp32_path}")  # 进度
    t1 = time.time()                                                   # 计时
    # dynamo=False：走的仍是 TorchScript 导出器。它只依赖 onnx 包，不需要 onnxscript，
    # 因此不必升级 numpy/protobuf —— 本机 numpy 1.26.4 + protobuf 3.20.2 下实测可用。
    torch.onnx.export(
        model, (ids, mask), str(fp32_path),
        input_names=["input_ids", "attention_mask"],                   # 输入名与 ORT 侧对齐
        output_names=["logits"],                                       # 输出名同上
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}},                         # 批与序列维动态，避免写死形状
        opset_version=args.opset, do_constant_folding=True, dynamo=False,
    )
    print(f"      完成，耗时 {time.time() - t1:.1f}s，大小 {fp32_path.stat().st_size / 1e6:.0f}MB")

    print("[3/4] 校验导出结果与 PyTorch 输出一致")                          # 进度
    import onnxruntime as ort                                          # 延迟导入
    with torch.no_grad():                                              # 关梯度
        ref = model(input_ids=ids, attention_mask=mask).logits.numpy()  # PyTorch 参考输出
    sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])   # ORT 会话
    got = sess.run(None, {"input_ids": ids.numpy(), "attention_mask": mask.numpy()})[0]   # ORT 输出
    gap = float(np.abs(ref - got).max())                               # 最大绝对误差
    print(f"      fp32 最大绝对误差 = {gap:.2e}（应 <1e-3）")               # 打印校验
    if gap > 1e-3:                                                     # 误差过大说明导出有问题
        print("[错误] 导出结果与 PyTorch 不一致，停止。")                    # 提示
        return 3                                                       # 退出码 3
    del sess                                                           # 关掉会话，否则 Windows 下删不掉源文件句柄

    # 旧导出器对 2.2GB 的模型会把权重拆成上百个散文件（ONNX external data），
    # 目录又乱又难搬运。这里合并成单个 .onnx.data，产物目录只留 3~4 个文件。
    print("      合并外部权重为单个 model.onnx.data…")                     # 进度
    import onnx                                                        # 延迟导入
    merged = onnx.load(str(fp32_path))                                 # 连同外部数据一起读入内存
    tmp_onnx = work / "_merged.onnx"                                   # 先写到临时名，避免覆盖源文件
    onnx.save_model(merged, str(tmp_onnx), save_as_external_data=True,  # 以外部数据形式保存
                    all_tensors_to_one_file=True, location="model.onnx.data",   # 全部权重合并到一个文件
                    size_threshold=1024)                               # 大于 1KB 的张量才外置
    keep = {"model.onnx.data", tmp_onnx.name}                         # 要保留的：合并后的权重 + 临时主文件
    for f in work.iterdir():                                           # 清掉导出器留下的散文件
        if f.is_file() and f.name not in keep:                         # 其余全部删除
            f.unlink()                                                 # 删除（含上一轮的 model.onnx）
    tmp_onnx.rename(fp32_path)                                         # 临时名换回 model.onnx
    print(f"      model.onnx + model.onnx.data = {fp32_path.stat().st_size / 1e6:.1f}MB"
          f" + {(work / 'model.onnx.data').stat().st_size / 1e6:.0f}MB")   # 打印合并后大小

    if not args.skip_quant:                                            # 需要量化
        int8_path = work / "model.int8.onnx"                           # int8 ONNX 路径（同样先落中转目录）
        print(f"[4/4] 动态 int8 量化（权重量化，激活运行时量化）→ {int8_path}")   # 进度
        t2 = time.time()                                               # 计时
        from onnxruntime.quantization import QuantType, quantize_dynamic   # 量化接口
        quantize_dynamic(model_input=str(fp32_path), model_output=str(int8_path), weight_type=QuantType.QInt8)
        print(f"      完成，耗时 {time.time() - t2:.1f}s，大小 {int8_path.stat().st_size / 1e6:.0f}MB")

        qsess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])   # 量化模型会话
        qgot = qsess.run(None, {"input_ids": ids.numpy(), "attention_mask": mask.numpy()})[0]   # 量化输出
        # 只看 sigmoid 后的排序分：重排只用分数排序，不再关心 logits 绝对值的细微偏移
        sig = lambda x: 1.0 / (1.0 + np.exp(-x.reshape(-1)))           # 转 0~1 相关度
        print(f"      int8 vs fp32 分数：torch={np.round(sig(ref), 4).tolist()} "
              f"int8={np.round(sig(qgot), 4).tolist()}")               # 打印两组分数便于直观看差距

    names = [fp32_path.name] if args.skip_quant else [fp32_path.name, int8_path.name]   # 主文件名
    names.append("model.onnx.data")                                    # fp32 模型的外部权重文件
    for n in names:                                                    # 从中转目录拷回最终目录
        s = work / n                                                   # 源路径
        if s.exists():                                                 # 存在才拷（跳过量化时会缺）
            shutil.copy2(s, out / n)                                   # 拷贝（目标路径含中文不影响纯文件拷贝）
    artifacts = [out / n for n in names if (out / n).exists()]         # 实际产出的文件清单
    for f in ("config.json", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json", "sentencepiece.bpe.model"):   # 分词器与配置一并拷贝
        if (src / f).exists():                                         # 存在才拷
            shutil.copy2(src / f, out / f)                             # 覆盖式拷贝到产物目录
    (out / "README_onnx.txt").write_text(                              # 写一份产物说明，避免后人不知道来路
        f"由 scripts/export_reranker_onnx.py 生成\n"
        f"源模型：{src}\nopset={args.opset}，dynamo=False，动态 batch/seq\n"
        f"model.onnx = fp32；model.int8.onnx = 动态 int8 量化（权重 QInt8）\n"
        f"注意：导出必须走纯 ASCII 中转目录（默认 {work}），torch 导出器在中文路径下会失败。\n",
        encoding="utf-8")

    print("[校验] 确认 onnxruntime 能从最终（含中文的）产物目录加载")          # 进度
    for p in artifacts:                                                # 逐个试加载
        if p.suffix == ".data":                                        # 外部权重文件不是模型，不能单独加载
            print(f"      {p.name} 跳过（外部权重，随主模型一起加载）")       # 说明而非报错
            continue                                                   # 下一个
        try:                                                           # ORT 对非 ASCII 路径未必都能处理
            ort.InferenceSession(str(out / p.name), providers=["CPUExecutionProvider"])   # 建会话
            print(f"      {p.name} 可加载 ✅")                            # 通过
        except Exception as exc:                                       # 加载失败
            print(f"      {p.name} 加载失败 ❌ {type(exc).__name__}: {exc}")   # 明确报出
            print(f"      该文件需保留在 ASCII 目录：{work}")               # 给出替代方案
    print(f"完成。产物目录：{out}")                                       # 收尾
    return 0                                                           # 成功退出码


if __name__ == "__main__":            # 仅在直接运行时执行
    raise SystemExit(main())          # 用返回值作为退出码
