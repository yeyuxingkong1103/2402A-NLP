# AutoDL + LLaMA-Factory 微调 Qwen3.8-27B 完整方案

> 目标模型：Qwen3.8-27B（27B 稠密 + 视觉编码器，2026-08-14 开源，Apache 2.0）
> 数据规模：约 1 万条真实 SFT 数据
> 训练方法：LLaMA-Factory + QLoRA（4bit 量化）
> 推理加速：vLLM / SGLang（含内置 MTP 推测解码）
> 更新日期：2026-09-28

---

## 0. 模型基本信息

| 项目 | 规格 |
|---|---|
| 参数量 | 27B（稠密 Dense，含视觉编码器共约 27.8B） |
| 类型 | 原生视觉语言模型（图像/视频 + 文本输入，文本输出） |
| 网络结构 | 64 层 = 48 层 Gated DeltaNet 线性注意力 + 16 层 Gated Attention 全注意力（3:1 混合） |
| 词表大小 | 248,320 |
| 原生上下文 | 262,144 tokens（YaRN 可扩展到 1,000,000） |
| BF16 权重体积 | 约 52 GB |
| 架构类名 | Qwen3_5ForConditionalGeneration（config.json 里带 vision_config） |
| 特殊能力 | 内置 MTP 草稿头（推测解码不用额外下模型）；默认开启思考模式，可用 reasoning_effort 调深度 |
| 下载地址 | HuggingFace：Qwen/Qwen3.8-27B ｜ ModelScope：Qwen/Qwen3.8-27B |

其他可下载版本（部署用，非微调必须）：
- `Qwen/Qwen3.8-27B-FP8`（官方 FP8，约 29 GB）
- `unsloth/Qwen3.8-27B-NVFP4`（约 22 GB，需 Blackwell 架构显卡如 5090）
- `unsloth/Qwen3.8-27B-GGUF`（Q4_K_M 约 17 GB，24GB 卡部署用）

---

## 1. 开什么卡、几张、扩容多大

### 1.1 显卡（AutoDL 2026-09 官网价，会员 95 折）

| 方案 | 配置 | 单价 | 显存评估 | 结论 |
|---|---|---|---|---|
| ✅ 首选 | RTX PRO 6000 96GB × 1 | 6.98 元/时（会员 95 折） | QLoRA 基座约 15GB + 适配器/优化器/激活 ≈ 25GB，96GB 极其宽裕；甚至可不开量化直接 BF16 LoRA | 1 万条数据最稳、最快 |
| 备选 | A800-80GB × 1 | 5.59 元/时 | QLoRA 约 25GB，80GB 宽裕 | PRO 6000 缺货时备选 |
| 备选 | RTX 5090 × 1（32GB） | 2.78 元/时 | QLoRA 可行，注意 Blackwell 需新版 bitsandbytes | 预算敏感选它 |

**结论：开 1 张 RTX PRO 6000 96GB**，单卡即可完成 QLoRA，不需要多卡分布式。96GB 显存宽裕，可在 yaml 中适当提高 `cutoff_len` 和 `per_device_train_batch_size`。

### 1.2 存储扩容

| 盘 | 默认 | 建议 | 原因 |
|---|---|---|---|
| 系统盘 | 30 GB | 扩容到 50 GB | conda + pip 依赖 + LLaMA-Factory 源码约 20GB |
| 数据盘 | 免费 50 GB | 扩容到 100 GB | BF16 模型 52GB + 合并导出 52GB + 数据/checkpoint ≈ 80~90GB，100GB 够用且留有余量 |

数据盘费用：约 0.0066 元/日/GB（会员价），100GB ≈ 0.66 元/日。**注意：付费数据盘关机也计费**，不用时记得缩容或释放实例。

> 若同时保留原版权重 + 合并导出权重 + checkpoint，100GB 略紧；如预算允许可扩到 150GB 留缓冲。但 1 万条数据量小、checkpoint 不多，100GB 一般够。

### 1.3 费用与时间粗估

- 1 万条数据（平均每条约 800 token）1 个 epoch：RTX PRO 6000 96GB 上约 1~3 小时
- 建议跑 2~3 个 epoch：约 3~9 小时
- GPU 费用：约 20~65 元/轮（含小样本试跑）+ 存储约 0.66 元/日

---

## 2. 创建实例（完整步骤）

1. 登录 AutoDL 控制台 → 租用新实例
2. GPU 选 **RTX PRO 6000 96GB**，计费选「按量计费」
3. 页面下方扩容：**系统盘 50GB，数据盘 100GB**（下单选可扩容的主机）
4. 镜像选官方：**`PyTorch 2.8.0 / Python 3.12 / CUDA 12.8`**
   - RTX PRO 6000 是 Blackwell 架构，**CUDA 必须 ≥ 12.8** 才能被 nvidia-smi 正确识别并调用算子，CUDA 12.4 不可用
   - AutoDL 满足条件的镜像里首选此版：PyTorch 2.8.0 稳定，LLaMA-Factory / vLLM / flash-linear-attention / bitsandbytes 均已适配
   - 另有 `PyTorch 2.12 / Python 3.12 / CUDA 13.0` 也满足 CUDA 要求，但太新，第三方库（尤其 bitsandbytes、FLA 这类需编译的）可能未跟进，**不推荐**
5. 创建完成后，先 **关机 → 无卡模式开机**（0.1 元/时），把环境装好、模型下完，最后再带卡开机训练，省钱

---

## 3. 环境配置（SSH 登录后逐条执行）

```bash
# 3.1 conda 环境和包缓存指到数据盘（防止 30G 系统盘写满）
cat > ~/.condarc <<'EOF'
envs_dirs:
  - /root/autodl-tmp/conda/envs
pkgs_dirs:
  - /root/autodl-tmp/conda/pkgs
EOF

# 3.2 环境变量写进 ~/.bashrc（HF 镜像 + 缓存路径都放数据盘）
cat >> ~/.bashrc <<'EOF'
export HF_HOME=/root/autodl-tmp/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export MODELSCOPE_CACHE=/root/autodl-tmp/modelscope
export TORCH_HOME=/root/autodl-tmp/torch
EOF
source ~/.bashrc

# 3.3 开启 AutoDL 学术加速（GitHub / HF 下载提速，每次开机执行一次）
source /etc/network_turbo

# 3.4 验证基础环境（必须全过再往下走，Blackwell 必做小运算测试）
nvidia-smi                                  # 能看到 RTX PRO 6000、96GB 显存
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import torch; a=torch.randn(1024,1024,device='cuda'); print((a@a).sum().item() is not None)"  # Blackwell 算子实测，防"能看到卡但调不动"
df -h /root/autodl-tmp                      # 确认数据盘约 100G
```

> 镜像已是 PyTorch 2.8.0 / CUDA 12.8，**无需额外升级 PyTorch 或 CUDA**。RTX PRO 6000（Blackwell）必须用 CUDA ≥ 12.8，否则 `nvidia-smi` 能看到卡但 `torch.cuda` 调不动算子。
>
> bitsandbytes 必须装新版：`pip install -U bitsandbytes`（旧版不支持 Blackwell，QLoRA 4bit 量化会报错）。

---

## 4. 模型下载（具体指令）

### 4.1 方式一：ModelScope（推荐，国内直连、速度快）

```bash
pip install -U modelscope

# 下载官方 BF16 全量权重（约 52GB，微调用这个）
modelscope download --model Qwen/Qwen3.8-27B \
    --local_dir /root/autodl-tmp/models/Qwen3.8-27B
```

新版 CLI（modelscope-hub）等价写法：

```bash
pip install -U modelscope-hub
ms download Qwen/Qwen3.8-27B --local-dir /root/autodl-tmp/models/Qwen3.8-27B --max-workers 8
```

可选：官方 FP8 版本（约 29GB，部署提速用）

```bash
modelscope download --model Qwen/Qwen3.8-27B-FP8 \
    --local_dir /root/autodl-tmp/models/Qwen3.8-27B-FP8
```

### 4.2 方式二：HuggingFace 镜像（hf-mirror）

```bash
pip install -U huggingface_hub hf_transfer
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

# 新版命令 hf，旧版是 huggingface-cli download（支持断点续传，中断后重跑同一条命令即可）
hf download Qwen/Qwen3.8-27B --local-dir /root/autodl-tmp/models/Qwen3.8-27B
```

如需 NVFP4 / GGUF 量化版（部署用）：

```bash
hf download unsloth/Qwen3.8-27B-NVFP4 --local-dir /root/autodl-tmp/models/Qwen3.8-27B-NVFP4
hf download unsloth/Qwen3.8-27B-GGUF --local-dir /root/autodl-tmp/models/Qwen3.8-27B-GGUF
```

### 4.3 下载后校验（必做，防止缺文件报错）

```bash
# ① 文件数量、大小检查（safetensors 应有多个分片，总大小 ≈ 52GB）
ls -lh /root/autodl-tmp/models/Qwen3.8-27B | head -40
du -sh /root/autodl-tmp/models/Qwen3.8-27B

# ② 读取 config.json 校验模型结构
python - <<'EOF'
import json
c = json.load(open("/root/autodl-tmp/models/Qwen3.8-27B/config.json"))
print("架构:", c.get("architectures"))
print("层数:", c.get("num_hidden_layers"), "隐藏维度:", c.get("hidden_size"))
print("带视觉编码器:", "vision_config" in c)
print("最大位置:", c.get("max_position_embeddings"))
EOF
```

⚠️ **重要坑（长上下文静默截断）**：unsloth 的 NVFP4 仓库里 `tokenizer.json` 内置了 `truncation: max_length=2048`，会让超过 2048 的输入被无声截断（官方原版没有此问题）。若使用该版本，部署前检查：

```bash
python - <<'EOF'
import json
print(json.load(open("/root/autodl-tmp/models/Qwen3.8-27B-NVFP4/tokenizer.json")).get("truncation"))
# 输出必须是 None；若是 {"max_length": 2048, ...} 需要手动改成 null
EOF
```

---

## 5. 安装 LLaMA-Factory（含线性注意力加速）

```bash
source /etc/network_turbo    # git clone 提速
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
pip install -e .
pip install -r requirements/metrics.txt --no-build-isolation

# Gated DeltaNet 线性注意力加速算子（Qwen3.5/3.8 同源架构建议安装，训练/推理更快）
pip uninstall fla-core flash-linear-attention -y
pip install -U git+https://github.com/fla-org/flash-linear-attention

# 验证安装
llamafactory-cli version
```

验证框架是否已适配 Qwen3.8（架构标识为 qwen3_5）：

```bash
grep -n "qwen3_5\|qwen3_vl" src/llamafactory/data/template.py | head
```

- 有输出：记下模板名（如 `qwen3_5`），后面 `template:` 就填它
- 没有输出：执行 `git pull` 升级到最新代码；还不行就先用 WebUI 看模板下拉框，或改用 ms-swift / 官方 Transformers 脚本微调

---

## 6. 数据准备（1 万条）

> 实际选用数据集：`silk-road/ChatHaruhi-54K-Role-Playing-Dialogue`（中文小说/动漫角色扮演，54,726 条，抽前 1 万条）
> 转换脚本：项目根目录 `convert_haruhi.py`（streaming 模式，已跑通，输出 10000 条到 `data/my_data.json`）
> 数据样例：system 人设 + human/gpt 多轮对话，角色含韦小宝等

### 6.1 纯文本 SFT（ShareGPT 格式，推荐）

```json
{
  "conversations": [
    {"from": "system", "value": "你是心理咨询助手"},
    {"from": "human", "value": "我最近总是失眠怎么办？"},
    {"from": "gpt", "value": "建议先记录睡眠日记……"}
  ]
}
```

### 6.2 多模态 SFT（数据带图片时）

```json
{
  "messages": [
    {"role": "user", "content": "<image>这张图里是什么？"},
    {"role": "assistant", "content": "……"}
  ],
  "images": ["/root/autodl-tmp/data/images/0001.jpg"]
}
```

### 6.3 注册数据集（关键，不注册必报 KeyError）

把数据文件（如 `my_data.json`）放到 `LLaMA-Factory/data/` 目录下，编辑 `data/dataset_info.json`，在开头加入：

```json
{
  "my_data": {
    "file_name": "my_data.json",
    "formatting": "sharegpt",
    "columns": { "messages": "conversations" },
    "tags": {
      "role_tag": "from",
      "content_tag": "value",
      "user_tag": "human",
      "assistant_tag": "gpt"
    }
  }
}
```

注意：`columns`/`tags` 里的字段名必须和你的 json 完全一致，这是最常见的报错来源。

---

## 7. 训练配置（QLoRA，1 万条）

新建 `examples/train_lora/qwen3_8_27b_qlora.yaml`：

```yaml
### 模型
model_name_or_path: /root/autodl-tmp/models/Qwen3.8-27B
trust_remote_code: true

### 方法
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 16
lora_alpha: 32
quantization_bit: 4          # QLoRA 4bit 量化；96GB 显存宽裕，想跑 BF16 全精度 LoRA 可删掉这两行
quantization_method: bitsandbytes

### 数据
dataset: my_data
template: qwen3_5            # ← 以第 5 步 grep 结果为准
cutoff_len: 4096            # 96GB 显存宽裕可加长；纯文本最低 2048，带图训练建议 4096 起
max_samples: 10000
overwrite_cache: true
preprocessing_num_workers: 16

### 输出
output_dir: saves/qwen3.8-27b-qlora
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: true
report_to: none

### 训练
per_device_train_batch_size: 1     # 96GB 显存宽裕可调到 2，提速
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 2.0          # 1 万条数据建议跑 2~3 轮，看 loss 曲线再决定
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
```

启动训练：

```bash
# 前台（调试用）
llamafactory-cli train examples/train_lora/qwen3_8_27b_qlora.yaml 2>&1 | tee train.log

# 后台（正式跑，断连不影响）
nohup llamafactory-cli train examples/train_lora/qwen3_8_27b_qlora.yaml > train.log 2>&1 &
tail -f train.log
```

如果仓促中断需要续训：在 yaml 里加 `resume_from_checkpoint: saves/qwen3.8-27b-qlora/checkpoint-500` 再启动即可。

> 思考模式说明：Qwen3.8 默认开启思考（输出 `<thinking>...</thinking>`）。你的 SFT 数据如果包含思维链内容，需保持与模板一致的格式；不需要思维链时，训练数据的 assistant 部分直接写最终回复即可。

---

## 8. 合并导出 LoRA 权重

```bash
llamafactory-cli export \
    --model_name_or_path /root/autodl-tmp/models/Qwen3.8-27B \
    --adapter_name_or_path saves/qwen3.8-27b-qlora \
    --template qwen3_5 \
    --finetuning_type lora \
    --export_dir /root/autodl-tmp/models/Qwen3.8-27B-merged \
    --export_size 4 \
    --export_device cpu
```

---

## 9. vLLM / SGLang 加速部署

### 9.1 vLLM（推荐）

```bash
pip install -U vllm        # 架构不识别时安装 nightly 版本

vllm serve /root/autodl-tmp/models/Qwen3.8-27B-merged \
  --served-model-name qwen3.8-27b \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

说明：
- MTP 草稿头已内置在模型权重里（`model_mtp.safetensors` 注册在 index 文件中），**不需要额外下载草稿模型**，`--speculative-config` 不用写 `"model"` 字段，启动日志会显示 `Resolved architecture: Qwen3_5MTP` 即成功
- RTX PRO 6000 96GB 加载 BF16（52GB）后，32K 上下文的 KV Cache 约 7GB，96GB 单卡建议 `max-model-len` 设 32768~131072，显存仍有富余

### 9.2 SGLang

```bash
pip install "sglang[all]"

python -m sglang.launch_server \
  --model-path /root/autodl-tmp/models/Qwen3.8-27B-merged \
  --host 0.0.0.0 --port 30000 \
  --tp-size 1 \
  --context-length 32768 \
  --reasoning-parser qwen3
```

### 9.3 接口测试

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "你好，介绍一下你自己"}]
  }'
```

### 9.4 在 LLaMA-Factory 内用 vLLM 推理（可选）

```bash
llamafactory-cli api \
  --model_name_or_path /root/autodl-tmp/models/Qwen3.8-27B \
  --adapter_name_or_path saves/qwen3.8-27b-qlora \
  --template qwen3_5 \
  --infer_backend vllm \
  --port 8000
```

### 9.5 部署量化版本怎么选

| 显卡 | 建议版本 | 说明 |
|---|---|---|
| RTX PRO 6000 96GB | BF16 原版 | 52GB 权重 + KV Cache 空间充足，精度最好，首选 |
| A800-80GB | BF16 原版 | 52GB 权重装得下，精度最好 |
| H800 / 5090 | FP8（约 29GB） | 新卡原生支持 FP8 |
| RTX 5090 | NVFP4（约 22GB） | Blackwell 专用；注意第 4.3 节的 tokenizer 截断坑 |
| 24GB 卡 | GGUF Q4_K_M（约 17GB） | 走 llama.cpp 部署，约 23GB 总显存、支持 96K 上下文 |

---

## 10. 全流程验证清单（按序执行，防止报错）

1. **硬件**：`nvidia-smi` → 确认 RTX PRO 6000、显存 96GB
2. **PyTorch**：`torch.cuda.is_available()` → True
3. **模型完整性**：第 4.3 步的 config.json 校验脚本跑通
4. **数据格式**：`dataset_info.json` 注册好，用 `llamafactory-cli webui` 的 dataset preview 预览前几条，确认字段映射正确
5. **小样本试跑**：yaml 临时改 `max_samples: 200`、`max_steps: 20`，跑通（不 OOM、loss 下降、有 checkpoint）再改回来
6. **正式训练**：nohup 后台跑 + `tail -f train.log`
7. **推理验证**：先用 `llamafactory-cli chat`（huggingface 引擎）聊一句 → 再起 vLLM/SGLang → **务必用一段超过 2048 token 的长文本测试**（防 tokenizer 截断坑）

---

## 11. 常见报错速查

| 报错/现象 | 原因 | 解决 |
|---|---|---|
| CUDA out of memory | 显存不足 | 降 `cutoff_len`、`per_device_train_batch_size: 1`、确认开了 `quantization_bit: 4`、开梯度检查点 |
| flash-linear-attention 装不上 | 编译依赖问题 | 用 git 源安装（第 5 步）；暂时装不上也能跑，只是慢一些 |
| template 不存在 / 找不到 qwen3_5 | 框架版本旧 | `cd LLaMA-Factory && git pull` 升级，或查 WebUI 模板列表 |
| bitsandbytes 报错 | 版本与 CUDA/显卡不匹配 | `pip install -U bitsandbytes`（5090 等新卡必须用新版） |
| dataset KeyError | 数据字段与 dataset_info.json 不一致 | 对照第 6.3 步逐字段核对 |
| vLLM 不识别模型架构 | vLLM 版本旧 | 升级 vLLM（必要时用 nightly） |
| 长输入被忽略/前半段无响应 | tokenizer 内置截断（第 4.3 节） | 把 truncation 改为 null 或换官方原版权重 |
| 下载中断 | 网络波动 | 重跑同一条下载命令即可断点续传 |
| 训练速度极慢 | 未装 FLA / 未开 bf16 | 装 flash-linear-attention，确认 `bf16: true` |

---

## 12. 费用估算（RTX PRO 6000 96GB × 1，按量）

| 项目 | 估算 |
|---|---|
| 环境搭建 + 下模型（无卡模式） | 约 0.5~1 元（仅存储费） |
| 小样本试跑 + 正式训练（1 万条 × 2 epoch，约 3~9 小时） | 约 21~63 元 |
| 数据盘 100GB | 约 0.66 元/日（关机也计费） |
| 推理部署测试（vLLM/SGLang） | 约 3~10 元 |
| **合计** | **约 25~75 元/轮** |

省钱要点：无卡模式装环境 → 试跑通过再正式训练 → 训练完先关机、不用时缩容数据盘。
