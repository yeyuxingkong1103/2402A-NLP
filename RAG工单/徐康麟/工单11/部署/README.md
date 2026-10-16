# 工单11 部署文档（README）

**工单编号：人工智能NLP-RAG 项目-Embedding 模型微调任务**

## 1. 目录

```
部署/
├── 脚本/
│   └── 一键流水线.ps1      数据集生成 → 微调训练 → 微调前后评估 → 生成截图 → 单元测试
├── 配置/
│   ├── 超参.json           全部超参与阈值（数据/损失/训练/评估）
│   └── 路径.json           语料、索引、底座模型、输出目录
├── 日志/                   各步骤运行日志（含安全跑包装日志入口）
└── README.md / 交付物核验.md
```

## 2. 一键运行

```powershell
cd E:\gao6gongdan\工单11
pwsh -NoProfile -File 部署/脚本/一键流水线.ps1                 # 全流程
pwsh -NoProfile -File 部署/脚本/一键流水线.ps1 -跳过训练        # 复用已有模型，只重跑评估
pwsh -NoProfile -File 部署/脚本/一键流水线.ps1 -仅数据集         # 只生成数据集
```

流水线特性：
- 每步 `Tee-Object` 落盘到 `部署/日志/`，失败即停并返回原始退出码；
- 结束时自动清理 `__pycache__` / `.pytest_cache`；
- 启动即打印可用内存，便于复现当时的资源条件。

## 3. 单独运行各步骤

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
pwsh -NoProfile -File run_py.ps1 研发/src/generate_dataset.py     # ① 约 40 秒
pwsh -NoProfile -File run_py.ps1 研发/src/train.py                # ② 约 3 分钟（117 步 = 1 epoch）
pwsh -NoProfile -File run_py.ps1 研发/src/evaluate.py             # ③ 约 2 分钟
pwsh -NoProfile -File run_py.ps1 测试/生成截图.py                  # ④ 约 10 秒
pwsh -NoProfile -File run_py.ps1 -m pytest 测试 -q                 # ⑤ 约 30 秒
```

## 4. 配置项速查（`部署/配置/超参.json`）

| 组 | 关键项 | 当前值 |
| --- | --- | --- |
| 数据集 | 随机种子 / 最大序列长度 / 留出分块比例 | 20260826 / 128 / 0.12 |
| 损失函数 | 类型 / TripletLoss.margin / 权重 | `triplet+cosine` / 0.3 / triplet 1.0 : cosine 0.5 |
| 训练 | 学习率 / 批大小 / 训练样本上限 / 最大步数 | 3e-5 / 12 / 1400 / 117 |
| 训练 | 可训练层 / 线程数 / 设备 / 时间预算秒 | 最后 2 层+池化 / 6 / cpu / 1080 |
| 评估 | topk / ndcg_k / 端到端最大分块数 | [1,5,10] / 10 / 2500 |

环境变量可覆盖路径：`WO11_CHUNKS`、`WO11_INDEX_DIR`、`WO11_QUESTIONS`、`WO11_BASE_MODEL`、`WO11_OUT_DIR`、`WO11_NUM_THREADS`。

## 5. 资源与环境纪律

1. **纯 CPU**：`torch.set_num_threads(6)`，不使用 GPU（RTX 2060 让给并行工单）。
2. **不使用进程池**：源码中无 `multiprocessing` / `ProcessPoolExecutor` / `joblib` / `Pool(` / `n_jobs=-1`。
3. **不执行工单7 的脚本**：只读其 `data/processed/chunks.jsonl` 与 `data/index/`。
4. **不安装新依赖**：本工单零新增 pip 包；`pip` 落点红线（只读的工单1）未被触碰。
5. **重活使用共享安全跑**：`E:\gao6gongdan\_共享工具\安全跑.ps1` 包裹训练/评估，带内存监察与残留清理。

## 6. 运行环境（实测）

| 项 | 值 |
| --- | --- |
| 解释器 | `E:\gao6gongdan\工单1\.venv\Scripts\python.exe`（Python 3.11.15） |
| torch | 2.11.0+cpu |
| 逻辑核 | 12（训练用 6 线程） |
| 可用内存 | 施工期间 1149 ~ 3921 MB 波动（并行工单共用机器） |

详细环境事实见 `设计/环境事实.md`。
