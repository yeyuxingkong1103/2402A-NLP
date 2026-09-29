# 评测集说明

用于 RAGAS 评测与优化前后对比，共 **56 题**：法律 26、心理 15、金融 15。

> 初始 50 题（法律 20）全部是口语化提问。后续发现这类题目**不会触发
> 「按法条号精确命中」的元数据路**，该路径因此完全没被评测覆盖，
> 遂补充 6 道条号题（`law-21`~`law-26`，答案取自知识库真实法条原文）。
> 已有的 50 题历史判分结果保持不变，故 `scores_*.json` 中的样本数仍为 50。

## 设计原则

1. **提问用自然口语**，不照抄知识库原文——否则题目本身就在索引里，
   检索必然命中，指标会虚高，测不出真实能力。
2. **标准答案可由知识库内容支撑**，写成要点式而非大段摘抄，
   便于 RAGAS 判断语义覆盖而非字面重合。
3. **覆盖各角色的主要知识分支**，避免只考命中率最高的那一类。
4. 每题标注 `category`，便于分维度看优化收益落在哪里。

## 字段

| 字段 | 说明 |
|---|---|
| `id` | 题号，如 `law-01` |
| `role_key` | 角色标识 |
| `category` | 知识分支，用于分维度分析 |
| `question` | 用户提问（自然口语） |
| `ground_truth` | 参考答案要点 |

系统实际产出的 `answer` 与 `contexts` 由评测脚本运行时采集，不预先写死——
这样同一份评测集可以直接跑在优化前后的两个版本上做对照。

## 用法

三步：**采集 → 判分 → 对比**。采集与对比用项目 venv，判分必须用评测专用 venv
（ragas 依赖 langchain-core 0.3，与项目环境的 1.x 不兼容）。

```bash
# 1) 采集：对评测集跑一遍完整问答，存下回答与检索上下文
.venv/bin/python -m evals.collect --tag baseline

# 2) 判分：用评测专用 venv，由大模型按四个指标打分
evals/.venv/bin/python evals/score.py --tag baseline
evals/.venv/bin/python evals/score.py --tag baseline --limit 5   # 先小样本试跑

# 3) 对比：读 scores_{tag}.json，输出增益并写出 comparison.json
.venv/bin/python -m evals.compare --base ablated --opt optimized
```

对比脚本的参数是 `--base` / `--opt`（不是位置参数），默认
`baseline → optimized`。当前仓库里 `baseline` 是**已作废**的一轮
（两次采集之间知识库重建过，不可比），正式结论用的是 `ablated → optimized`。
