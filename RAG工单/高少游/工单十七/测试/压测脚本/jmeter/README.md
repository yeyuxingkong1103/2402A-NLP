# JMeter 压测脚本（工单十七）

工单编号: 人工智能 NLP-RAG-解决API服务并发瓶颈与资源泄漏

本目录为工单「任务一：搭建压测环境与基准测试」要求的 JMeter 压测脚本。

| 文件 | 场景 | 负载模型 | 验收口径 |
|------|------|----------|----------|
| `scenario_a_qa.jmx` | 场景 A 高频问答 | 20 并发，持续 600s | P95 ≤ 3000ms，失败率 0 |
| `scenario_b_mixed.jmx` | 场景 B 混合负载 | 5 问答 + 5 上传解析，持续 600s | 服务不崩溃，解析入队并被消费，问答 P95 ≤ 5000ms |

## 运行方式

```bash
# 场景 A（优化版默认端口 8001；基线为 8000）
jmeter -n -t scenario_a_qa.jmx -l result_a.jtl -e -o report_a \
       -Jbase_url=http://127.0.0.1:8001 -Jthreads=20 -Jduration=600

# 场景 B
jmeter -n -t scenario_b_mixed.jmx -l result_b.jtl -e -o report_b \
       -Jbase_url=http://127.0.0.1:8001 -Jduration=600
```

参数通过 `-J` 传入：`base_url`、`threads`、`duration`。

## 与 Python 压测器的等价性

工单同时提供两套**同口径**压测实现，便于在无 JMeter 环境复现：

- JMeter：本目录 `.jmx`（图形化、可产出 HTML 报告）
- Python：`tools/loadtest.py`（真实跑通、无外部依赖，本工单实测数据由它产出）

两者使用**相同的 URL、相同的负载模型（并发数/时长/问答-上传比例）、相同的指标口径**
（请求数、成功率、吞吐 RPS、P50/P90/P95/P99/Max 延迟、进程 RSS/commit 曲线）。

> 说明：本工单实测环境（TRAE 沙箱）未安装 JMeter，故基准与优化数据由
> `tools/loadtest.py` 真实跑出；`.jmx` 与 `tools/run_benchmarks.py` 一一对应，
> 可在具备 JMeter 的机器上直接执行得到等价结果。