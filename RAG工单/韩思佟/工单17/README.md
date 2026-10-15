# 工单17 · 解决API服务并发瓶颈与资源泄漏

工单编号：人工智能NLP-RAG-解决API服务并发瓶颈与资源泄漏

## 一、工单目标

JMeter 压测（20并发问答 / 10并发混合），py-spy + memory-profiler 定位瓶颈与泄漏点，给出优化补丁与 Prometheus 监控，P95≤3s/≤5s 且内存不持续上涨。

## 二、怎么跑

```bash
cd 工单17/code && jmeter -n -t ragflow_chat_bench.jmx -l result_scenarioA.jtl
python 工单17/code/parse_jtl.py result_scenarioA.jtl --tag 优化前
py-spy top --pid <pid>
```

## 三、产出物
- `工单17/问题排查分析报告.md`
- `工单17/code/ragflow_chat_bench.jmx`
- `工单17/code/parse_jtl.py`
- `工单17/产出物/`

## 五、需要截图的地方

见同目录 `截图清单.md`。