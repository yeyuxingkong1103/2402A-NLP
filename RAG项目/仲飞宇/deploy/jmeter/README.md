# Jmeter 压测

需求里的「性能测试/压力测试：Jmeter、QPS」。脚本 `rag-roleplay-load.jmx` 是一个可导入的
最小压测计划：50 并发、10s 预热、循环 10 次，对 `POST /chat` 打请求并断言返回含 `answer`。

## 用法

1. 确保应用已启动：`bash scripts/start_all.sh`，确认 `curl -s localhost:8000/health` 返回 ok。
2. 下载 [Apache JMeter](https://jmeter.apache.org/)（5.x），解压后运行：
   - GUI：`jmeter -t deploy/jmeter/rag-roleplay-load.jmx`
   - 命令行（推荐，无界面出报告）：
     ```bash
     jmeter -n -t deploy/jmeter/rag-roleplay-load.jmx \
            -l deploy/jmeter/result.jtl -e -o deploy/jmeter/report
     ```
3. 看结果：GUI 里点「聚合报告」；命令行生成的 HTML 报告在 `report/index.html`。

## 关注指标

- **QPS / 吞吐量**：聚合报告里的 Throughput（请求数/秒）。本系统瓶颈通常在 LLM 生成
  （每次 `/chat` 是完整一次 LLM 推理），QPS 受 Ollama 并发能力限制。
- **平均/90% 响应时间**：反应端到端延迟（检索 + LLM + 后处理）。
- **错误率**：断言失败或非 200 的占比，应接近 0。

## 压测前建议

- 本地 Ollama 并发有限，压测前确认 Ollama 没被别的任务占满；
- 想测「系统」而非「LLM」的极限，可临时把 `LLM_PROVIDER=dummy` 测接口/检索的吞吐上限；
- 横向扩展用 `deploy/nginx.conf` 把流量分到多个 uvicorn worker（注意 `MEMORY_BACKEND=redis`）。
