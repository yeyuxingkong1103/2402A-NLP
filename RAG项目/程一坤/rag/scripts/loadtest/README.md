# JMeter 压测计划运行说明

压测对象：`POST /api/v1/chat/stream`（流式问答，最重链路）为主，
`GET /health/ready` 与 `GET /api/v1/admin/documents` 作轻量对照。

## 1. 前置条件

- JMeter 5.6+（命令行模式不需要 GUI：`https://jmeter.apache.org/download_jmeter.cgi`）
- 后端已启动（`scripts/deploy/run.sh`，默认端口 8000，`/health/ready` 通过）
- 一个**已注册**的测试账号（建议提前 `python -m app.cli.create_admin --email <邮箱>`
  升级为管理员，否则 `GET /api/v1/admin/documents` 采样器会全部 403）
- Java 11+

## 2. 运行（非 GUI 模式）

账号密码通过命令行变量传入（密码会出现在进程列表里，测试机可接受；
更稳妥的做法见 §6 脚本测首字延迟——密码走环境变量）：

```bash
# 10 并发（小）
jmeter -n -t chat_stream_loadtest.jmx -l result_10.jtl \
  -JHOST=127.0.0.1 -JPORT=8000 \
  -JUSER=you@qq.com -JPASS=<密码> \
  -JTHREADS=10 -JRAMP_UP=5 -JLOOPS=3

# 50 并发（中）
jmeter -n -t chat_stream_loadtest.jmx -l result_50.jtl \
  -JHOST=<服务器IP> -JPORT=8000 \
  -JUSER=you@qq.com -JPASS=<密码> \
  -JTHREADS=50 -JRAMP_UP=15 -JLOOPS=3

# 100 并发（大）
jmeter -n -t chat_stream_loadtest.jmx -l result_100.jtl \
  -JHOST=<服务器IP> -JPORT=8000 \
  -JUSER=you@qq.com -JPASS=<密码> \
  -JTHREADS=100 -JRAMP_UP=30 -JLOOPS=2
```

三档并发请**串行跑**，且两档之间间隔几分钟（LLM/Embedding/Reranker 上游
有并发上限，混跑会互相降级，结果作废——本项目评测时已实测过：并发会把
上游打到 `RerankerApiError` 退回 RRF）。

## 3. 场景与变量

| 变量 | 含义 | 默认 |
|---|---|---|
| `HOST` / `PORT` | 后端地址 | 127.0.0.1 / 8000 |
| `THREADS` | 并发线程数 | 10 |
| `RAMP_UP` | 起压时间（秒） | 5 |
| `LOOPS` | 每线程循环次数 | 3 |
| `USER` / `PASS` | 登录账号（setUp 线程组登录取 token） | 空 |

流程：setUp 线程组先调 `POST /api/v1/auth/login` 提取 `data.access_token`
写入全局属性 `auth_token` → 主线程组三个采样器共用该 token。

主压测请求体（每个虚拟用户每次循环生成独立 `session_id`，服务端按首问自动建档）：

```json
{"session_id":"loadtest-<线程>-<循环>","character_id":"legal-assistant",
 "message":"公司解除劳动合同需要提前多久通知？",
 "options":{"top_k":8,"enable_query_rewrite":true,"enable_long_term_memory":false}}
```

`enable_long_term_memory=false`：避免压测把垃圾写进长期记忆库。

## 4. 结果口径

用 `result_*.jtl` 在 GUI 里打开聚合报告，或直接命令行生成：

```bash
jmeter -g result_100.jtl -o report_100/   # 生成 HTML 报告
```

| 指标 | jtl 来源字段 | 说明 |
|---|---|---|
| QPS | `elapsed`/时间窗 | 聚合报告 Throughput 列 |
| 平均响应 | `elapsed` | **注意**：chat/stream 的 elapsed 是"完整回答耗时"（LLM 生成完才断流），不是首字延迟 |
| P50/P95/P99 | `elapsed` | 聚合报告 50th/95th/99th Percentile |
| 错误率 | `success` | 含 HTTP 非 200、断言失败（见下） |
| 超时数 | `responseCode=Non HTTP response code` | chat/stream 响应超时设 180s，超过即计入错误 |
| 断言失败 | `success=false` + message | chat 采样器断言 SSE 必须收到 `message_end`（LLM 半途被护栏整段替换、上游报错都会失败——这些是真错误，不是压测噪音） |

### 首字延迟（重要口径说明）

JMeter 的 sampler 只在**整个响应结束**时记一次 `elapsed`，测不到 SSE
中间事件到达时间。首字延迟用本目录脚本单独测（口径：请求发出 → 首个
`event: token` 到达）：

```bash
# 方式一：账号密码走环境变量
export LOADTEST_PASSWORD=<密码>
python scripts/loadtest/first_token_latency.py \
  --host <服务器IP> --port 8000 --email you@qq.com --n 10

# 方式二：跳过登录，直接用已有令牌（Postman 跑一次 3.3 登录从响应里取）
export LOADTEST_TOKEN=<access_token>
python scripts/loadtest/first_token_latency.py \
  --host <服务器IP> --port 8000 --token-env LOADTEST_TOKEN --n 10
```

输出逐次首字/完整耗时 + P50/P95/平均。报告里两个指标分开写，
**不要把 JMeter 的 elapsed 当首字延迟**。

## 5. 成本估算（压测会真实调 LLM，先算钱再跑）

每次 chat/stream 的 token 消耗（按当前评测口径估算）：
prompt（系统提示词 + 检索上下文）≈ 1.2k tokens，completion ≈ 300~1200 tokens，
即 **每请求 ≈ 1.5k~2.4k tokens**（供应商单价按实际账单替换下表"单价"）。

| 档位 | 总请求数（THREADS×LOOPS） | 总 tokens ≈ | 费用 ≈ |
|---|---|---|---|
| 10 并发 | 30 | 5~7 万 | 单价×0.06M |
| 50 并发 | 150 | 23~36 万 | 单价×0.3M |
| 100 并发 | 200 | 30~48 万 | 单价×0.4M |

**低成本替代方案（推荐先做）**：压"检索链路"不含 LLM 生成，token 成本为零：

1. 在 JMeter GUI（或改 jmx）里**禁用** `POST /api/v1/chat/stream` 采样器，
   **启用**已内置的 `POST /api/v1/legal/search` 采样器（enabled="false" 那个），
   用 10/50/100 三档把整条"向量+关键词+重排"链路压透；
2. LLM 生成链路只用**小并发短循环**单独压（如 THREADS=10, LOOPS=2），
   结论以首字延迟脚本 + 单档聚合报告为准。

## 6. 注意事项

- `GET /api/v1/admin/documents` 需管理员 token，非管理员账号会 403（计入错误率），
  测前先 `create_admin`；
- 压测会话（`loadtest-*`）会真实落库，压完可按前缀清理；
- response_timeout（chat 180s / search 30s / health 10s）写在各采样器里，
  慢机器先压 10 并发观察 P99 再决定是否放宽。
