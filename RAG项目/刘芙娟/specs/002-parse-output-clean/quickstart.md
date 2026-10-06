# Quickstart: 清洗解析产物

## 前提

- 已跑过 S2 解析，`data/parsed/{doc_id}/` 下存在 `*_content_list.json`
- 运行环境：`D:/zg6_Project/9/med_rag/rag/python.exe`（**不要**用系统 python）

## 跑

```bash
cd D:/zg6_Project/9/med_rag

# 自检（D1 空格合并的 5 组基线），不碰任何产物
rag/python.exe backend/clean_parsed.py --self-test

# 只出报告、不写产物
rag/python.exe backend/clean_parsed.py --dry-run

# 全量清洗
rag/python.exe backend/clean_parsed.py

# 只洗一份
rag/python.exe backend/clean_parsed.py d6da41b5d356
```

## 看

1. 终端报告 —— 四类各自的命中数与规则明细
2. `data/clean/{doc_id}.dropped.jsonl` —— **先看这个**。每行 `drop_rule` 是否合理，抽查 5~10 条
3. `data/clean/{doc_id}.blocks.jsonl` —— 每行的 `heading_path` / `page` 是否符合预期

## 判断清洗是否可信（三条硬指标）

| 指标 | 期望 | 在哪看 |
|---|---|---|
| `<sup>` 引文标注计数 | 清洗前后**完全相等**（样例基线 23） | 报告「保护类」 |
| 被剔除块含紧急/剂量表述 | **0 条** | 报告「保护类」，出现即须人工确认 |
| 丢弃比例 | 正常 15%~20%；**>40% 告警** | 报告「剔除类」 |

## 退出码

`0` 成功 · `2` 输入问题 · `3` 未知块类型 · `4` 保护类被误杀 · `5` 清洗后为空 · `6` 缺依赖

出现 3 或 4 时**不要绕过**：3 意味着 MinerU 契约变了，4 意味着知识库正在缺一块。
