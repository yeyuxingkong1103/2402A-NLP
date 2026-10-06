# Implementation Plan: 解析产物清洗（S3）

**Feature**: `002-parse-output-clean` | **Spec**: [spec.md](./spec.md)

## 技术上下文

| 项 | 值 |
|---|---|
| 运行时 | `D:/zg6_Project/9/med_rag/rag/python.exe`（Python 3.12.14，宪法原则 I） |
| 依赖 | Python 标准库 + `jieba`（已装 0.42.1）。**无需任何下载** |
| 输入 | `data/parsed/{doc_id}/<文档名>/auto/<文档名>_content_list.json` |
| 输出 | `data/clean/{doc_id}.blocks.jsonl` / `.dropped.jsonl` / `.report.txt` |
| 入口 | `backend/clean_parsed.py`（单文件入口，其余在 `backend/clean/`） |
| 并发 | 无。离线批处理，逐文档串行 |
| 测试 | `--self-test` 跑 D1 的 5 组基线；无单测框架（与 `specs/001` 一致） |

## Constitution Check

| 原则 | 结论 |
|---|---|
| I. 环境锁定与依赖治理 | ✅ 只用锁定解释器；`jieba` 已在环境中，无新增依赖；文档与脚本头均写绝对路径 |
| II. 无据不答与强制溯源引用 | ✅ FR-015 引文标注计数强校验；`raw_text` 保底；`block_id` 可回溯 |
| III. 密钥零硬编码 | ✅ 无任何凭据 |
| IV. 紧急症状前置响应 | ✅ FR-017：被剔除块含紧急/剂量表述即告警 |
| V. 医疗安全边界 | ✅ 不猜测、不静默丢弃；未知类型硬失败 |

## 结构

```
backend/
  clean_parsed.py        唯一入口（argparse + 编排调用）
  clean/
    errors.py            退出码与 CleanError
    paths.py             输入定位 / 输出路径
    loader.py            读 + 契约校验（未知类型即失败）
    rules_drop.py        剔除类
    rules_protect.py     保护类判据 + 前后对比
    rules_convert.py     转换类：渲染、反斜杠、实体、表格->Markdown
    space_merge.py       D1：jieba 词内空格合并（含 5 组自检基线）
    rules_repair.py      修复类：跨页断句合并 + 两项只报告
    report.py            四类分组报告
    io_utils.py          原子写出（tmp + os.replace）
    pipeline.py          编排
```

## 关键设计决定

1. **`equation` 块跳过全部字符级转换** —— LaTeX 的反斜杠是语法，反转义会毁掉公式。
2. **同页合并的额外约束** —— 实测 p0 的断句发生在**同一页**（列切分），故页距允许 {0,1}；但页距为 0 时要求中间只夹 `empty:text` 块，避免误并相邻段落。
3. **保护复检分两层** —— ① 被剔除块含紧急/剂量表述 → 报警；② `<sup>` 计数前后必须相等 → 不等即非 0 退出。
4. **`char_len` 用非空白字符数** —— docs/04 §6 写的是「中文字符数」，但整段英文会算成 0，误导下游长度判据。此处是有意偏离，已写入脚本注释。
5. **`dropped` 的 block_id 加 `x` 前缀** —— 与保留块的连续序号处于同一 6 位命名空间，加前缀避免碰撞。

## 偏离与回写

- 转换类/修复类扩展了 `docs/04 §6`「绝不改写正文」，须人工确认后回写。
- 实测契约细节（`list` 的 `sub_type`/`list_items`、`page_footnote` 保留）须回写 `docs/04 §5`。
