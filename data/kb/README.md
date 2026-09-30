# 知识库目录说明

```
data/kb/
├── shared/               # 公共知识库：所有角色可见（平台说明、合规声明）
├── financial_planner/    # 角色：金融理财师
├── scientist/            # 角色：科学家
└── lawyer/               # 角色：律师
```

## 目录与角色作用域的对应关系

* 目录名 = 角色的 `id`（见 `configs/roles.yaml`），入库时写入 Milvus 的 `scope` 字段；
* 检索时按 `scope == "<role_id>" or scope == "shared"` 过滤，实现角色间知识隔离；
* 若在 `roles.yaml` 中启用其他角色（医生、教师等），新建同名目录并放入 `.md` 文件即可。

## 文件格式

支持 `.md` / `.markdown` / `.txt` / `.pdf`。推荐 Markdown，结构越清晰分块质量越高。

可选 front-matter 覆盖标题与标签：

```markdown
---
title: 资产配置基础
tags: 资产配置, 风险, 应急金
---

# 资产配置基础
...
```

## 分块规则（`configs/config.yaml` → `ingest`）

1. 按 `#` / `##` / `###` 标题切小节，保留「标题路径」作为引用定位；
2. 小节内按空行切段，表格（`|` 开头）保持整体不拆散；
3. 段落累积到 `chunk_size`(700) 成为一个块，相邻块重叠 `chunk_overlap`(120) 字符；
4. 单段超过 `max_chunk_chars`(1400) 时按句子切分；
5. 小于 `min_chunk_chars`(80) 的尾块优先并入上一块。

## 入库命令

```powershell
# 全量重建（启用角色 + shared）
D:\an\envs\rags_\python.exe tools\ingest_cli.py --all --recreate

# 只更新某个角色
D:\an\envs\rags_\python.exe tools\ingest_cli.py --roles lawyer

# 单文件入库
D:\an\envs\rags_\python.exe tools\ingest_cli.py --file data\kb\lawyer\01_contract_basics.md --role lawyer

# 查看知识库现状
D:\an\envs\rags_\python.exe tools\ingest_cli.py --report
```

> 入库完成后 `kb_version` 自增，内存中的 BM25 索引与 Redis 检索缓存会自动失效重建，
> 无需重启服务。
