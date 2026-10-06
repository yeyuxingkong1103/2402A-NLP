# 角色扮演系统改造计划

## Context

当前系统是按领域（medical/education）切换 LLM 人设的问答系统：`build_prompt(domain)` 从 `config.domain_expert` 读人设。用户想改成**角色扮演系统**：用户注册时选角色（医生/教师），LLM 全程按该角色人设回答。角色信息从 users 表拆出到独立的 roles 表，人设存数据库而非硬编码在 config.py。

角色固定为 3 个（doctor/teacher/admin），预置到 roles 表，不支持自定义。

## 改动文件清单

### 1. database.py — 拆表 + 预置角色

**现状**：一张 `users` 表，`role` 是字符串字段。

**改后**：
- 新增 `Role` 模型：`id`(PK), `name`(unique: doctor/teacher/admin), `persona`(TEXT: LLM 人设描述)
- `User` 模型：`role` 字段改为 `role_id`(FK → roles.id)
- `init_db()` 末尾预置 3 条角色记录：
  - doctor → "你是一位资深临床医生，用专业但通俗的方式回答"
  - teacher → "你是一位经验丰富的教师，用启发式的方式回答"
  - admin → "你是一位知识渊博的综合专家，用严谨全面的方式回答"
- 新增辅助函数 `get_role_persona(role_name) -> str`：查 roles 表返回人设文本，供 rag_chain 调用

### 2. config.py — 删 domain_expert

- 删除 `domain_expert` 字典（人设搬到 roles 表）
- 保留 `role_domains`（权限策略不搬，仍写在 config）
- 保留其他所有配置不变

### 3. rag_chain.py — build_prompt 读角色人设

**现状**：`build_prompt(domain)` 读 `config.domain_expert.get(domain, fallback)`

**改后**：
- `build_prompt(domain, persona=None)` — persona 由调用方传入
- persona 不为 None 时用它，为 None 时用默认 fallback "严谨的中文知识库问答助手"
- `get_rag_chain(domain, user_id, persona=None)` — 缓存键加 persona，避免不同角色复用同一条链
- 内部调用 `build_prompt(domain, persona)` 传入 persona

### 4. main.py — CLI 层传角色

- `cmd_ask`/`cmd_chat` 新增 `--persona` 可选参数（字符串）
- 传给 `get_rag_chain(domain, user_id, persona=persona)`
- 不传时走默认人设（向后兼容）

### 5. api.py — API 层从 JWT 取角色人设

**现状**：`/ask` 从 JWT 取 `user["user_id"]`，调 `get_rag_chain(req.domain, user["user_id"])`

**改后**：
- 从 JWT 取 `user["role"]`（已有，无需改 JWT 逻辑）
- 调 `database.get_role_persona(user["role"])` 拿人设文本
- 传给 `get_rag_chain(req.domain, user["user_id"], persona=persona)`

### 不改的文件
- auth.py — JWT 已携带 role，`check_access` 逻辑不变
- vector_store.py / hybrid_search.py — 检索逻辑不变
- chat_store.py — Redis 历史不变
- loaders.py — 文档加载不变

## 数据库迁移

MySQL 刚切换完还没有正式数据，直接重建：
```sql
DROP DATABASE IF EXISTS rag;
CREATE DATABASE rag CHARACTER SET utf8mb4;
```
然后运行 `python -c "from database import init_db; init_db()"` 自动建表 + 预置角色。

## 验证

1. 建库建表：`python -c "from database import init_db; init_db()"` 无报错
2. 注册医生用户，ask 一个医疗问题，确认回答是临床医生口吻
3. 注册教师用户，ask 同一个问题，确认回答风格不同
4. CLI 测试：`python main.py ask "失眠症怎么治" --domain medical --persona "你是一位资深临床医生"`
5. 确认 [编号] 引用功能正常（RAG 硬约束未变）
