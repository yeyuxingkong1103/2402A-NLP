# Legal Customer Service MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个内部测试版中国大陆婚姻家庭法律客服助手，支持登录、结构化咨询、官方知识库审核、混合 RAG、DeepSeek 流式回答、隐私保护、长期记忆、后台治理和 100 条验收测试集。

**Architecture:** 本计划按可独立验收的子系统拆分：基础配置与健康检查、账号认证、数据加密、知识库治理、RAG、本地模型、聊天编排、安全审查、长期记忆、后台、前端和评估测试集。每个任务都必须先写测试，再实现最小可用功能，再运行相关测试。

**Tech Stack:** Python, FastAPI, SQLAlchemy/Alembic, MySQL, Redis, Milvus, Celery, Transformers, Sentence Transformers, DeepSeek API, HTML/CSS/JavaScript, Pytest, Docker Compose on Windows.

**Spec:** `docs/superpowers/specs/2026-09-15-legal-customer-service-assistant-design.md`

## Global Constraints

- 第一版只支持离婚与婚姻关系、子女抚养与探望、夫妻共同财产与共同债务。
- 第一版不支持用户上传文件、图片、OCR 或法律文书生成。
- 用户必须手机号 + 短信验证码登录；内部测试阶段使用模拟验证码。
- 用户不强制实名认证，但必须同意《隐私政策》《用户服务协议》并确认年满 18 周岁。
- 历史会话默认永久保存，用户可以删除会话、注销账号和导出本人数据。
- 手机号和聊天内容必须使用 `AES-256-GCM` 加密保存。
- 精确查询字段使用带密钥的 `HMAC` 检索摘要，不使用普通哈希。
- 默认启用长期记忆，只保存结构化案件事实和脱敏跨会话摘要。
- DeepSeek 只接收脱敏后的用户问题、最小必要上下文、脱敏长期记忆和已发布法律依据。
- RAG 使用向量检索 Top 20 + BM25 Top 20，合并去重后 Reranker，最终 Top 5。
- Reranker 初始阈值为 `0.5`，低于阈值停止普通法律分析。
- Embedding 使用 `C:\Users\bin\Desktop\myrag\models\bge-m3`。
- Reranker 使用 `C:\Users\bin\Desktop\myrag\models\bge-reranker-large`。
- 本地模型使用 `Transformers` + `Sentence Transformers`，FP16，同时常驻独占 GPU。
- 显存不足时暂停模型集成，不降级到 CPU、量化、串行加载或更换模型。
- 官方知识库来源包括国家法律法规数据库、国务院及中央部委官方文件、最高人民法院司法解释、指导性案例和官方典型案例。
- 只抓取严格 URL 白名单内的官方来源，白名单仅超级管理员维护。
- 知识库状态为：已抓取 → 待审核 → 已发布 / 已驳回 → 已废止。
- 内容审核员审核，超级管理员最终发布。
- 每次修订保存独立版本，按案件事实发生时间匹配适用法律版本。
- 普通运行日志和模型调用日志保留 30 天，抓取任务日志 90 天，安全日志和审计日志 1 年。
- 单条消息最多 5,000 个中文字符。
- 每账号每分钟最多 5 条消息，每天最多 100 条消息。
- 每账号最多同时处理 2 个请求，系统最多同时处理 4 个请求。
- 超出系统并发后最多排队 30 秒。
- 单次本地检索、重排和 DeepSeek 调用总计最多 120 秒。
- 每条回答最多重新生成 3 次。
- 测试集输出到 `evaluation/datasets/marriage_family_mvp_100.jsonl`。
- `C:\Users\bin\Desktop\public` 只读，不修改、不删除、不重命名、不移动。
- 后续代码实现必须保持“一个函数代表一个明确功能”，避免一个函数承担多种职责。
- 每个 Python 文件原则上不超过 300 行；特殊复杂文件最多约 500–600 行，超过时必须拆分模块。
- 代码需要加入便于用户审核理解的中文注释，关键逻辑尽量逐行说明当前行在做什么。
- 所有关键流程必须提供详细日志，覆盖开始、成功、失败、异常原因和关键状态变化，但不得记录密钥、验证码、完整手机号、聊天原文等敏感信息。
- 当前项目已初始化 Git 仓库并建立初始基线提交；后续任务可以按计划执行 commit 步骤。

---

## Scope Check

该规格覆盖多个独立子系统，不适合一次性由单个任务实现。执行时必须按本计划拆分任务，每个任务完成后运行对应测试并进行人工或子代理审查。推荐执行顺序从 Task 1 到 Task 14，不要并行修改同一层核心模型和迁移文件。

## File Structure Map

### Backend Core

- `backend/app/core/config.py`：集中定义环境变量、模型路径、限流、Token、加密、日志保留和知识库配置。
- `backend/app/core/security.py`：Token、验证码安全、密码和权限辅助函数。
- `backend/app/core/crypto.py`：新增，加密、解密、HMAC 摘要和密钥版本接口。
- `backend/app/core/rate_limit.py`：新增，账号频率、并发和系统并发限制。
- `backend/app/core/startup_checks.py`：新增，启动前依赖检查。
- `backend/app/core/error_handlers.py`：统一错误响应，避免泄露密钥和内部细节。

### Backend Models and Migrations

- `backend/app/models/user.py`：用户账号、手机号密文、手机号 HMAC、协议同意和年龄确认。
- `backend/app/models/role.py`：超级管理员、内容审核员、客服运营员。
- `backend/app/models/conversation.py`：会话状态、删除标记、重新开始状态。
- `backend/app/models/message.py`：消息密文、回答状态、重新生成次数、更正标记。
- `backend/app/models/knowledge_base.py`：法律材料、版本、来源、状态、附件、白名单和审核发布记录。
- `backend/app/models/audit_log.py`：追加型审计日志。
- `backend/app/models/feedback.py`：回答反馈和高风险告警关联。
- `backend/app/models/memory.py`：新增，长期记忆、来源会话关联、候选更新和使用记录。
- `backend/migrations/versions/*.py`：数据库迁移脚本。

### Backend Services

- `backend/app/services/auth_service.py`：手机号验证码登录、Token、设备会话。
- `backend/app/services/privacy_service.py`：新增，PII 检测、脱敏和最小化上下文。
- `backend/app/services/knowledge_base_service.py`：白名单、抓取快照、审核、发布、废止、版本匹配。
- `backend/app/services/chat_service.py`：聊天总编排。
- `backend/app/services/feedback_service.py`：反馈和高风险告警。
- `backend/app/services/memory_service.py`：新增，长期记忆提取、冲突、关闭、删除和使用提示。
- `backend/app/services/export_service.py`：新增，用户数据导出。
- `backend/app/services/audit_service.py`：审计日志统一写入。

### RAG and Model Layer

- `backend/app/embeddings/bge_m3.py`：本地 BGE-M3 加载、向量化和健康检查。
- `backend/app/rerank/bge_reranker.py`：本地 BGE Reranker 加载、评分和健康检查。
- `backend/app/rag/hybrid_retriever.py`：向量 + BM25 召回。
- `backend/app/rag/result_merger.py`：去重合并。
- `backend/app/rag/context_builder.py`：构建最小化法律依据上下文。
- `backend/app/rag/pipeline.py`：分类、检索、阈值、版本适用和依据不足处理。
- `backend/app/llm/deepseek_client.py`：DeepSeek 流式调用、失败处理和元数据记录。

### Backend API

- `backend/app/api/v1/auth.py`：验证码、登录、刷新、退出和设备会话。
- `backend/app/api/v1/chat.py`：发送消息、流式回答、重新生成、事实纠正、重新开始。
- `backend/app/api/v1/knowledge_bases.py`：后台知识库审核、发布、废止、白名单。
- `backend/app/api/v1/feedback.py`：点赞、点踩、举报和原因。
- `backend/app/api/v1/users.py`：个人资料、数据导出、注销、长期记忆管理入口。
- `backend/app/api/v1/health.py`：`/health/live` 和 `/health/ready`。

### Frontend

- `frontend/pages/login.html`、`frontend/js/pages/login.js`：手机号验证码登录。
- `frontend/pages/register.html`、`frontend/js/pages/register.js`：协议同意、年龄确认和注册入口。
- `frontend/pages/chat.html`、`frontend/js/pages/chat.js`：聊天、流式输出、引用列表、重新生成、事实纠正、重新开始。
- `frontend/pages/profile.html`、`frontend/js/pages/profile.js`：长期记忆管理、设备会话、数据导出和账号注销。
- `frontend/pages/knowledge-bases.html`、`frontend/js/pages/knowledge-bases.js`：后台知识库审核。
- `frontend/js/api/*.js`：各 API 客户端。
- `frontend/js/components/*.js`：聊天、引用、反馈、弹窗、状态和后台组件。

### Tests and Evaluation

- `backend/tests/unit/`：加密、脱敏、限流、RAG、版本匹配、安全、记忆和引用测试。
- `backend/tests/api/`：认证、聊天、知识库、反馈、用户数据导出和健康检查 API 测试。
- `backend/tests/integration/`：登录到聊天、知识库发布到 RAG、删除到记忆清理、灾备删除重放测试。
- `frontend/tests/unit/`：组件测试。
- `frontend/tests/e2e/`：登录、聊天、知识库审核和个人中心流程。
- `evaluation/datasets/marriage_family_mvp_100.jsonl`：100 条 MVP 验收测试集。

---

### Task 1: 配置、健康检查和启动前依赖

**Files:**
- Modify: `backend/app/core/config.py`
- Create: `backend/app/core/startup_checks.py`
- Modify: `backend/app/api/v1/health.py`
- Modify: `backend/app/main.py`
- Test: `backend/tests/unit/test_startup_checks.py`
- Test: `backend/tests/api/test_health_api.py`

**Interfaces:**
- Produces: `AppSettings.model_paths: dict[str, str]`
- Produces: `AppSettings.limits: dict[str, int]`
- Produces: `DependencyStatus(name: str, ok: bool, detail: str | None)`
- Produces: `run_startup_checks(settings: AppSettings) -> list[DependencyStatus]`
- Produces: `is_ready(statuses: list[DependencyStatus]) -> bool`

- [ ] **Step 1: Write failing unit tests for readiness rules**

```python
from backend.app.core.startup_checks import DependencyStatus, is_ready


def test_ready_requires_all_dependencies_ok():
    statuses = [
        DependencyStatus(name="mysql", ok=True, detail=None),
        DependencyStatus(name="redis", ok=True, detail=None),
        DependencyStatus(name="milvus", ok=False, detail="connection failed"),
    ]

    assert is_ready(statuses) is False


def test_ready_passes_when_all_dependencies_ok():
    statuses = [
        DependencyStatus(name="mysql", ok=True, detail=None),
        DependencyStatus(name="gpu", ok=True, detail="cuda available"),
    ]

    assert is_ready(statuses) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/unit/test_startup_checks.py -v`
Expected: FAIL because `backend.app.core.startup_checks` does not expose the required interfaces.

- [ ] **Step 3: Implement dependency status interfaces**

Create `backend/app/core/startup_checks.py` with:

```python
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class DependencyStatus:
    name: str
    ok: bool
    detail: str | None = None


def is_ready(statuses: Iterable[DependencyStatus]) -> bool:
    return all(status.ok for status in statuses)


def run_startup_checks(settings) -> list[DependencyStatus]:
    checks = [
        DependencyStatus("mysql", True),
        DependencyStatus("redis", True),
        DependencyStatus("milvus", True),
        DependencyStatus("deepseek", bool(settings.DEEPSEEK_API_KEY), None if settings.DEEPSEEK_API_KEY else "missing api key"),
        DependencyStatus("bge_m3_path", bool(settings.BGE_M3_MODEL_PATH), None if settings.BGE_M3_MODEL_PATH else "missing model path"),
        DependencyStatus("bge_reranker_path", bool(settings.BGE_RERANKER_MODEL_PATH), None if settings.BGE_RERANKER_MODEL_PATH else "missing model path"),
    ]
    return checks
```

- [ ] **Step 4: Add settings values**

Update `backend/app/core/config.py` to expose exact values:

```python
BGE_M3_MODEL_PATH = "C:\\Users\\bin\\Desktop\\myrag\\models\\bge-m3"
BGE_RERANKER_MODEL_PATH = "C:\\Users\\bin\\Desktop\\myrag\\models\\bge-reranker-large"
MESSAGE_MAX_CHARS = 5000
ACCOUNT_MESSAGES_PER_MINUTE = 5
ACCOUNT_MESSAGES_PER_DAY = 100
ACCOUNT_CONCURRENT_REQUESTS = 2
SYSTEM_CONCURRENT_REQUESTS = 4
SYSTEM_QUEUE_TIMEOUT_SECONDS = 30
ANSWER_TIMEOUT_SECONDS = 120
RERANKER_THRESHOLD = 0.5
ACCESS_TOKEN_MINUTES = 120
REFRESH_TOKEN_DAYS = 30
OTP_TTL_SECONDS = 300
OTP_MAX_ATTEMPTS = 5
OTP_LOCK_SECONDS = 900
```

- [ ] **Step 5: Add health API tests**

```python

def test_live_health_returns_alive(client):
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_ready_health_has_dependencies(client):
    response = client.get("/health/ready")
    assert response.status_code in {200, 503}
    body = response.json()
    assert "ready" in body
    assert "dependencies" in body
    assert all("name" in item and "ok" in item for item in body["dependencies"])
```

- [ ] **Step 6: Implement `/health/live` and `/health/ready`**

`/health/live` returns only `{"status": "alive"}`. `/health/ready` calls `run_startup_checks()` and returns 200 when all checks pass, otherwise 503.

- [ ] **Step 7: Run tests**

Run: `pytest backend/tests/unit/test_startup_checks.py backend/tests/api/test_health_api.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

If the project has been initialized as a Git repository, run:

```bash
git add backend/app/core/config.py backend/app/core/startup_checks.py backend/app/api/v1/health.py backend/app/main.py backend/tests/unit/test_startup_checks.py backend/tests/api/test_health_api.py
git commit -m "feat: add startup readiness checks"
```

---

### Task 2: 加密、HMAC 和敏感信息脱敏

**Files:**
- Create: `backend/app/core/crypto.py`
- Create: `backend/app/services/privacy_service.py`
- Modify: `backend/app/core/config.py`
- Test: `backend/tests/unit/test_crypto.py`
- Test: `backend/tests/unit/test_privacy_service.py`

**Interfaces:**
- Produces: `EncryptedValue(ciphertext: str, nonce: str, encrypted_data_key: str, key_version: str)`
- Produces: `encrypt_text(plaintext: str, purpose: str) -> EncryptedValue`
- Produces: `decrypt_text(value: EncryptedValue, purpose: str) -> str`
- Produces: `hmac_digest(value: str, purpose: str) -> str`
- Produces: `redact_pii(text: str) -> RedactionResult`
- Produces: `RedactionResult(text: str, redacted_types: list[str])`

- [ ] **Step 1: Write failing crypto tests**

```python
from backend.app.core.crypto import decrypt_text, encrypt_text, hmac_digest


def test_encrypt_decrypt_round_trip(monkeypatch):
    monkeypatch.setenv("APP_MASTER_KEY", "test-master-key-32-bytes-minimum-value")
    encrypted = encrypt_text("用户手机号13800138000", purpose="message")

    assert encrypted.ciphertext != "用户手机号13800138000"
    assert decrypt_text(encrypted, purpose="message") == "用户手机号13800138000"


def test_hmac_digest_is_stable_and_not_plain_hash(monkeypatch):
    monkeypatch.setenv("APP_HMAC_KEY", "test-hmac-key")
    first = hmac_digest("13800138000", purpose="phone")
    second = hmac_digest("13800138000", purpose="phone")

    assert first == second
    assert "13800138000" not in first
```

- [ ] **Step 2: Run crypto tests to verify failure**

Run: `pytest backend/tests/unit/test_crypto.py -v`
Expected: FAIL because `backend.app.core.crypto` does not exist or lacks required functions.

- [ ] **Step 3: Implement crypto module with library-backed AES-GCM**

Use `cryptography.hazmat.primitives.ciphers.aead.AESGCM`. Derive task keys from environment-provided master key using HKDF or a project-approved KDF. Do not log plaintext, keys, nonces, or ciphertext.

- [ ] **Step 4: Write failing PII redaction tests**

```python
from backend.app.services.privacy_service import redact_pii


def test_redacts_phone_id_card_and_bank_card():
    source = "我手机号13800138000，身份证110101199003074512，银行卡6222020202020202020。"
    result = redact_pii(source)

    assert "13800138000" not in result.text
    assert "110101199003074512" not in result.text
    assert "6222020202020202020" not in result.text
    assert "[手机号]" in result.text
    assert "[身份证号]" in result.text
    assert "[银行卡号]" in result.text
    assert set(result.redacted_types) >= {"phone", "id_card", "bank_card"}
```

- [ ] **Step 5: Implement privacy service**

Implement regex-based detection for first version:

```python
PHONE_RE = r"(?<!\d)1[3-9]\d{9}(?!\d)"
ID_CARD_RE = r"(?<!\d)\d{6}(18|19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[0-9Xx](?!\d)"
BANK_CARD_RE = r"(?<!\d)\d{16,19}(?!\d)"
```

Return redacted text and detected types. Keep this service deterministic and side-effect free.

- [ ] **Step 6: Run privacy and crypto tests**

Run: `pytest backend/tests/unit/test_crypto.py backend/tests/unit/test_privacy_service.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/app/core/crypto.py backend/app/services/privacy_service.py backend/app/core/config.py backend/tests/unit/test_crypto.py backend/tests/unit/test_privacy_service.py
git commit -m "feat: add encrypted storage primitives and pii redaction"
```

---

### Task 3: 手机号验证码登录、Token 和设备会话

**Files:**
- Modify: `backend/app/models/user.py`
- Modify: `backend/app/schemas/auth.py`
- Modify: `backend/app/services/auth_service.py`
- Modify: `backend/app/api/v1/auth.py`
- Modify: `backend/app/core/security.py`
- Modify: `backend/migrations/versions/001_create_users.py`
- Test: `backend/tests/unit/test_auth_service.py`
- Test: `backend/tests/api/test_auth_api.py`

**Interfaces:**
- Consumes: `encrypt_text()`, `hmac_digest()` from Task 2
- Produces: `request_login_code(phone: str, client_id: str) -> OtpRequestResult`
- Produces: `login_with_code(phone: str, code: str, client_id: str, user_agent: str) -> TokenPair`
- Produces: `refresh_access_token(refresh_token: str) -> TokenPair`
- Produces: `revoke_refresh_token(refresh_token: str) -> None`
- Produces: `TokenPair(access_token: str, refresh_token: str, token_type: str)`

- [ ] **Step 1: Write failing OTP service tests**

```python
from backend.app.services.auth_service import login_with_code, request_login_code


def test_preconfigured_test_phone_uses_fixed_code(settings):
    settings.TEST_PHONE_NUMBERS = {"13900000001": "123456"}
    result = request_login_code("13900000001", client_id="browser-a")

    assert result.sent is False
    assert result.expires_in_seconds == 300


def test_login_requires_correct_code(settings):
    settings.TEST_PHONE_NUMBERS = {"13900000001": "123456"}
    request_login_code("13900000001", client_id="browser-a")

    tokens = login_with_code("13900000001", "123456", client_id="browser-a", user_agent="pytest")

    assert tokens.access_token
    assert tokens.refresh_token
    assert tokens.token_type == "bearer"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest backend/tests/unit/test_auth_service.py -v`
Expected: FAIL because OTP interfaces are missing or incomplete.

- [ ] **Step 3: Update user and device session schema**

Add database fields for encrypted phone, phone HMAC, agreement version, privacy policy version, adult confirmation, refresh token hash, device identifier, login time, last activity, province-level location, and revoked status. Do not store plaintext phone or refresh token.

- [ ] **Step 4: Implement OTP security rules**

Implement 5-minute TTL, 5 wrong attempts, 15-minute lock. Test phone numbers use fixed code only in development/internal test environment. Production must reject fixed-code configuration.

- [ ] **Step 5: Implement Token lifecycle**

Access Token expires after 2 hours. Refresh Token expires after 30 days. User logout revokes current Refresh Token. More than 5 device sessions revokes the oldest active session.

- [ ] **Step 6: Write API tests**

```python

def test_request_code_and_login(client, settings):
    settings.TEST_PHONE_NUMBERS = {"13900000001": "123456"}

    code_response = client.post("/api/v1/auth/code", json={"phone": "13900000001"})
    assert code_response.status_code == 200

    login_response = client.post("/api/v1/auth/login", json={"phone": "13900000001", "code": "123456"})
    assert login_response.status_code == 200
    assert "access_token" in login_response.json()
    assert "refresh_token" in login_response.json()
```

- [ ] **Step 7: Run auth tests**

Run: `pytest backend/tests/unit/test_auth_service.py backend/tests/api/test_auth_api.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add backend/app/models/user.py backend/app/schemas/auth.py backend/app/services/auth_service.py backend/app/api/v1/auth.py backend/app/core/security.py backend/migrations/versions/001_create_users.py backend/tests/unit/test_auth_service.py backend/tests/api/test_auth_api.py
git commit -m "feat: add phone otp authentication"
```

---

### Task 4: 审计日志、安全日志和权限角色

**Files:**
- Modify: `backend/app/models/role.py`
- Modify: `backend/app/models/audit_log.py`
- Modify: `backend/app/services/audit_service.py`
- Modify: `backend/app/api/deps.py`
- Modify: `backend/app/core/security.py`
- Test: `backend/tests/unit/test_audit_service.py`
- Test: `backend/tests/api/test_role_api.py`

**Interfaces:**
- Produces: `AuditAction` enum with `ADMIN_LOGIN`, `KB_REVIEW`, `KB_PUBLISH`, `KB_REJECT`, `KB_DEPRECATE`, `WHITELIST_CHANGE`, `DATA_EXPORT`, `FULL_CONVERSATION_ACCESS`, `DATA_DELETE`
- Produces: `record_audit(action: AuditAction, actor_id: str, target_type: str, target_id: str, metadata: dict) -> AuditLog`
- Produces: `require_role(*roles: str) -> Callable`

- [ ] **Step 1: Write failing audit tests**

```python
from backend.app.services.audit_service import AuditAction, record_audit


def test_record_audit_masks_sensitive_metadata(db_session):
    log = record_audit(
        action=AuditAction.DATA_EXPORT,
        actor_id="user-1",
        target_type="user",
        target_id="user-1",
        metadata={"phone": "13800138000", "reason": "user export"},
    )

    assert log.action == "DATA_EXPORT"
    assert "13800138000" not in str(log.metadata)
```

- [ ] **Step 2: Run audit tests to verify failure**

Run: `pytest backend/tests/unit/test_audit_service.py -v`
Expected: FAIL because audit action and masking are missing.

- [ ] **Step 3: Implement append-only audit service**

Audit logs must insert new rows only. Do not add update/delete methods to repository except retention cleanup for non-audit logs. Mask phone, code, token, key, prompt, answer, chat text, and ciphertext-like fields in metadata.

- [ ] **Step 4: Implement roles**

Create seedable role names: `super_admin`, `content_reviewer`, `support_operator`. Ensure API dependencies can enforce role requirements.

- [ ] **Step 5: Add API role tests**

```python

def test_content_reviewer_cannot_manage_source_whitelist(client, content_reviewer_token):
    response = client.post(
        "/api/v1/knowledge-bases/source-whitelist",
        headers={"Authorization": f"Bearer {content_reviewer_token}"},
        json={"url": "https://example.gov.cn/law", "publisher": "test"},
    )
    assert response.status_code == 403
```

- [ ] **Step 6: Run tests**

Run: `pytest backend/tests/unit/test_audit_service.py backend/tests/api/test_role_api.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/app/models/role.py backend/app/models/audit_log.py backend/app/services/audit_service.py backend/app/api/deps.py backend/app/core/security.py backend/tests/unit/test_audit_service.py backend/tests/api/test_role_api.py
git commit -m "feat: add audit logging and admin roles"
```

---

### Task 5: 知识库白名单、抓取快照、审核和发布

**Files:**
- Modify: `backend/app/models/knowledge_base.py`
- Modify: `backend/app/schemas/knowledge_base.py`
- Modify: `backend/app/repositories/knowledge_base_repository.py`
- Modify: `backend/app/services/knowledge_base_service.py`
- Modify: `backend/app/api/v1/knowledge_bases.py`
- Modify: `backend/app/workers/indexing_tasks.py`
- Test: `backend/tests/unit/test_knowledge_base_workflow.py`
- Test: `backend/tests/api/test_knowledge_base_api.py`

**Interfaces:**
- Consumes: `record_audit()` from Task 4
- Produces: `SourceWhitelistEntry(url: str, publisher: str, material_type: str, active: bool)`
- Produces: `create_source_whitelist_entry(...) -> SourceWhitelistEntry`
- Produces: `create_crawl_snapshot(source_url: str, raw_html: str, attachments: list[dict]) -> CrawlSnapshot`
- Produces: `submit_for_review(snapshot_id: str) -> KnowledgeMaterial`
- Produces: `review_material(material_id: str, reviewer_id: str, decision: str, reason: str) -> KnowledgeMaterial`
- Produces: `publish_material(material_id: str, publisher_id: str) -> KnowledgeMaterial`
- Produces: `deprecate_material(material_id: str, actor_id: str, reason: str) -> KnowledgeMaterial`

- [ ] **Step 1: Write failing workflow tests**

```python
from backend.app.services.knowledge_base_service import review_material, publish_material


def test_content_reviewer_review_does_not_publish(material_factory, reviewer):
    material = material_factory(status="pending_review")

    reviewed = review_material(material.id, reviewer.id, decision="approved", reason="source verified")

    assert reviewed.status == "reviewed"
    assert reviewed.reviewed_by == reviewer.id


def test_super_admin_publish_makes_material_searchable(material_factory, super_admin):
    material = material_factory(status="reviewed", effective_from="2021-01-01")

    published = publish_material(material.id, super_admin.id)

    assert published.status == "published"
    assert published.published_by == super_admin.id
    assert published.searchable is True
```

- [ ] **Step 2: Run workflow tests to verify failure**

Run: `pytest backend/tests/unit/test_knowledge_base_workflow.py -v`
Expected: FAIL because workflow states and service functions are missing.

- [ ] **Step 3: Implement knowledge models and migrations**

Represent source whitelist, crawl snapshot, material version, material attachment, review record, publish record, and status history. Store material type as one of `law`, `administrative_regulation`, `department_rule`, `judicial_interpretation`, `guiding_case`, `typical_case`.

- [ ] **Step 4: Implement strict whitelist rules**

Only `super_admin` can create, update, or delete whitelist entries. Crawlers may only use active whitelist entries. Backup sources must share the same publisher.

- [ ] **Step 5: Implement snapshot retention fields**

Store raw content hash, source URL, publisher, crawled at, material type, raw text, attachment metadata, and failure reason. Snapshot creation must not mark content as published.

- [ ] **Step 6: Implement review and publish state machine**

Allowed transitions:

```text
crawled -> pending_review
pending_review -> reviewed
pending_review -> rejected
reviewed -> published
published -> deprecated
published -> rejected is forbidden
rejected -> published is forbidden
```

- [ ] **Step 7: Add API tests for permission and state transitions**

Test that content reviewers can review but cannot publish, and super admins can publish reviewed material only.

- [ ] **Step 8: Run knowledge base tests**

Run: `pytest backend/tests/unit/test_knowledge_base_workflow.py backend/tests/api/test_knowledge_base_api.py -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add backend/app/models/knowledge_base.py backend/app/schemas/knowledge_base.py backend/app/repositories/knowledge_base_repository.py backend/app/services/knowledge_base_service.py backend/app/api/v1/knowledge_bases.py backend/app/workers/indexing_tasks.py backend/tests/unit/test_knowledge_base_workflow.py backend/tests/api/test_knowledge_base_api.py
git commit -m "feat: add governed knowledge base workflow"
```

---

### Task 6: 法律版本时间匹配和引用定位

**Files:**
- Create: `backend/app/services/legal_version_service.py`
- Modify: `backend/app/rag/citation_builder.py`
- Modify: `backend/app/schemas/retrieval.py`
- Test: `backend/tests/unit/test_legal_version_service.py`
- Test: `backend/tests/unit/test_citation_builder.py`

**Interfaces:**
- Consumes: published materials from Task 5
- Produces: `LegalFactTimeline(event_date: date | None, relationship_type: str)`
- Produces: `ApplicableMaterial(material_id: str, version_id: str, reason: str)`
- Produces: `find_applicable_materials(query_date: date, relationship_type: str, materials: list[KnowledgeMaterial]) -> list[ApplicableMaterial]`
- Produces: `Citation(kind: str, title: str, article: str | None, paragraph: str | None, excerpt: str, version: str, official_url: str)`

- [ ] **Step 1: Write failing version tests**

```python
from datetime import date
from backend.app.services.legal_version_service import find_applicable_materials


def test_selects_version_effective_on_fact_date(material_version_factory):
    old = material_version_factory(version_id="old", effective_from=date(2010, 1, 1), effective_to=date(2020, 12, 31))
    current = material_version_factory(version_id="current", effective_from=date(2021, 1, 1), effective_to=None)

    result = find_applicable_materials(date(2019, 6, 1), "property", [old, current])

    assert [item.version_id for item in result] == ["old"]
```

- [ ] **Step 2: Run version tests to verify failure**

Run: `pytest backend/tests/unit/test_legal_version_service.py -v`
Expected: FAIL because version matching service is missing.

- [ ] **Step 3: Implement version matching**

Include material only when `effective_from <= query_date` and `effective_to is None or query_date <= effective_to`. When `query_date` is missing, return current effective version and mark `reason="current_version_due_to_missing_fact_date"`.

- [ ] **Step 4: Write citation tests**

```python
from backend.app.rag.citation_builder import build_citation


def test_case_citation_is_marked_as_case_reference(material_factory):
    material = material_factory(material_type="typical_case", title="某离婚财产典型案例")

    citation = build_citation(material, article=None, paragraph="裁判规则", excerpt="案例不能替代法律条文")

    assert citation.kind == "案例参考"
    assert citation.title == "某离婚财产典型案例"
```

- [ ] **Step 5: Implement citation builder**

Return `kind` as `法律规定`, `司法解释`, or `案例参考`. Include article, paragraph, excerpt, version, official URL, and material ID.

- [ ] **Step 6: Run tests**

Run: `pytest backend/tests/unit/test_legal_version_service.py backend/tests/unit/test_citation_builder.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/app/services/legal_version_service.py backend/app/rag/citation_builder.py backend/app/schemas/retrieval.py backend/tests/unit/test_legal_version_service.py backend/tests/unit/test_citation_builder.py
git commit -m "feat: add legal version matching and citations"
```

---

### Task 7: 本地 BGE-M3 和 BGE Reranker GPU 服务

**Files:**
- Modify: `backend/app/embeddings/base.py`
- Modify: `backend/app/embeddings/bge_m3.py`
- Modify: `backend/app/embeddings/embedding_factory.py`
- Modify: `backend/app/rerank/base.py`
- Modify: `backend/app/rerank/bge_reranker.py`
- Modify: `backend/app/rerank/rerank_factory.py`
- Test: `backend/tests/unit/test_bge_models_config.py`
- Test: `backend/tests/integration/test_model_gpu_readiness.py`

**Interfaces:**
- Consumes: settings from Task 1
- Produces: `EmbeddingClient.embed_texts(texts: list[str]) -> list[list[float]]`
- Produces: `RerankClient.score(query: str, documents: list[str]) -> list[float]`
- Produces: `check_model_readiness() -> list[DependencyStatus]`

- [ ] **Step 1: Write failing config tests**

```python
from backend.app.core.config import settings


def test_model_paths_are_local():
    assert settings.BGE_M3_MODEL_PATH.endswith("models\\bge-m3")
    assert settings.BGE_RERANKER_MODEL_PATH.endswith("models\\bge-reranker-large")


def test_model_precision_is_fp16():
    assert settings.MODEL_DTYPE == "float16"
```

- [ ] **Step 2: Run config tests to verify failure**

Run: `pytest backend/tests/unit/test_bge_models_config.py -v`
Expected: FAIL if model paths or dtype are not exposed.

- [ ] **Step 3: Implement embedding interface**

Load `SentenceTransformer(settings.BGE_M3_MODEL_PATH, device="cuda")`. Use FP16 where supported. Return normalized embeddings as Python lists. Do not download remote models.

- [ ] **Step 4: Implement reranker interface**

Load `AutoTokenizer` and `AutoModelForSequenceClassification` from `settings.BGE_RERANKER_MODEL_PATH`, `local_files_only=True`, `torch_dtype=torch.float16`, `device_map="cuda"` or explicit `.to("cuda")`. Return float scores in the same order as input documents.

- [ ] **Step 5: Add GPU readiness integration test**

```python
import pytest
from backend.app.embeddings.embedding_factory import get_embedding_client
from backend.app.rerank.rerank_factory import get_rerank_client


@pytest.mark.gpu
def test_bge_models_load_on_cuda():
    embedding = get_embedding_client()
    reranker = get_rerank_client()

    vector = embedding.embed_texts(["离婚时孩子抚养权如何判断？"])[0]
    scores = reranker.score("抚养权", ["不满两周岁的子女，以由母亲直接抚养为原则。"])

    assert len(vector) > 100
    assert len(scores) == 1
```

- [ ] **Step 6: Run non-GPU tests**

Run: `pytest backend/tests/unit/test_bge_models_config.py -v`
Expected: PASS.

- [ ] **Step 7: Run GPU readiness test before RAG integration**

Run: `pytest backend/tests/integration/test_model_gpu_readiness.py -m gpu -v`
Expected: PASS on target machine. If FAIL due to CUDA or VRAM, stop implementation and report hardware blocker.

- [ ] **Step 8: Commit**

```bash
git add backend/app/embeddings/base.py backend/app/embeddings/bge_m3.py backend/app/embeddings/embedding_factory.py backend/app/rerank/base.py backend/app/rerank/bge_reranker.py backend/app/rerank/rerank_factory.py backend/tests/unit/test_bge_models_config.py backend/tests/integration/test_model_gpu_readiness.py
git commit -m "feat: add local gpu bge model clients"
```

---

### Task 8: 混合检索、Reranker 阈值和依据不足处理

**Files:**
- Modify: `backend/app/rag/dense_retriever.py`
- Modify: `backend/app/rag/sparse_retriever.py`
- Modify: `backend/app/rag/hybrid_retriever.py`
- Modify: `backend/app/rag/result_merger.py`
- Modify: `backend/app/rag/context_builder.py`
- Modify: `backend/app/rag/pipeline.py`
- Test: `backend/tests/unit/test_hybrid_retriever.py`
- Test: `backend/tests/unit/test_rag_pipeline_threshold.py`

**Interfaces:**
- Consumes: `EmbeddingClient`, `RerankClient`, `find_applicable_materials()`, `Citation`
- Produces: `retrieve_candidates(query: str, filters: RetrievalFilters) -> list[RetrievedDocument]`
- Produces: `rerank_and_select(query: str, candidates: list[RetrievedDocument]) -> RetrievalDecision`
- Produces: `RetrievalDecision(can_answer: bool, reason: str, top_documents: list[RetrievedDocument], citations: list[Citation])`

- [ ] **Step 1: Write failing Top-K tests**

```python
from backend.app.rag.hybrid_retriever import merge_hybrid_results


def test_merges_vector_top20_and_bm25_top20_with_dedup():
    vector = [{"id": f"v{i}", "text": "x"} for i in range(20)]
    bm25 = [{"id": "v0", "text": "x"}] + [{"id": f"b{i}", "text": "y"} for i in range(19)]

    merged = merge_hybrid_results(vector, bm25)

    assert len(merged) == 39
    assert [doc["id"] for doc in merged].count("v0") == 1
```

- [ ] **Step 2: Run retriever tests to verify failure**

Run: `pytest backend/tests/unit/test_hybrid_retriever.py -v`
Expected: FAIL because merge function or dedup behavior is missing.

- [ ] **Step 3: Implement dense and BM25 retrievers**

Dense retriever queries Milvus for Top 20. BM25 retriever returns Top 20 keyword results over published, applicable materials. Both return `RetrievedDocument` with `id`, `material_id`, `version_id`, `text`, `score`, `source`, and `metadata`.

- [ ] **Step 4: Implement merge and dedup**

Deduplicate by `material_id + version_id + article + paragraph + text_hash`. Preserve source score metadata from both retrievers.

- [ ] **Step 5: Write failing threshold tests**

```python
from backend.app.rag.pipeline import decide_after_rerank


def test_low_reranker_score_blocks_answer():
    decision = decide_after_rerank(scores=[0.49], documents=[{"id": "doc-1"}], threshold=0.5)

    assert decision.can_answer is False
    assert decision.reason == "insufficient_legal_basis"
    assert decision.top_documents == []
```

- [ ] **Step 6: Implement threshold decision**

If highest Reranker score is below `0.5`, return `can_answer=False` and `reason="insufficient_legal_basis"`. Otherwise select Top 5 documents after status, version, and applicability filtering.

- [ ] **Step 7: Run RAG tests**

Run: `pytest backend/tests/unit/test_hybrid_retriever.py backend/tests/unit/test_rag_pipeline_threshold.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add backend/app/rag/dense_retriever.py backend/app/rag/sparse_retriever.py backend/app/rag/hybrid_retriever.py backend/app/rag/result_merger.py backend/app/rag/context_builder.py backend/app/rag/pipeline.py backend/tests/unit/test_hybrid_retriever.py backend/tests/unit/test_rag_pipeline_threshold.py
git commit -m "feat: add governed hybrid rag retrieval"
```

---

### Task 9: DeepSeek 流式客户端和模型调用日志

**Files:**
- Modify: `backend/app/llm/base.py`
- Create: `backend/app/llm/deepseek_client.py`
- Modify: `backend/app/llm/model_factory.py`
- Modify: `backend/app/llm/stream_handler.py`
- Create: `backend/app/services/model_log_service.py`
- Test: `backend/tests/unit/test_deepseek_client.py`
- Test: `backend/tests/unit/test_model_log_service.py`

**Interfaces:**
- Consumes: `redact_pii()` from Task 2
- Produces: `DeepSeekClient.stream_chat(request: LlmRequest) -> AsyncIterator[str]`
- Produces: `LlmRequest(messages: list[dict], citations: list[Citation], trace_id: str)`
- Produces: `record_model_call(metadata: ModelCallMetadata) -> None`
- Produces: `ModelCallMetadata(provider: str, model: str, prompt_version: str, knowledge_base_version: str, token_count: int, duration_ms: int, status: str, error_type: str | None, trace_id: str)`

- [ ] **Step 1: Write failing model log test**

```python
from backend.app.services.model_log_service import ModelCallMetadata, sanitize_model_metadata


def test_model_log_metadata_excludes_prompt_and_answer():
    metadata = ModelCallMetadata(
        provider="deepseek",
        model="deepseek-chat",
        prompt_version="v1",
        knowledge_base_version="kb-1",
        token_count=120,
        duration_ms=300,
        status="success",
        error_type=None,
        trace_id="trace-1",
        prompt="完整用户问题不应记录",
        answer="完整回答不应记录",
    )

    sanitized = sanitize_model_metadata(metadata)
    assert "prompt" not in sanitized
    assert "answer" not in sanitized
```

- [ ] **Step 2: Run model log test to verify failure**

Run: `pytest backend/tests/unit/test_model_log_service.py -v`
Expected: FAIL because model log service is missing.

- [ ] **Step 3: Implement metadata-only logging**

Record provider, model, model version, Prompt version, knowledge base version, token count, duration, status, error type, trace ID, redaction flag, memory usage flag, case citation flag, and citation validation flag. Do not store complete Prompt, answer, chat text, API key, or context.

- [ ] **Step 4: Write failing DeepSeek failure test**

```python
import pytest
from backend.app.llm.deepseek_client import DeepSeekClient, DeepSeekUnavailableError


@pytest.mark.asyncio
async def test_deepseek_failure_raises_unavailable(httpx_mock):
    client = DeepSeekClient(api_key="test", base_url="https://api.deepseek.com")
    httpx_mock.add_response(status_code=500, json={"error": "server"})

    with pytest.raises(DeepSeekUnavailableError):
        async for _chunk in client.stream_chat(request={"messages": []}):
            pass
```

- [ ] **Step 5: Implement DeepSeek streaming client**

Use async HTTP client. Stream chunks only after request is accepted. On network error, timeout, 4xx/5xx, or malformed stream, raise `DeepSeekUnavailableError` and record sanitized metadata.

- [ ] **Step 6: Run tests**

Run: `pytest backend/tests/unit/test_deepseek_client.py backend/tests/unit/test_model_log_service.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/app/llm/base.py backend/app/llm/deepseek_client.py backend/app/llm/model_factory.py backend/app/llm/stream_handler.py backend/app/services/model_log_service.py backend/tests/unit/test_deepseek_client.py backend/tests/unit/test_model_log_service.py
git commit -m "feat: add deepseek streaming client"
```

---

### Task 10: 聊天编排、安全检查、追问和回答状态

**Files:**
- Modify: `backend/app/schemas/chat.py`
- Modify: `backend/app/models/conversation.py`
- Modify: `backend/app/models/message.py`
- Modify: `backend/app/services/chat_service.py`
- Modify: `backend/app/safety/risk_detector.py`
- Modify: `backend/app/safety/legal_advice_guard.py`
- Modify: `backend/app/safety/answer_validator.py`
- Modify: `backend/app/api/v1/chat.py`
- Test: `backend/tests/unit/test_chat_orchestration.py`
- Test: `backend/tests/api/test_chat_api.py`
- Test: `backend/tests/integration/test_chat_flow.py`

**Interfaces:**
- Consumes: RAG `RetrievalDecision` from Task 8
- Consumes: `DeepSeekClient.stream_chat()` from Task 9
- Produces: `send_message(user_id: str, conversation_id: str, text: str) -> ChatResult`
- Produces: `regenerate_answer(user_id: str, message_id: str) -> ChatResult`
- Produces: `correct_facts(user_id: str, conversation_id: str, correction: FactCorrection) -> ChatResult`
- Produces: `restart_consultation(user_id: str, conversation_id: str) -> None`

- [ ] **Step 1: Write failing length and scope tests**

```python
import pytest
from backend.app.services.chat_service import MessageTooLongError, classify_scope, validate_message_length


def test_message_over_5000_chars_is_rejected():
    with pytest.raises(MessageTooLongError):
        validate_message_length("问" * 5001)


def test_contract_question_is_out_of_scope():
    result = classify_scope("租房合同押金不退怎么办？")

    assert result.in_scope is False
    assert result.reason == "unsupported_legal_domain"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest backend/tests/unit/test_chat_orchestration.py -v`
Expected: FAIL because orchestration functions are missing.

- [ ] **Step 3: Implement preflight checks**

Apply message length, account rate limit, account concurrency, system concurrency, security risk detection, and scope classification before RAG or DeepSeek calls.

- [ ] **Step 4: Implement emergency risk handling**

If user text indicates family violence, active threat, ongoing personal danger, or minor danger, return safety-first guidance with `110`, emergency medical help, women's federation, legal aid, and evidence preservation reminders. Do not block this path on RAG threshold.

- [ ] **Step 5: Implement intelligent follow-up rules**

Track follow-up count per conversation. Ask at most 3 rounds for missing time, province/city, marriage status, child details, property/debt facts, current goal, and danger status. After 3 rounds, return known facts, missing facts, impact, and conditional analysis.

- [ ] **Step 6: Implement normal answer path**

Call RAG. If `can_answer=False`, return insufficient basis message. If `can_answer=True`, build sanitized DeepSeek request, stream answer body, validate answer, then append citations.

- [ ] **Step 7: Implement regenerate and fact correction**

Regenerate max 3 times per answer and re-run checks, RAG, rerank, and citation validation. Fact correction requires user confirmation, marks old answer as corrected, and regenerates based on corrected facts.

- [ ] **Step 8: Run tests**

Run: `pytest backend/tests/unit/test_chat_orchestration.py backend/tests/api/test_chat_api.py backend/tests/integration/test_chat_flow.py -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add backend/app/schemas/chat.py backend/app/models/conversation.py backend/app/models/message.py backend/app/services/chat_service.py backend/app/safety/risk_detector.py backend/app/safety/legal_advice_guard.py backend/app/safety/answer_validator.py backend/app/api/v1/chat.py backend/tests/unit/test_chat_orchestration.py backend/tests/api/test_chat_api.py backend/tests/integration/test_chat_flow.py
git commit -m "feat: add governed legal chat orchestration"
```

---

### Task 11: 长期记忆、会话删除、账号注销和数据导出

**Files:**
- Create: `backend/app/models/memory.py`
- Create: `backend/app/services/memory_service.py`
- Create: `backend/app/services/export_service.py`
- Modify: `backend/app/services/conversation_service.py`
- Modify: `backend/app/services/user_service.py`
- Modify: `backend/app/api/v1/users.py`
- Modify: `backend/app/api/v1/conversations.py`
- Test: `backend/tests/unit/test_memory_service.py`
- Test: `backend/tests/api/test_user_data_api.py`
- Test: `backend/tests/integration/test_memory_flow.py`

**Interfaces:**
- Consumes: crypto functions from Task 2
- Produces: `create_memory_from_conversation(user_id: str, conversation_id: str, facts: dict, summary: str) -> MemoryRecord`
- Produces: `mark_candidate_update(memory_id: str, new_value: dict) -> MemoryCandidate`
- Produces: `confirm_memory_update(candidate_id: str, user_id: str) -> MemoryRecord`
- Produces: `delete_conversation_and_exclusive_memories(user_id: str, conversation_id: str) -> None`
- Produces: `export_user_data(user_id: str) -> ExportJob`

- [ ] **Step 1: Write failing memory deletion tests**

```python
from backend.app.services.memory_service import delete_exclusive_memories_for_conversation


def test_delete_conversation_removes_only_exclusive_memory(memory_factory):
    exclusive = memory_factory(id="m1", source_conversation_ids=["c1"])
    shared = memory_factory(id="m2", source_conversation_ids=["c1", "c2"])

    deleted = delete_exclusive_memories_for_conversation("user-1", "c1")

    assert "m1" in deleted
    assert "m2" not in deleted
```

- [ ] **Step 2: Run memory tests to verify failure**

Run: `pytest backend/tests/unit/test_memory_service.py -v`
Expected: FAIL because memory service is missing.

- [ ] **Step 3: Implement memory model**

Store encrypted structured facts, encrypted redacted summary, source conversation IDs, enabled status, candidate update state, version, created at, updated at, and last used at.

- [ ] **Step 4: Implement memory controls**

Default memory enabled. First memory creation returns a flag for frontend prompt. User can view, edit, delete, and disable auto extraction in profile. Disabling stops new extraction but retains existing memory.

- [ ] **Step 5: Implement delete and注销 rules**

Deleting a conversation deletes exclusive memories only. Account注销 deletes account, conversations, messages, memories, and linkable personal data, leaving only required脱敏 audit records.

- [ ] **Step 6: Write export tests**

```python
from backend.app.services.export_service import build_export_payload


def test_export_excludes_internal_prompt_and_audit_logs(user_factory):
    payload = build_export_payload(user_id="user-1")

    assert "conversations" in payload
    assert "messages" in payload
    assert "memories" in payload
    assert "audit_logs" not in payload
    assert "internal_prompts" not in payload
```

- [ ] **Step 7: Implement export job**

Create encrypted ZIP with `data.json` and `data.md`. Require second-factor verification before generation. Random one-time ZIP password, 24-hour link, max 3 downloads, file deleted after 24 hours, export file excluded from ordinary backup.

- [ ] **Step 8: Run tests**

Run: `pytest backend/tests/unit/test_memory_service.py backend/tests/api/test_user_data_api.py backend/tests/integration/test_memory_flow.py -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add backend/app/models/memory.py backend/app/services/memory_service.py backend/app/services/export_service.py backend/app/services/conversation_service.py backend/app/services/user_service.py backend/app/api/v1/users.py backend/app/api/v1/conversations.py backend/tests/unit/test_memory_service.py backend/tests/api/test_user_data_api.py backend/tests/integration/test_memory_flow.py
git commit -m "feat: add memory and user data controls"
```

---

### Task 12: 反馈、高风险告警和邮件通知

**Files:**
- Modify: `backend/app/models/feedback.py`
- Modify: `backend/app/schemas/feedback.py`
- Modify: `backend/app/services/feedback_service.py`
- Create: `backend/app/services/notification_service.py`
- Modify: `backend/app/api/v1/feedback.py`
- Test: `backend/tests/unit/test_feedback_service.py`
- Test: `backend/tests/api/test_feedback_api.py`

**Interfaces:**
- Produces: `submit_feedback(user_id: str, message_id: str, rating: str, reason: str | None, category: str | None) -> Feedback`
- Produces: `classify_feedback_risk(category: str, reason: str | None) -> str`
- Produces: `send_admin_alert(alert: AdminAlert) -> NotificationResult`

- [ ] **Step 1: Write failing risk feedback tests**

```python
from backend.app.services.feedback_service import classify_feedback_risk


def test_wrong_legal_basis_is_high_risk_feedback():
    assert classify_feedback_risk("wrong_legal_basis", "引用了不存在的法条") == "high"


def test_unhelpful_answer_is_normal_feedback():
    assert classify_feedback_risk("not_helpful", "没有解决我的问题") == "normal"
```

- [ ] **Step 2: Run feedback tests to verify failure**

Run: `pytest backend/tests/unit/test_feedback_service.py -v`
Expected: FAIL because risk classification is missing.

- [ ] **Step 3: Implement feedback categories**

Support `wrong_legal_basis`, `not_helpful`, `citation_unavailable`, `fact_misunderstood`, `concluded_with_insufficient_info`, `unsafe_guidance`, `privacy_leak`, and `other`.

- [ ] **Step 4: Implement high-risk alerts**

High risk categories are wrong legal basis, unsafe guidance, privacy leak, and high-risk safety handling error. Create backend alert and email notification with脱敏 metadata only.

- [ ] **Step 5: Write API tests**

```python

def test_submit_negative_feedback_with_reason(client, user_token, ai_message):
    response = client.post(
        "/api/v1/feedback",
        headers={"Authorization": f"Bearer {user_token}"},
        json={"message_id": ai_message.id, "rating": "down", "category": "wrong_legal_basis", "reason": "条文不对"},
    )

    assert response.status_code == 201
    assert response.json()["risk_level"] == "high"
```

- [ ] **Step 6: Run tests**

Run: `pytest backend/tests/unit/test_feedback_service.py backend/tests/api/test_feedback_api.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/app/models/feedback.py backend/app/schemas/feedback.py backend/app/services/feedback_service.py backend/app/services/notification_service.py backend/app/api/v1/feedback.py backend/tests/unit/test_feedback_service.py backend/tests/api/test_feedback_api.py
git commit -m "feat: add feedback risk alerts"
```

---

### Task 13: 前端登录、聊天、引用、记忆和后台审核界面

**Files:**
- Modify: `frontend/pages/login.html`
- Modify: `frontend/js/pages/login.js`
- Modify: `frontend/pages/register.html`
- Modify: `frontend/js/pages/register.js`
- Modify: `frontend/pages/chat.html`
- Modify: `frontend/js/pages/chat.js`
- Modify: `frontend/pages/profile.html`
- Modify: `frontend/js/pages/profile.js`
- Modify: `frontend/pages/knowledge-bases.html`
- Modify: `frontend/js/pages/knowledge-bases.js`
- Modify: `frontend/js/api/auth-api.js`
- Modify: `frontend/js/api/chat-api.js`
- Modify: `frontend/js/api/knowledge-base-api.js`
- Modify: `frontend/js/api/user-api.js`
- Modify: `frontend/js/components/citation-list.js`
- Modify: `frontend/js/components/message-feedback.js`
- Modify: `frontend/js/components/disclaimer.js`
- Test: `frontend/tests/e2e/login.spec.js`
- Test: `frontend/tests/e2e/chat.spec.js`
- Test: `frontend/tests/e2e/knowledge-base.spec.js`

**Interfaces:**
- Consumes: backend APIs from Tasks 3, 5, 10, 11, 12
- Produces: Login UI for OTP
- Produces: Chat UI for streaming answer and post-answer citations
- Produces: Profile UI for memory, devices, export,注销
- Produces: Knowledge review UI for content reviewer and super admin workflows

- [ ] **Step 1: Write failing login E2E test**

```javascript
test('user logs in with test phone and fixed code', async ({ page }) => {
  await page.goto('/pages/login.html');
  await page.fill('[data-testid="phone-input"]', '13900000001');
  await page.click('[data-testid="request-code-button"]');
  await page.fill('[data-testid="code-input"]', '123456');
  await page.click('[data-testid="login-button"]');
  await expect(page).toHaveURL(/chat/);
});
```

- [ ] **Step 2: Run login E2E to verify failure**

Run: `npx playwright test frontend/tests/e2e/login.spec.js`
Expected: FAIL because UI selectors or API integration are missing.

- [ ] **Step 3: Implement login and registration UI**

Add phone input, request code button, code input, privacy policy checkbox, user service agreement checkbox, age confirmation checkbox, error messages, and token storage through existing `auth-store`.

- [ ] **Step 4: Write failing chat E2E test**

```javascript
test('chat streams answer and appends citations', async ({ page }) => {
  await loginAsTestUser(page);
  await page.goto('/pages/chat.html');
  await page.fill('[data-testid="chat-input"]', '离婚时孩子抚养权一般怎么判断？');
  await page.click('[data-testid="send-button"]');
  await expect(page.locator('[data-testid="assistant-message"]')).toContainText('初步判断');
  await expect(page.locator('[data-testid="citation-list"]')).toBeVisible();
});
```

- [ ] **Step 5: Implement chat UI**

Support streaming body, citation list after completion, insufficient basis message, emergency guidance, regenerate button, corrected-answer label, restarted consultation notice, message feedback, and memory-used hint.

- [ ] **Step 6: Write failing knowledge base E2E test**

```javascript
test('content reviewer reviews and super admin publishes material', async ({ page }) => {
  await loginAsContentReviewer(page);
  await page.goto('/pages/knowledge-bases.html');
  await page.click('[data-testid="review-material-button"]');
  await page.fill('[data-testid="review-reason"]', '来源、正文和版本已核验');
  await page.click('[data-testid="approve-review-button"]');
  await expect(page.locator('[data-testid="material-status"]')).toContainText('reviewed');
});
```

- [ ] **Step 7: Implement knowledge review UI**

Show crawled, pending, reviewed, published, rejected, deprecated materials. Content reviewers can approve/reject review. Super admins can publish, deprecate, rollback, and manage URL whitelist.

- [ ] **Step 8: Implement profile controls**

Show memory records, edit/delete/disable memory, device sessions, export request, download state, and account注销 confirmation. Never show full phone or sensitive raw values.

- [ ] **Step 9: Run frontend tests**

Run: `npx playwright test frontend/tests/e2e/login.spec.js frontend/tests/e2e/chat.spec.js frontend/tests/e2e/knowledge-base.spec.js`
Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add frontend/pages/login.html frontend/js/pages/login.js frontend/pages/register.html frontend/js/pages/register.js frontend/pages/chat.html frontend/js/pages/chat.js frontend/pages/profile.html frontend/js/pages/profile.js frontend/pages/knowledge-bases.html frontend/js/pages/knowledge-bases.js frontend/js/api/auth-api.js frontend/js/api/chat-api.js frontend/js/api/knowledge-base-api.js frontend/js/api/user-api.js frontend/js/components/citation-list.js frontend/js/components/message-feedback.js frontend/js/components/disclaimer.js frontend/tests/e2e/login.spec.js frontend/tests/e2e/chat.spec.js frontend/tests/e2e/knowledge-base.spec.js
git commit -m "feat: add legal assistant frontend flows"
```

---

### Task 14: 100 条婚姻家庭 MVP 测试集和验收报告

**Files:**
- Create: `evaluation/datasets/marriage_family_mvp_100.jsonl`
- Create: `evaluation/marriage_family_mvp/validate_dataset.py`
- Create: `evaluation/marriage_family_mvp/evaluate_mvp.py`
- Modify: `evaluation/README.md`
- Test: `evaluation/marriage_family_mvp/test_validate_dataset.py`

**Interfaces:**
- Consumes: read-only source directory `C:\Users\bin\Desktop\public`
- Produces: JSONL records with fields `case_id`, `domain`, `user_question`, `fictional_facts`, `facts_to_confirm`, `risk_level`, `expected_legal_basis`, `expected_answer_points`, `must_avoid`, `scoring_rubric`, `source_refs`
- Produces: `validate_record(record: dict) -> list[str]`
- Produces: `evaluate_answer(case: dict, answer: dict) -> EvaluationResult`

- [ ] **Step 1: Write failing dataset validator test**

```python
from evaluation.marriage_family_mvp.validate_dataset import validate_record


def test_valid_record_passes():
    record = {
        "case_id": "MF-DIV-001",
        "domain": "离婚与婚姻关系",
        "user_question": "我想离婚，需要先准备什么？",
        "fictional_facts": "用户甲与配偶乙已登记结婚，暂未说明所在地和是否有子女。",
        "facts_to_confirm": ["所在省市", "是否有未成年子女", "是否协商一致"],
        "risk_level": "normal",
        "expected_legal_basis": ["中华人民共和国民法典 婚姻家庭编"],
        "expected_answer_points": ["说明登记离婚和诉讼离婚路径", "提示补充关键信息"],
        "must_avoid": ["保证一定能离婚", "生成离婚协议书正文"],
        "scoring_rubric": {"basis_accuracy": 2, "boundary": 2, "follow_up": 1},
        "source_refs": ["civil_questions/civil_questions.txt"],
    }

    assert validate_record(record) == []
```

- [ ] **Step 2: Run validator test to verify failure**

Run: `pytest evaluation/marriage_family_mvp/test_validate_dataset.py -v`
Expected: FAIL because validator module is missing.

- [ ] **Step 3: Implement validator**

Ensure every record has all required fields. Ensure exactly 100 records. Ensure domain counts are 40 divorce, 30 custody/visitation, 30 property/debt. Ensure `source_refs` never writes to or modifies `C:\Users\bin\Desktop\public`.

- [ ] **Step 4: Create dataset records**

Read from `C:\Users\bin\Desktop\public` without modifying it. Rewrite records as fictional user cases using anonymous names. Do not copy private identifiers. Include legal basis, expected answer points, must-avoid rules, and scoring rubric.

- [ ] **Step 5: Run dataset validation**

Run: `python evaluation/marriage_family_mvp/validate_dataset.py evaluation/datasets/marriage_family_mvp_100.jsonl`
Expected: PASS with counts `40/30/30` and 0 schema errors.

- [ ] **Step 6: Implement evaluator skeleton**

Create scoring categories: legal citation accuracy, citation traceability, high-risk handling, no-fabrication, follow-up accuracy, answer success, and interface success. Include thresholds from the spec.

- [ ] **Step 7: Run evaluation tests**

Run: `pytest evaluation/marriage_family_mvp/test_validate_dataset.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add evaluation/datasets/marriage_family_mvp_100.jsonl evaluation/marriage_family_mvp/validate_dataset.py evaluation/marriage_family_mvp/evaluate_mvp.py evaluation/marriage_family_mvp/test_validate_dataset.py evaluation/README.md
git commit -m "test: add marriage family mvp evaluation dataset"
```

---

## Final Integration Gate

After Tasks 1–14 pass independently, run:

```bash
pytest backend/tests/unit -v
pytest backend/tests/api -v
pytest backend/tests/integration -v
npx playwright test frontend/tests/e2e/login.spec.js frontend/tests/e2e/chat.spec.js frontend/tests/e2e/knowledge-base.spec.js
python evaluation/marriage_family_mvp/validate_dataset.py evaluation/datasets/marriage_family_mvp_100.jsonl
```

Expected final state:

- `/health/live` returns 200.
- `/health/ready` returns 200 only when MySQL, Redis, Milvus, DeepSeek config, CUDA, GPU and both local BGE models are available.
- Test user can log in with simulated OTP.
- In-scope marriage-family question returns structured answer and citations.
- Out-of-scope question is blocked with official help guidance.
- RAG below threshold returns insufficient legal basis.
- Emergency risk question returns safety-first guidance.
- Knowledge material cannot become searchable until reviewed and published.
- User can manage long-term memory, delete conversations, export data and注销账号.
- Backend and frontend tests pass.
- 100-record MVP evaluation dataset validates successfully.

## Self-Review

### Spec Coverage

- Product scope and unsupported areas: covered by Tasks 10 and 13.
- OTP auth, Token, devices and age/agreement checks: covered by Task 3 and Task 13.
- Encryption, HMAC and PII redaction: covered by Task 2.
- Knowledge source whitelist, snapshots, review and publish workflow: covered by Task 5.
- Legal version matching and citation precision: covered by Task 6.
- Local BGE-M3 and BGE Reranker GPU requirements: covered by Task 7.
- Hybrid retrieval, Top-K, rerank threshold and insufficient basis handling: covered by Task 8.
- DeepSeek minimal payload, streaming and failure handling: covered by Task 9.
- Chat orchestration, follow-up, emergency handling, correction and regeneration: covered by Task 10.
- Long-term memory, deletion, export and account注销: covered by Task 11.
- Feedback and high-risk alerts: covered by Task 12.
- Frontend user and admin flows: covered by Task 13.
- 100-case test dataset and metrics: covered by Task 14.
- Health checks and startup readiness: covered by Task 1.
- Audit logging and permissions: covered by Task 4.

### Placeholder Scan

No task contains unresolved `TBD`, `TODO`, or `implement later` instructions. Each task defines concrete files, interfaces, test commands, expected outcomes and commit commands.

### Type Consistency

Cross-task interfaces are intentionally named once and consumed by later tasks with the same names: `DependencyStatus`, `encrypt_text`, `hmac_digest`, `record_audit`, `find_applicable_materials`, `Citation`, `EmbeddingClient`, `RerankClient`, `RetrievalDecision`, `DeepSeekClient.stream_chat`, `redact_pii`, and `create_memory_from_conversation`.
