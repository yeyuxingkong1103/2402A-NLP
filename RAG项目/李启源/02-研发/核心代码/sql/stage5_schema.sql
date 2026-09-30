-- ============================================
-- 阶段5：多轮对话与记忆系统 - 数据库Schema
-- ============================================

-- 0. 租户表
-- 文件末尾的「插入默认租户」会往 tenants 里写数据，但原来整个 sql/ 目录
-- 没有任何地方建过这张表，导入到那一条 INSERT 就会以
-- ERROR 1146 Table 'tenants' doesn't exist 中断，后面的初始化数据全不执行。
-- 其余表的 tenant_id 只是普通列、没有外键，所以这里按它们的风格补上即可。
CREATE TABLE IF NOT EXISTS tenants (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    tenant_name VARCHAR(200) NOT NULL COMMENT '租户名称',
    tenant_key VARCHAR(100) NOT NULL UNIQUE COMMENT '租户标识（英文唯一键）',
    status ENUM('active', 'inactive', 'suspended') DEFAULT 'active',
    settings JSON COMMENT '租户级配置',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_tenant_key (tenant_key),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='租户';

-- 1. 用户表
CREATE TABLE IF NOT EXISTS users (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    tenant_id BIGINT NOT NULL COMMENT '租户ID',
    user_external_id VARCHAR(255) UNIQUE COMMENT '外部用户ID（对接第三方系统）',
    username VARCHAR(100),
    email VARCHAR(255),
    phone VARCHAR(50),
    status ENUM('active', 'inactive', 'banned') DEFAULT 'active',
    preferences JSON COMMENT '用户偏好配置',
    metadata JSON COMMENT '其他元数据',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_tenant_user (tenant_id, user_external_id),
    INDEX idx_email (email),
    INDEX idx_phone (phone),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户表';

-- 2. 角色配置表
CREATE TABLE IF NOT EXISTS roles (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    tenant_id BIGINT NOT NULL,
    role_name VARCHAR(100) NOT NULL COMMENT '角色名称',
    role_key VARCHAR(50) NOT NULL COMMENT '角色标识: customer_service, sales, tech_support',

    -- 角色人设配置
    persona_name VARCHAR(50) COMMENT '人设名称: 小云、小智、小美',
    persona_description TEXT COMMENT '人设描述',
    system_prompt TEXT COMMENT '系统提示词',
    greeting_message TEXT COMMENT '欢迎语',

    -- 行为配置
    temperature DECIMAL(3,2) DEFAULT 0.7 COMMENT 'LLM温度参数',
    max_tokens INT DEFAULT 1000 COMMENT '最大生成token数',
    top_k INT DEFAULT 5 COMMENT '检索Top-K',
    enable_rerank BOOLEAN DEFAULT TRUE COMMENT '是否启用重排序',
    enable_memory BOOLEAN DEFAULT TRUE COMMENT '是否启用记忆',
    max_memory_turns INT DEFAULT 5 COMMENT '最大记忆轮数',

    -- 业务规则
    allowed_intents JSON COMMENT '允许处理的意图类型',
    restricted_topics JSON COMMENT '禁止讨论的话题',
    escalation_rules JSON COMMENT '转接规则',

    status ENUM('active', 'inactive') DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    UNIQUE KEY uk_tenant_role_key (tenant_id, role_key),
    INDEX idx_tenant (tenant_id),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='角色配置表';

-- 3. 会话表
CREATE TABLE IF NOT EXISTS sessions (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    session_id VARCHAR(100) UNIQUE NOT NULL COMMENT '会话唯一ID',
    tenant_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    role_id BIGINT NOT NULL COMMENT '当前使用的角色',

    -- 会话状态
    status ENUM('active', 'archived', 'expired') DEFAULT 'active',
    message_count INT DEFAULT 0 COMMENT '消息总数',

    -- 时间信息
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_active_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP NULL,

    -- 会话元数据
    channel VARCHAR(50) COMMENT '渠道: web, mobile, api, wechat',
    source_ip VARCHAR(50) COMMENT '来源IP',
    user_agent TEXT COMMENT 'User Agent',
    metadata JSON COMMENT '其他元数据',

    INDEX idx_user_sessions (user_id, started_at),
    INDEX idx_tenant (tenant_id),
    INDEX idx_status (status),
    INDEX idx_last_active (last_active_at),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (role_id) REFERENCES roles(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='会话表';

-- 4. 用户偏好表
CREATE TABLE IF NOT EXISTS user_preferences (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    user_id BIGINT NOT NULL,
    preference_key VARCHAR(100) NOT NULL COMMENT 'language, notification, theme, response_style',
    preference_value TEXT,
    source VARCHAR(50) DEFAULT 'manual' COMMENT '来源: manual, learned, imported',
    confidence DECIMAL(3,2) DEFAULT 1.0 COMMENT '置信度 0-1',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    UNIQUE KEY uk_user_pref (user_id, preference_key),
    INDEX idx_user (user_id),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户偏好表';

-- 5. 对话归档表
CREATE TABLE IF NOT EXISTS conversation_archives (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    session_id VARCHAR(100) NOT NULL,
    user_id BIGINT NOT NULL,
    tenant_id BIGINT NOT NULL,
    role_id BIGINT,

    -- 对话内容
    messages JSON COMMENT '完整对话消息（压缩存储）',
    summary TEXT COMMENT '对话摘要',

    -- 统计信息
    turn_count INT COMMENT '对话轮数',
    total_tokens INT COMMENT '总token数',
    total_duration_seconds INT COMMENT '总耗时（秒）',

    -- 质量评估
    satisfaction_score DECIMAL(3,2) COMMENT '满意度评分 0-5',
    resolved BOOLEAN COMMENT '是否已解决',
    feedback TEXT COMMENT '用户反馈',

    -- 意图分析
    main_intents JSON COMMENT '主要意图列表',
    detected_entities JSON COMMENT '检测到的实体',

    archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_user (user_id, archived_at),
    INDEX idx_session (session_id),
    INDEX idx_tenant (tenant_id),
    INDEX idx_resolved (resolved),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='对话归档表';

-- 6. 角色切换日志
CREATE TABLE IF NOT EXISTS role_switch_logs (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    session_id VARCHAR(100) NOT NULL,
    user_id BIGINT NOT NULL,
    from_role_id BIGINT COMMENT '原角色ID',
    to_role_id BIGINT NOT NULL COMMENT '新角色ID',
    reason VARCHAR(255) COMMENT '切换原因',
    switch_type ENUM('manual', 'auto', 'escalation') DEFAULT 'manual' COMMENT '切换类型',
    switched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_session (session_id),
    INDEX idx_user (user_id),
    INDEX idx_switched_at (switched_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='角色切换日志';

-- 7. 记忆压缩日志
CREATE TABLE IF NOT EXISTS memory_compression_logs (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    session_id VARCHAR(100) NOT NULL,
    compression_type ENUM('partial', 'final') COMMENT 'partial=中间压缩, final=会话结束',

    -- 压缩前后对比
    before_message_count INT COMMENT '压缩前消息数',
    after_message_count INT COMMENT '压缩后消息数',
    before_tokens INT COMMENT '压缩前token数',
    after_tokens INT COMMENT '压缩后token数',

    summary_text TEXT COMMENT '生成的摘要',
    compression_ratio DECIMAL(4,2) COMMENT '压缩比',

    compressed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_session (session_id),
    INDEX idx_compressed_at (compressed_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='记忆压缩日志';

-- ============================================
-- 初始化数据
-- ============================================

-- 插入默认租户
INSERT INTO tenants (id, tenant_name, tenant_key, status) VALUES
(1, '默认租户', 'default', 'active')
ON DUPLICATE KEY UPDATE tenant_name = tenant_name;

-- 插入测试用户
INSERT INTO users (id, tenant_id, user_external_id, username, email, status) VALUES
(1001, 1, 'test_user_001', '测试用户001', 'test001@example.com', 'active'),
(1002, 1, 'test_user_002', '测试用户002', 'test002@example.com', 'active')
ON DUPLICATE KEY UPDATE username = username;

-- 插入角色配置
INSERT INTO roles (
    tenant_id, role_name, role_key,
    persona_name, persona_description, system_prompt, greeting_message,
    temperature, max_tokens, top_k, enable_rerank, enable_memory, max_memory_turns,
    allowed_intents, restricted_topics, escalation_rules, status
) VALUES
-- 客服角色
(
    1, '客服专员', 'customer_service',
    '小云', '热情专业的客服专员，擅长处理退换货、物流、售后等问题',
    '你是一位名叫"小云"的客服专员。你的职责是：
1. 解答用户关于订单、物流、退换货、售后政策的问题
2. 态度热情、耐心、专业，使用"您"称呼用户
3. 仅基于知识库回答，不编造信息
4. 如果知识库没有答案，引导用户联系人工客服
5. 重要信息（订单号、时间、金额）必须准确无误
6. 每个回答都要标注来源 [来源: xxx]',
    '您好！我是客服专员小云，很高兴为您服务。请问有什么可以帮您的吗？',
    0.7, 1000, 5, TRUE, TRUE, 5,
    '["order_inquiry", "refund_inquiry", "shipping_inquiry", "policy_question", "product_question"]',
    '["politics", "religion", "personal_privacy"]',
    '{"no_answer": "转人工客服", "complaint": "转投诉部门", "technical": "转技术支持"}',
    'active'
),
-- 销售角色
(
    1, '销售顾问', 'sales',
    '小美', '亲切专业的销售顾问，擅长产品推荐和购买咨询',
    '你是一位名叫"小美"的销售顾问。你的职责是：
1. 为用户推荐合适的产品
2. 解答产品功能、价格、优惠活动等问题
3. 态度亲切、专业，善于引导用户购买
4. 强调产品优势和性价比
5. 不夸大产品功能，实事求是',
    '您好！我是销售顾问小美，很高兴为您服务。今天想了解什么产品呢？',
    0.8, 1200, 5, TRUE, TRUE, 5,
    '["product_inquiry", "price_inquiry", "promotion_inquiry", "comparison", "recommendation"]',
    '["after_sales", "complaints"]',
    '{"after_sales": "转客服", "technical": "转技术支持"}',
    'active'
),
-- 技术支持角色
(
    1, '技术支持', 'tech_support',
    '小智', '专业的技术支持工程师，擅长解决技术问题',
    '你是一位名叫"小智"的技术支持工程师。你的职责是：
1. 诊断和解决用户的技术问题
2. 提供详细的技术指导和操作步骤
3. 态度专业、严谨，确保技术准确性
4. 使用用户能理解的语言解释技术问题
5. 必要时提供图文教程或视频链接',
    '您好！我是技术支持工程师小智，请详细描述您遇到的技术问题，我会尽力帮您解决。',
    0.6, 1500, 5, TRUE, TRUE, 5,
    '["technical_issue", "troubleshooting", "setup_help", "compatibility", "error_diagnosis"]',
    '["sales", "pricing", "refund"]',
    '{"sales_question": "转销售", "refund": "转客服"}',
    'active'
)
ON DUPLICATE KEY UPDATE role_name = role_name;

-- ============================================
-- 索引优化建议
-- ============================================

-- 如果会话表数据量很大，考虑按时间分区
-- ALTER TABLE sessions PARTITION BY RANGE (UNIX_TIMESTAMP(started_at)) (
--     PARTITION p202609 VALUES LESS THAN (UNIX_TIMESTAMP('2026-10-01')),
--     PARTITION p202610 VALUES LESS THAN (UNIX_TIMESTAMP('2026-11-01')),
--     ...
-- );

-- 如果需要全文检索对话内容
-- ALTER TABLE conversation_archives ADD FULLTEXT INDEX ft_summary (summary);

-- ============================================
-- 查询示例
-- ============================================

-- 查询用户的活跃会话
-- SELECT s.session_id, s.message_count, s.last_active_at, r.persona_name
-- FROM sessions s
-- JOIN roles r ON s.role_id = r.id
-- WHERE s.user_id = 1001 AND s.status = 'active'
-- ORDER BY s.last_active_at DESC
-- LIMIT 10;

-- 查询用户的历史对话摘要
-- SELECT session_id, summary, turn_count, archived_at
-- FROM conversation_archives
-- WHERE user_id = 1001
-- ORDER BY archived_at DESC
-- LIMIT 10;

-- 查询角色切换频率
-- SELECT
--     r1.persona_name AS from_role,
--     r2.persona_name AS to_role,
--     COUNT(*) AS switch_count
-- FROM role_switch_logs l
-- LEFT JOIN roles r1 ON l.from_role_id = r1.id
-- JOIN roles r2 ON l.to_role_id = r2.id
-- WHERE l.switched_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
-- GROUP BY l.from_role_id, l.to_role_id
-- ORDER BY switch_count DESC;
