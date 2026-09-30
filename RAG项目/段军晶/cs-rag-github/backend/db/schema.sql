-- ============================================================================
--  计算机专业知识库 RAG 问答助手 —— MySQL 表结构
-- ----------------------------------------------------------------------------
--  设计说明：
--    MySQL 是「页码溯源」的唯一真相源。向量库只负责找出「哪些块相关」，
--    而「这块是什么、来自哪一页」一律以本库为准。这样即使向量库重建，
--    溯源信息也不会丢失。
--
--  执行方式：
--    mysql -u root -p < backend/db/schema.sql
--    或由 backend/db/mysql.py 中的 init_schema() 自动执行（幂等）
-- ============================================================================


-- ==================== 表 1：知识库文档元信息 ====================
CREATE TABLE IF NOT EXISTS `kb_documents` (
  `doc_id`       VARCHAR(64)  NOT NULL                COMMENT '文档ID，由文件内容哈希生成，内容不变则ID不变',
  `file_name`    VARCHAR(255) NOT NULL                COMMENT '源文件名（前端来源展示用）',
  `file_path`    VARCHAR(512) NOT NULL DEFAULT ''     COMMENT '源文件绝对路径',
  `file_hash`    CHAR(64)     NOT NULL                COMMENT '文件SHA256，用于重复上传检测',
  `page_count`   INT          NOT NULL DEFAULT 0      COMMENT '文档总页数',
  `chunk_count`  INT          NOT NULL DEFAULT 0      COMMENT '切分块数',
  `parse_engine` VARCHAR(32)  NOT NULL DEFAULT ''     COMMENT '实际使用的解析引擎：mineru / fallback',
  `status`       VARCHAR(16)  NOT NULL DEFAULT 'pending'
                 COMMENT '处理状态：pending待处理 / parsed已解析 / indexed已入库 / failed失败',
  `error_msg`    VARCHAR(512) NOT NULL DEFAULT ''     COMMENT '失败原因（status=failed 时填写）',
  `created_at`   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '上传时间',
  `updated_at`   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
                 ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`doc_id`),
  UNIQUE KEY `uk_file_hash` (`file_hash`),
  KEY `idx_status` (`status`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='知识库文档元信息';


-- ==================== 表 2：知识库分块元数据（页码溯源核心表） ====================
-- 重要：page_no 与 page_nums 为双字段设计
--   page_no   —— 主页码，取该块覆盖的首个内容块页码，用于前端展示
--   page_nums —— 完整页码列表（JSON 数组），跨页块会包含多个页码
-- 只存单个页码会导致用户翻到该页却找不到跨页内容的后半段，溯源精度受损。
CREATE TABLE IF NOT EXISTS `kb_chunks` (
  `chunk_id`      VARCHAR(64)  NOT NULL               COMMENT '块ID',
  `doc_id`        VARCHAR(64)  NOT NULL               COMMENT '所属文档ID',
  `chunk_index`   INT          NOT NULL               COMMENT '块在文档内的序号，从0开始',
  `page_no`       INT          NOT NULL               COMMENT '主页码（1基），前端展示用',
  `page_nums`     JSON         NOT NULL               COMMENT '完整页码列表，如 [3,4]；页码溯源的完整信息',
  `content`       MEDIUMTEXT   NOT NULL               COMMENT '块正文',
  `content_type`  VARCHAR(16)  NOT NULL DEFAULT 'text'
                  COMMENT '块类型：text正文 / table表格 / image图片',
  `section_title` VARCHAR(255) NOT NULL DEFAULT ''    COMMENT '所属章节标题，提升可读性',
  `source_file`   VARCHAR(255) NOT NULL DEFAULT ''    COMMENT '源文件名（冗余字段，展示时无需联表）',
  `char_count`    INT          NOT NULL DEFAULT 0     COMMENT '块字符数，便于统计与排查',
  `created_at`    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '入库时间',
  PRIMARY KEY (`chunk_id`),
  KEY `idx_doc` (`doc_id`),
  KEY `idx_doc_page` (`doc_id`, `page_no`),
  KEY `idx_type` (`content_type`),
  CONSTRAINT `fk_chunk_doc` FOREIGN KEY (`doc_id`)
    REFERENCES `kb_documents` (`doc_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='知识库分块元数据（页码溯源真相源）';


-- ==================== 表 3：问答记录 ====================
-- 说明：本表不对前端提供任何查询接口，仅用于后端排查与离线分析
CREATE TABLE IF NOT EXISTS `qa_records` (
  `id`         BIGINT       NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `session_id` VARCHAR(64)  NOT NULL DEFAULT ''     COMMENT '会话ID',
  `question`   TEXT         NOT NULL                COMMENT '用户问题',
  `answer`     MEDIUMTEXT                           COMMENT '生成的答案',
  `sources`    JSON                                 COMMENT '溯源列表 [{file_name,page_no,chunk_id,score}]',
  `retrieved`  JSON                                 COMMENT '检索命中明细（Trace，仅后端留存）',
  `pipeline`   VARCHAR(16)  NOT NULL DEFAULT 'v1'   COMMENT '使用的链路版本：v1/v2/v3',
  `latency_ms` INT          NOT NULL DEFAULT 0      COMMENT '端到端耗时（毫秒）',
  `created_at` DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '记录时间',
  PRIMARY KEY (`id`),
  KEY `idx_session` (`session_id`),
  KEY `idx_created` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='问答记录（仅后端排查用，不对外提供查询接口）';


-- ============================================================================
--  增量表（前端改造配套：用户登录 / 历史记录 / 收藏）
--  说明：本节只「新增表」，不改动 kb_documents / kb_chunks / qa_records 三张既有表，
--        也不影响 PDF 解析、分块、向量化、检索、重排、生成的任何逻辑。
--        全部为 CREATE TABLE IF NOT EXISTS，init_schema() 幂等执行，重启自动建表。
-- ============================================================================


-- ==================== 表 4：用户账号 ====================
CREATE TABLE IF NOT EXISTS `users` (
  `user_id`       VARCHAR(32)  NOT NULL               COMMENT '用户ID，服务端生成的稳定标识',
  `username`      VARCHAR(64)  NOT NULL               COMMENT '登录账号，唯一；输入即登录，不存在则自动创建',
  `display_name`  VARCHAR(64)  NOT NULL DEFAULT ''    COMMENT '界面展示名，默认取账号',
  `avatar`        VARCHAR(255) NOT NULL DEFAULT ''    COMMENT '头像：emoji 标识，或自定义头像地址（/avatars/xxx.jpg）',
  `created_at`    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '首次登录时间',
  `last_login_at` DATETIME     NULL                   COMMENT '最近一次登录时间',
  PRIMARY KEY (`user_id`),
  UNIQUE KEY `uk_username` (`username`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='用户账号（课程演示口径：无密码、无加密、账号即身份）';


-- ==================== 表 5：对话会话（会话绑定登录用户） ====================
CREATE TABLE IF NOT EXISTS `chat_sessions` (
  `session_id`    VARCHAR(64)  NOT NULL               COMMENT '会话ID，与 Redis 多轮上下文共用同一 ID',
  `user_id`       VARCHAR(32)  NOT NULL               COMMENT '归属用户ID，历史记录按此隔离',
  `title`         VARCHAR(64)  NOT NULL DEFAULT ''    COMMENT '会话标题，取首个提问的前 12 字',
  `role`          VARCHAR(16)  NOT NULL DEFAULT 'student' COMMENT '会话身份：student学生 / workplace职场',
  `message_count` INT          NOT NULL DEFAULT 0     COMMENT '该会话已保存的问答条数',
  `created_at`    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at`    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
                  ON UPDATE CURRENT_TIMESTAMP          COMMENT '最近问答时间，历史列表按此倒序',
  PRIMARY KEY (`session_id`),
  KEY `idx_user_updated` (`user_id`, `updated_at`),
  CONSTRAINT `fk_session_user` FOREIGN KEY (`user_id`)
    REFERENCES `users` (`user_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='对话会话，绑定登录用户（历史记录永不因清缓存丢失）';


-- ==================== 表 6：问答消息（历史记录的持久化真相源） ====================
-- 与 qa_records 的分工：
--   qa_records  —— 后端排查用，不对外提供查询接口（既有契约，保持不变）
--   chat_messages —— 面向用户的历史记录，按 user_id 隔离并对外提供查询接口
-- sources 存「快照」：知识库重建后历史记录里的溯源仍可回看，不会变空壳。
CREATE TABLE IF NOT EXISTS `chat_messages` (
  `id`         BIGINT       NOT NULL AUTO_INCREMENT   COMMENT '自增主键，前端收藏用的 message_id',
  `session_id` VARCHAR(64)  NOT NULL                  COMMENT '所属会话ID',
  `user_id`    VARCHAR(32)  NOT NULL                  COMMENT '归属用户ID',
  `question`   TEXT         NOT NULL                  COMMENT '用户提问',
  `answer`     MEDIUMTEXT                             COMMENT '答案正文',
  `sources`    JSON                                   COMMENT '溯源快照 [{file_name,page_no,page_nums,summary,section_title,content_type}]',
  `role`       VARCHAR(16)  NOT NULL DEFAULT 'student' COMMENT '本条问答使用的身份',
  `latency_ms` INT          NOT NULL DEFAULT 0        COMMENT '端到端耗时（毫秒）',
  `created_at` DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '提问时间',
  PRIMARY KEY (`id`),
  KEY `idx_session_id` (`session_id`, `id`),
  KEY `idx_user_time` (`user_id`, `created_at`),
  CONSTRAINT `fk_message_session` FOREIGN KEY (`session_id`)
    REFERENCES `chat_sessions` (`session_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='问答消息，用户历史记录的持久化来源';


-- ==================== 表 7：用户收藏 ====================
-- 收藏存「问题 + 答案 + 溯源」整条快照，只存 message_id 会在知识库重建后变空壳。
-- 不加指向 chat_messages 的外键：删除历史消息时，用户已收藏的快照应当保留。
CREATE TABLE IF NOT EXISTS `user_favorites` (
  `id`         BIGINT       NOT NULL AUTO_INCREMENT   COMMENT '自增主键',
  `user_id`    VARCHAR(32)  NOT NULL                  COMMENT '收藏归属用户ID',
  `message_id` BIGINT       NULL                      COMMENT '来源问答消息ID，用于判断收藏状态',
  `session_id` VARCHAR(64)  NOT NULL DEFAULT ''       COMMENT '来源会话ID',
  `question`   TEXT         NOT NULL                  COMMENT '问题快照',
  `answer`     MEDIUMTEXT                             COMMENT '答案快照',
  `sources`    JSON                                   COMMENT '溯源快照',
  `role`       VARCHAR(16)  NOT NULL DEFAULT 'student' COMMENT '收藏时使用的身份',
  `created_at` DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '收藏时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_user_message` (`user_id`, `message_id`),
  KEY `idx_user_time` (`user_id`, `created_at`),
  CONSTRAINT `fk_favorite_user` FOREIGN KEY (`user_id`)
    REFERENCES `users` (`user_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='用户收藏（绑定用户，持久化 MySQL，清除浏览器缓存不丢失）';