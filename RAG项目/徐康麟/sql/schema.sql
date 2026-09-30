-- 法律 RAG 业务库表结构（MySQL 8+）
-- 与 legal_rag/business/store.py 中的 SCHEMA_STATEMENTS 保持一致，
-- 供 Navicat / 手工建库使用。
--
-- 用法：
--   CREATE DATABASE IF NOT EXISTS legal_rag DEFAULT CHARACTER SET utf8mb4;
--   USE legal_rag;
--   SOURCE sql/schema.sql;

CREATE TABLE IF NOT EXISTS users (
    user_id     VARCHAR(64)  PRIMARY KEY COMMENT '用户 ID',
    name        VARCHAR(128) DEFAULT '' COMMENT '昵称',
    created_at  DOUBLE       DEFAULT 0  COMMENT '创建时间戳',
    updated_at  DOUBLE       DEFAULT 0  COMMENT '更新时间戳'
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COMMENT = '用户信息表';

CREATE TABLE IF NOT EXISTS roles (
    role_id     VARCHAR(64)  PRIMARY KEY COMMENT '角色 ID',
    name        VARCHAR(128) DEFAULT '' COMMENT '角色名',
    domain      VARCHAR(64)  DEFAULT '' COMMENT '领域：法律/医疗/金融/社交…',
    persona     TEXT                    COMMENT 'system persona 模板',
    updated_at  DOUBLE       DEFAULT 0
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COMMENT = '角色信息表';

CREATE TABLE IF NOT EXISTS sessions (
    session_id  VARCHAR(128) PRIMARY KEY COMMENT '会话 ID',
    user_id     VARCHAR(64)  NOT NULL    COMMENT '所属用户',
    role_id     VARCHAR(64)  NOT NULL    COMMENT '当前角色',
    title       VARCHAR(255) DEFAULT ''  COMMENT '会话标题',
    created_at  DOUBLE       DEFAULT 0,
    updated_at  DOUBLE       DEFAULT 0,
    KEY idx_sessions_user (user_id, updated_at)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COMMENT = '会话列表';

CREATE TABLE IF NOT EXISTS documents (
    doc_id      VARCHAR(128) PRIMARY KEY COMMENT '文档 ID',
    source      VARCHAR(255) DEFAULT ''  COMMENT '来源文件名',
    role_id     VARCHAR(64)  DEFAULT ''  COMMENT '归属角色（分区隔离用）',
    md5         VARCHAR(64)  DEFAULT ''  COMMENT '内容指纹，用于增量去重',
    chunks      INTEGER      DEFAULT 0   COMMENT '切分出的 chunk 数',
    updated_at  DOUBLE       DEFAULT 0,
    KEY idx_documents_role (role_id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COMMENT = '文档管理表';

CREATE TABLE IF NOT EXISTS settings (
    key         VARCHAR(128) PRIMARY KEY COMMENT '配置项',
    value       TEXT                     COMMENT '配置值',
    updated_at  DOUBLE       DEFAULT 0
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COMMENT = '系统配置表';
