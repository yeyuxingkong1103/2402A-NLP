-- ============================================================
-- 基于 RAG 的心理医生多角色陪伴系统 —— MySQL 数据库 DDL
-- 数据库：rag_roleplay   字符集：utf8mb4
-- 说明：服务启动时会自动建库建表（SQLAlchemy），
--       唯一事实源是 src/models + alembic 迁移；本文件仅供手工初始化与交付审查（G11）。
-- ============================================================

CREATE DATABASE IF NOT EXISTS rag_roleplay
DEFAULT CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

USE rag_roleplay;

-- ---------- 用户 ----------
CREATE TABLE IF NOT EXISTS users (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  username VARCHAR(64) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  email VARCHAR(128),
  phone VARCHAR(32),
  nickname VARCHAR(64),
  avatar VARCHAR(512),
  status TINYINT DEFAULT 1,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  last_login_at DATETIME,
  INDEX idx_users_email (email),
  INDEX idx_users_phone (phone)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------- 系统角色 ----------
CREATE TABLE IF NOT EXISTS sys_roles (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  role_code VARCHAR(64) UNIQUE NOT NULL,
  role_name VARCHAR(64) NOT NULL,
  description VARCHAR(255)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS user_sys_roles (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  user_id BIGINT NOT NULL,
  role_id BIGINT NOT NULL,
  UNIQUE KEY uk_user_role (user_id, role_id),
  FOREIGN KEY (user_id) REFERENCES users(id),
  FOREIGN KEY (role_id) REFERENCES sys_roles(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------- 心理医生角色 ----------
CREATE TABLE IF NOT EXISTS counselor_personas (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  persona_code VARCHAR(64) UNIQUE NOT NULL,
  name VARCHAR(64) NOT NULL,
  title VARCHAR(128),
  therapy_type VARCHAR(64),
  style VARCHAR(255),
  methods VARCHAR(255),
  greeting TEXT,
  system_prompt TEXT NOT NULL,
  knowledge_scope VARCHAR(255),
  avatar VARCHAR(512),
  safety_boundary TEXT,
  model_params JSON,
  status TINYINT DEFAULT 1,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS user_persona_preferences (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  user_id BIGINT NOT NULL,
  persona_id BIGINT NOT NULL,
  is_default TINYINT DEFAULT 0,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_user_persona (user_id, persona_id),
  FOREIGN KEY (user_id) REFERENCES users(id),
  FOREIGN KEY (persona_id) REFERENCES counselor_personas(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------- 会话与消息 ----------
CREATE TABLE IF NOT EXISTS conversations (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  user_id BIGINT NOT NULL,
  persona_id BIGINT NOT NULL,
  title VARCHAR(255),
  status TINYINT DEFAULT 1,
  message_count INT DEFAULT 0,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX idx_conv_user (user_id),
  INDEX idx_conv_persona (persona_id),
  FOREIGN KEY (user_id) REFERENCES users(id),
  FOREIGN KEY (persona_id) REFERENCES counselor_personas(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS messages (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  conversation_id BIGINT NOT NULL,
  role VARCHAR(32) NOT NULL,
  content TEXT NOT NULL,
  tokens INT,
  refs JSON,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_msg_conv (conversation_id),
  FOREIGN KEY (conversation_id) REFERENCES conversations(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------- 知识库 ----------
CREATE TABLE IF NOT EXISTS knowledge_docs (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  persona_id BIGINT NOT NULL,
  title VARCHAR(255),
  source VARCHAR(512),
  file_path VARCHAR(512),
  file_type VARCHAR(32),
  status VARCHAR(32) DEFAULT 'pending',
  chunk_count INT DEFAULT 0,
  error_msg TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX idx_doc_persona (persona_id),
  FOREIGN KEY (persona_id) REFERENCES counselor_personas(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS knowledge_chunks (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  doc_id BIGINT NOT NULL,
  persona_id BIGINT NOT NULL,
  chunk_id BIGINT NOT NULL,
  content TEXT NOT NULL,
  summary VARCHAR(2048),
  milvus_id BIGINT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_chunk_doc (doc_id),
  INDEX idx_chunk_persona (persona_id),
  FOREIGN KEY (doc_id) REFERENCES knowledge_docs(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------- 日志 ----------
CREATE TABLE IF NOT EXISTS audit_logs (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  user_id BIGINT,
  action VARCHAR(128),
  detail VARCHAR(4000),
  ip VARCHAR(64),
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_audit_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS login_logs (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  user_id BIGINT,
  ip VARCHAR(64),
  user_agent VARCHAR(512),
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_login_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------- 初始化数据：系统角色 ----------
INSERT IGNORE INTO sys_roles (role_code, role_name, description) VALUES
 ('admin', '管理员', '系统管理员，可管理用户、角色、知识库'),
 ('user', '普通用户', '普通用户，可进行心理陪伴对话');

-- ---------- 初始化数据：三个心理医生角色 ----------
-- 提示词与开场白由 scripts/seed_personas.py 幂等写入（内容较长，见 src/services/persona_seed.py）