CREATE DATABASE IF NOT EXISTS mentalheal_rag
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE mentalheal_rag;

CREATE TABLE IF NOT EXISTS users (
  id BIGINT NOT NULL AUTO_INCREMENT,
  user_id VARCHAR(64) NOT NULL,
  username VARCHAR(64) NOT NULL,
  password_hash VARCHAR(256) NOT NULL,
  display_name VARCHAR(80) NOT NULL,
  account_role VARCHAR(20) NOT NULL DEFAULT 'user',
  is_active TINYINT(1) NOT NULL DEFAULT 1,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_users_user_id (user_id),
  UNIQUE KEY uq_users_username (username)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS ai_roles (
  id BIGINT NOT NULL AUTO_INCREMENT,
  role_id VARCHAR(64) NOT NULL,
  name VARCHAR(80) NOT NULL,
  description VARCHAR(255) NOT NULL,
  system_instruction TEXT NOT NULL,
  is_active TINYINT(1) NOT NULL DEFAULT 1,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_ai_roles_role_id (role_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS chat_sessions (
  id BIGINT NOT NULL AUTO_INCREMENT,
  session_id VARCHAR(64) NOT NULL,
  user_id VARCHAR(64) NOT NULL,
  title VARCHAR(120) NOT NULL,
  last_message VARCHAR(300) NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  role_id VARCHAR(64) NOT NULL DEFAULT 'mental-health',
  status VARCHAR(20) NOT NULL DEFAULT 'active',
  PRIMARY KEY (id),
  UNIQUE KEY uq_chat_sessions_session_id (session_id),
  KEY idx_chat_sessions_user_updated (user_id, updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS chat_messages (
  id BIGINT NOT NULL AUTO_INCREMENT,
  message_id VARCHAR(64) NOT NULL,
  session_id VARCHAR(64) NOT NULL,
  user_id VARCHAR(64) NOT NULL,
  role VARCHAR(20) NOT NULL,
  content TEXT NOT NULL,
  citations_json TEXT NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_chat_messages_message_id (message_id),
  KEY idx_chat_messages_session_created (session_id, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS knowledge_bases (
  id BIGINT NOT NULL AUTO_INCREMENT,
  knowledge_base_id VARCHAR(64) NOT NULL,
  name VARCHAR(120) NOT NULL,
  description VARCHAR(500) NOT NULL DEFAULT '',
  is_active TINYINT(1) NOT NULL DEFAULT 1,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_knowledge_bases_knowledge_base_id (knowledge_base_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS documents (
  id BIGINT NOT NULL AUTO_INCREMENT,
  document_id VARCHAR(256) NOT NULL,
  source_file VARCHAR(512) NOT NULL,
  raw_path VARCHAR(1024) NOT NULL,
  processed_path VARCHAR(1024) NOT NULL,
  markdown_path VARCHAR(1024) NOT NULL,
  page_count INT NOT NULL DEFAULT 0,
  total_text_chars INT NOT NULL DEFAULT 0,
  weak_page_count INT NOT NULL DEFAULT 0,
  non_rag_page_count INT NOT NULL DEFAULT 0,
  table_page_count INT NOT NULL DEFAULT 0,
  content_type_counts JSON NOT NULL,
  cleaning_flag_counts JSON NOT NULL,
  is_enabled TINYINT(1) NOT NULL DEFAULT 1,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_documents_document_id (document_id),
  KEY idx_documents_enabled_updated (is_enabled, updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS knowledge_chunks (
  id BIGINT NOT NULL AUTO_INCREMENT,
  chunk_id VARCHAR(256) NOT NULL,
  document_id VARCHAR(256) NOT NULL,
  source_file VARCHAR(512) NOT NULL,
  page_start INT NOT NULL DEFAULT 0,
  page_end INT NOT NULL DEFAULT 0,
  chunk_index INT NOT NULL DEFAULT 0,
  chunk_type VARCHAR(32) NOT NULL DEFAULT 'text',
  char_count INT NOT NULL DEFAULT 0,
  token_estimate INT NOT NULL DEFAULT 0,
  has_table TINYINT(1) NOT NULL DEFAULT 0,
  quality_score FLOAT NOT NULL DEFAULT 1.0,
  milvus_collection VARCHAR(128) NOT NULL DEFAULT 'knowledge_chunks',
  knowledge_version VARCHAR(128) NOT NULL DEFAULT '',
  text_preview VARCHAR(512) NOT NULL DEFAULT '',
  metadata_json JSON NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_knowledge_chunks_chunk_id (chunk_id),
  KEY idx_knowledge_chunks_document (document_id),
  KEY idx_knowledge_chunks_page (document_id, page_start, page_end)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS knowledge_versions (
  id BIGINT NOT NULL AUTO_INCREMENT,
  version_name VARCHAR(128) NOT NULL,
  description TEXT NOT NULL,
  milvus_collection VARCHAR(128) NOT NULL,
  vector_model VARCHAR(512) NOT NULL,
  vector_dim INT NOT NULL,
  document_count INT NOT NULL DEFAULT 0,
  chunk_count INT NOT NULL DEFAULT 0,
  status VARCHAR(32) NOT NULL DEFAULT 'active',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_knowledge_versions_name (version_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS ingest_jobs (
  id BIGINT NOT NULL AUTO_INCREMENT,
  job_name VARCHAR(128) NOT NULL,
  status VARCHAR(32) NOT NULL DEFAULT 'queued',
  raw_dir VARCHAR(1024) NOT NULL,
  processed_dir VARCHAR(1024) NOT NULL,
  chunks_dir VARCHAR(1024) NOT NULL,
  embeddings_dir VARCHAR(1024) NOT NULL,
  document_count INT NOT NULL DEFAULT 0,
  page_count INT NOT NULL DEFAULT 0,
  chunk_count INT NOT NULL DEFAULT 0,
  note TEXT NOT NULL,
  summary_json JSON NOT NULL,
  started_at DATETIME NOT NULL,
  finished_at DATETIME NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_ingest_jobs_created (created_at),
  KEY idx_ingest_jobs_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS rag_evaluation_runs (
  id BIGINT NOT NULL AUTO_INCREMENT,
  run_id VARCHAR(64) NOT NULL,
  status VARCHAR(32) NOT NULL DEFAULT 'queued',
  dataset_path VARCHAR(512) NOT NULL,
  sample_count INT NOT NULL DEFAULT 0,
  metrics_json JSON NOT NULL,
  result_json JSON NOT NULL,
  error_message TEXT NOT NULL,
  started_at DATETIME NULL,
  finished_at DATETIME NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_rag_evaluation_runs_run_id (run_id),
  KEY idx_rag_evaluation_runs_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS long_term_memories (
  id BIGINT NOT NULL AUTO_INCREMENT,
  memory_id VARCHAR(64) NOT NULL,
  user_id VARCHAR(64) NOT NULL,
  memory_type VARCHAR(32) NOT NULL DEFAULT 'preference',
  content TEXT NOT NULL,
  is_confirmed TINYINT(1) NOT NULL DEFAULT 1,
  is_active TINYINT(1) NOT NULL DEFAULT 1,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_long_term_memories_memory_id (memory_id),
  KEY idx_long_term_memories_user_active (user_id, is_active, is_confirmed, updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

INSERT IGNORE INTO ai_roles
  (role_id, name, description, system_instruction)
VALUES
  ('mental-health', '心理健康助手', '通用心理健康科普与陪伴', '你是温和、稳妥的通用心理健康科普与陪伴助手。'),
  ('stress-management', '压力管理助手', '帮助识别压力并练习可执行的调节方法', '你专注于压力识别、情绪调节和可执行的日常减压方法。'),
  ('youth-companion', '青少年陪伴助手', '面向青少年的安全、友善心理支持', '你面向青少年提供安全、尊重、易懂的心理健康陪伴。');

INSERT IGNORE INTO knowledge_bases
  (knowledge_base_id, name, description)
VALUES
  ('default', '默认心理健康知识库', '当前项目导入的心理健康 PDF 知识库');
