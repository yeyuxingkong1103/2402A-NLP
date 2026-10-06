-- Run against mentalheal_rag_dev after taking a database backup.
-- This migration is idempotent except for the ALTER TABLE statements; run those
-- only if the columns are not already present.

ALTER TABLE users
  ADD COLUMN account_role VARCHAR(20) NOT NULL DEFAULT 'user',
  ADD COLUMN is_active TINYINT(1) NOT NULL DEFAULT 1;

ALTER TABLE chat_sessions
  ADD COLUMN status VARCHAR(20) NOT NULL DEFAULT 'active';

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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

INSERT IGNORE INTO ai_roles
  (role_id, name, description, system_instruction)
VALUES
  ('mental-health', '心理健康助手', '通用心理健康科普与陪伴', '你是温和、稳妥的通用心理健康科普与陪伴助手。'),
  ('stress-management', '压力管理助手', '帮助识别压力并练习可执行的调节方法', '你专注于压力识别、情绪调节和可执行的日常减压方法。'),
  ('youth-companion', '青少年陪伴助手', '面向青少年的安全、友善心理支持', '你面向青少年提供安全、尊重、易懂的心理健康陪伴。');
