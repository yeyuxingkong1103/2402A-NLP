-- User-confirmed long-term memories. No automatic chat extraction writes this table.
-- Run after 002_knowledge_evaluation.sql.

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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
