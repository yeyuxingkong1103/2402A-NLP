-- Knowledge-base and evaluation tables for the current ORM models.
-- Run after 001_auth_roles.sql. Statements are safe to re-run.

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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

INSERT IGNORE INTO knowledge_bases
  (knowledge_base_id, name, description)
VALUES
  ('default', '默认心理健康知识库', '当前项目导入的心理健康 PDF 知识库');
