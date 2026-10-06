-- MySQL 主数据表，对齐技术方案 7.1 与附录 C。
--
-- 三表分工：law 存法律本身，law_version 存版本与效力，article 存法条主数据。
-- 全部用 CREATE TABLE IF NOT EXISTS —— 本文件不会被容器自动执行：
-- mysql_data 卷已存在，docker-entrypoint-initdb.d 只在数据目录为空时跑一次，
-- 现在挂进去也不会生效。由 backend/app/ingest/load_mysql.py 显式执行。
--
-- 本期 article 表只写条级 1260 行，故 paragraph_no / item_no 两列恒为 NULL，
-- 款/项级编号留在 Milvus 的块级 payload 里——这是分工不是漏填。

CREATE TABLE IF NOT EXISTS law (
  law_id   VARCHAR(64)  PRIMARY KEY,
  law_name VARCHAR(128) NOT NULL
) DEFAULT CHARSET = utf8mb4;

CREATE TABLE IF NOT EXISTS law_version (
  law_id         VARCHAR(64) NOT NULL,
  law_version    VARCHAR(32) NOT NULL,
  effective_date DATE        NOT NULL,
  status         ENUM('现行有效','已废止','待生效') NOT NULL,
  PRIMARY KEY (law_id, law_version),
  KEY idx_status (status)
) DEFAULT CHARSET = utf8mb4;

CREATE TABLE IF NOT EXISTS article (
  id             BIGINT       PRIMARY KEY AUTO_INCREMENT,
  law_id         VARCHAR(64)  NOT NULL,
  law_version    VARCHAR(32)  NOT NULL,
  article_no     VARCHAR(16)  NOT NULL,
  article_no_cn  VARCHAR(32)  NOT NULL,
  paragraph_no   INT          NULL,
  item_no        VARCHAR(16)  NULL,
  path           VARCHAR(255) NOT NULL,
  text           TEXT         NOT NULL,
  effective_date DATE         NOT NULL,
  status         ENUM('现行有效','已废止','待生效') NOT NULL,
  replaced_by    VARCHAR(64)  NULL,
  replaces       VARCHAR(64)  NULL,
  source_hash    CHAR(32)     NOT NULL,
  team_id        BIGINT       NULL,
  KEY idx_law_art (law_id, article_no),
  KEY idx_status (status)
) DEFAULT CHARSET = utf8mb4;
