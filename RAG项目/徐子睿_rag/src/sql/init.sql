-- ============================================================
-- sql/init.sql —— MySQL 建表脚本（新架构 src/ 的关系库结构）
--
-- 用途：由 docker-compose.middleware.yml 挂载到
--   /docker-entrypoint-initdb.d/init.sql
-- MySQL 容器**首次初始化**时会自动执行它建好全部表。
--
-- 与代码的对应关系：
--   这些表与 src/models/database.py 里的 SQLAlchemy 模型一一对应。
--   SQLAlchemy 的 Base.metadata.create_all() 也能建出同样的表，
--   本文件的作用是：让 MySQL 容器一起来就带好完整的空表结构，
--   不必等应用首次启动时才去建。
--
-- 注意：MySQL 的 CREATE TABLE IF NOT EXISTS 只建缺失的表，
-- 不会修改已存在表的结构。改表结构需要另写 ALTER 或迁移脚本。
-- ============================================================

-- 用户表：注册登录的主体
CREATE TABLE IF NOT EXISTS user (
  id INT AUTO_INCREMENT PRIMARY KEY,
  username VARCHAR(128) UNIQUE NOT NULL,          -- 用户名，唯一约束防重名注册（应用层也会先查一次，这里是最终防线）
  password_hash VARCHAR(256) NOT NULL,            -- 口令的 PBKDF2 哈希值，绝不存明文
  tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',-- 所属租户，多租户隔离维度
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 角色表：角色扮演的角色卡
-- 注意 role 是 MySQL 的关键字，引用时需加反引号（本文件的建表语句无需加，SQLAlchemy 侧已处理）
CREATE TABLE IF NOT EXISTS role (
  id INT AUTO_INCREMENT PRIMARY KEY,
  role_id VARCHAR(64) NOT NULL,                   -- 业务角色标识（lawyer / psychologist 等），向量库的过滤字段用的就是它
  tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
  config JSON NOT NULL,                           -- 整份角色卡的 JSON（persona / style / safety_notice / bound_kb ...）
                                                  -- 用 JSON 存是为了加字段不必改表结构
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,  -- 由 MySQL 在更新时自动维护
  -- 同租户内角色标识唯一；跨租户可重名（两个租户能各有自己的 lawyer）
  UNIQUE KEY uq_role_tenant(role_id, tenant_id)
);

-- 会话表：一个用户在某个角色下的一段连续对话
-- 注意业务表名 session 也是 MySQL 关键字，引用时同样需要反引号
CREATE TABLE IF NOT EXISTS session (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NOT NULL,                           -- 归属用户（会话按用户隔离，越权访问会被应用层拦成 404）
  role_id VARCHAR(64) NOT NULL,                   -- 业务角色标识。存字符串而非外键：角色来源可能是内置或配置文件，做外键会把来源锁死
  tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
  title VARCHAR(256) DEFAULT '',                  -- 会话标题，由首条消息自动生成，便于在列表里辨认
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP  -- 会话列表按它倒序排，最近活跃的浮到最前
);

-- 消息表：对话的每一条发言
CREATE TABLE IF NOT EXISTS message (
  id INT AUTO_INCREMENT PRIMARY KEY,
  session_id INT NOT NULL,                        -- 所属会话
  user_id INT NOT NULL,                           -- 发言用户（冗余字段：便于按用户审计，不必 JOIN session）
  role_id VARCHAR(64) NOT NULL,                   -- 业务角色标识（同样冗余，便于按角色统计）
  speaker VARCHAR(32) NOT NULL,                   -- 发言方："user" 或 "assistant"
  content TEXT NOT NULL,                          -- 消息正文
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 知识库文档登记表：一份文档一行
CREATE TABLE IF NOT EXISTS knowledge_doc (
  id INT AUTO_INCREMENT PRIMARY KEY,
  role_id VARCHAR(64) NOT NULL,                   -- 归属角色（同一份文件可以给不同角色各建一份）
  tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
  doc_source VARCHAR(512) NOT NULL,               -- 文档来源（文件路径）
  status VARCHAR(32) DEFAULT 'ready',             -- 构建状态：ready / building / failed
                                                  -- 查询侧靠它判断"这份文档的数据是否完整可用"
  metadata_json JSON,                             -- 附加元数据（解析器、分块数、分块策略等）
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- 知识库分块明细表：一个 chunk 一行
-- 注意本表没有建 doc_id 与 knowledge_doc.id 的外键约束：
--   Milvus 里存的是同一个 doc_id，两边靠它对齐；
--   加外键会让"删文档"的操作顺序被数据库强制约束，而当前实现是先删向量再删元数据。
CREATE TABLE IF NOT EXISTS knowledge_chunk (
  id INT AUTO_INCREMENT PRIMARY KEY,
  doc_id INT NOT NULL,                            -- 所属文档 id，与 Milvus 记录里的 doc_id 对应
  role_id VARCHAR(64) NOT NULL,
  tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
  content TEXT NOT NULL,                          -- 分块正文（应用层入库时会截断到 8192 字符）
  summary VARCHAR(1024) DEFAULT '',               -- 摘要，供知识库列表页展示，避免拖回全文
  parent_id INT NULL,                             -- 父块序号，可空。是逻辑序号而非外键（见 chunkers.add_parent_child）
  page INT DEFAULT -1,                            -- 页码，-1 表示"不适用"（单页文档或记忆条目）
  metadata_json JSON,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 用户反馈表：对某条回答或某次会话的评价
CREATE TABLE IF NOT EXISTS feedback (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NOT NULL,                           -- 评价人
  session_id INT NULL,                            -- 关联会话，可空（只对整体体验评分时不传）
  message_id INT NULL,                            -- 关联到具体那条回答，可空
  value FLOAT NOT NULL,                           -- 评分数值。用 FLOAT 而非枚举，以同时支持"点赞/点踩"和"1-5 分"等交互
  comment TEXT,                                   -- 文字评论
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
