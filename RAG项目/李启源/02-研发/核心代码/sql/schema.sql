-- MySQL Schema for RAG Knowledge Base System
-- Version: 1.0
-- Description: Document and chunk metadata storage

-- ============================================
-- 知识库表
-- ============================================
CREATE TABLE IF NOT EXISTS knowledge_bases (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(200) NOT NULL COMMENT '知识库名称',
    description TEXT COMMENT '知识库描述',
    tenant_id BIGINT NOT NULL DEFAULT 1 COMMENT '租户ID',
    owner_id BIGINT COMMENT '所有者ID',
    status ENUM('active', 'archived', 'deleted') DEFAULT 'active' COMMENT '状态',
    settings JSON COMMENT '知识库设置',
    created_at TIMESTAMP(6) DEFAULT CURRENT_TIMESTAMP(6),
    updated_at TIMESTAMP(6) DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='知识库表';

-- ============================================
-- 文档表
-- ============================================
CREATE TABLE IF NOT EXISTS documents (
    id VARCHAR(64) PRIMARY KEY COMMENT '文档ID，通常为UUID',
    knowledge_base_id BIGINT NOT NULL COMMENT '所属知识库ID',
    version_no INT NOT NULL DEFAULT 1 COMMENT '版本号',
    file_name VARCHAR(500) NOT NULL COMMENT '文件名',
    storage_path VARCHAR(1000) NOT NULL COMMENT '存储路径',
    mime_type VARCHAR(100) COMMENT 'MIME类型',
    file_size BIGINT COMMENT '文件大小(字节)',
    file_sha256 CHAR(64) NOT NULL COMMENT '文件SHA256哈希',
    status ENUM('pending', 'processing', 'processed', 'failed', 'deleted') DEFAULT 'pending' COMMENT '处理状态',
    review_status ENUM('pending', 'approved', 'rejected') DEFAULT 'pending' COMMENT '审核状态',
    parser_type VARCHAR(50) COMMENT '使用的解析器：pymupdf, pdfplumber, paddleocr, mineru',
    chunking_strategy VARCHAR(50) COMMENT '分块策略：fixed, sentence, paragraph, heading, semantic',
    page_count INT COMMENT '页数',
    character_count INT COMMENT '字符数',
    chunk_count INT COMMENT '分块数量',
    source_metadata JSON COMMENT '源文档元数据',
    error_message TEXT COMMENT '错误信息',
    tenant_id BIGINT NOT NULL DEFAULT 1 COMMENT '租户ID',
    role_id BIGINT DEFAULT 0 COMMENT '角色ID',
    created_by BIGINT COMMENT '创建者ID',
    created_at TIMESTAMP(6) DEFAULT CURRENT_TIMESTAMP(6),
    updated_at TIMESTAMP(6) DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    processed_at TIMESTAMP(6) NULL COMMENT '处理完成时间',
    INDEX idx_kb_id (knowledge_base_id),
    INDEX idx_file_sha256 (file_sha256),
    INDEX idx_status (status),
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_created_at (created_at),
    UNIQUE KEY uk_kb_sha256_version (knowledge_base_id, file_sha256, version_no),
    FOREIGN KEY (knowledge_base_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='文档表';

-- ============================================
-- 文档块表
-- ============================================
CREATE TABLE IF NOT EXISTS chunks (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    document_id VARCHAR(64) NOT NULL COMMENT '所属文档ID',
    chunk_id VARCHAR(128) NOT NULL COMMENT '块ID，全局唯一',
    milvus_id VARCHAR(128) NOT NULL COMMENT 'Milvus中的ID',
    chunk_index INT NOT NULL COMMENT '块索引（在文档中的顺序）',
    content TEXT NOT NULL COMMENT '块内容',
    summary TEXT COMMENT '块摘要',
    source VARCHAR(200) COMMENT '来源（文件名:页码）',
    page_start INT COMMENT '起始页码',
    page_end INT COMMENT '结束页码',
    parent_id VARCHAR(128) COMMENT '父块ID',
    parent_summary TEXT COMMENT '父块摘要',
    text_hash CHAR(64) NOT NULL COMMENT '内容哈希',
    token_count INT COMMENT 'Token数量',
    embedding_status ENUM('pending', 'completed', 'failed') DEFAULT 'pending' COMMENT '嵌入状态',
    metadata JSON COMMENT '其他元数据',
    created_at TIMESTAMP(6) DEFAULT CURRENT_TIMESTAMP(6),
    updated_at TIMESTAMP(6) DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    INDEX idx_document_id (document_id),
    INDEX idx_chunk_id (chunk_id),
    INDEX idx_milvus_id (milvus_id),
    INDEX idx_text_hash (text_hash),
    INDEX idx_embedding_status (embedding_status),
    FULLTEXT KEY ft_chunk_content (content),
    UNIQUE KEY uk_chunk_id (chunk_id),
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='文档块表';

-- ============================================
-- 父块表（可选，用于父子块架构）
-- ============================================
CREATE TABLE IF NOT EXISTS parent_blocks (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    document_id VARCHAR(64) NOT NULL COMMENT '所属文档ID',
    parent_id VARCHAR(128) NOT NULL COMMENT '父块ID',
    content TEXT NOT NULL COMMENT '父块内容',
    summary TEXT COMMENT '父块摘要',
    page_start INT COMMENT '起始页码',
    page_end INT COMMENT '结束页码',
    child_count INT DEFAULT 0 COMMENT '子块数量',
    metadata JSON COMMENT '其他元数据',
    created_at TIMESTAMP(6) DEFAULT CURRENT_TIMESTAMP(6),
    updated_at TIMESTAMP(6) DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    INDEX idx_document_id (document_id),
    INDEX idx_parent_id (parent_id),
    UNIQUE KEY uk_parent_id (parent_id),
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='父块表';

-- ============================================
-- 处理日志表
-- ============================================
CREATE TABLE IF NOT EXISTS processing_logs (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    document_id VARCHAR(64) NOT NULL COMMENT '文档ID',
    stage ENUM('parse', 'chunk', 'embed', 'index', 'complete') NOT NULL COMMENT '处理阶段',
    status ENUM('started', 'completed', 'failed') NOT NULL COMMENT '状态',
    message TEXT COMMENT '日志消息',
    details JSON COMMENT '详细信息',
    duration_ms INT COMMENT '耗时（毫秒）',
    created_at TIMESTAMP(6) DEFAULT CURRENT_TIMESTAMP(6),
    INDEX idx_document_id (document_id),
    INDEX idx_stage (stage),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='处理日志表';

-- ============================================
-- 示例数据
-- ============================================
INSERT INTO knowledge_bases (name, description, tenant_id) VALUES
('默认知识库', '系统默认知识库', 1),
('产品文档', '产品使用文档和说明', 1),
('技术文档', '技术开发文档', 1);
