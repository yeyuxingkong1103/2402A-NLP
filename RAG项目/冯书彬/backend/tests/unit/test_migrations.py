from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory


ROOT = Path(__file__).resolve().parents[3]


def test_alembic_revision_chain_reaches_expected_head():
    # 读取仓库配置验证所有迁移 revision 均可被 Alembic 发现。
    config = Config(str(ROOT / "alembic.ini"))
    scripts = ScriptDirectory.from_config(config)
    revisions = list(scripts.walk_revisions())

    assert scripts.get_heads() == ["013_expand_document_chunk_ids"]
    assert len(revisions) == 13
    assert {revision.revision for revision in revisions} == {
        "001_create_users",
        "002_create_roles",
        "003_create_conversations",
        "004_create_messages",
        "005_create_knowledge_bases",
        "006_create_documents",
        "007_create_feedback",
        "008_create_audit_logs",
        "009_create_memories_exports",
        "010_expand_knowledge_text",
        "011_expand_chat_external_ids",
        "012_add_document_chunk_hierarchy",
        "013_expand_document_chunk_ids",
    }


def test_alembic_config_does_not_embed_database_credentials():
    # 数据库连接必须从 DATABASE_URL 环境变量读取，配置文件不保存凭据。
    config_text = (ROOT / "alembic.ini").read_text(encoding="utf-8")

    assert "sqlalchemy.url =" in config_text
    assert "mysql://" not in config_text
    assert "password" not in config_text.lower()
