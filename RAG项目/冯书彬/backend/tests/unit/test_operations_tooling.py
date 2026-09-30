from pathlib import Path

import pytest

from backend.app.core.metrics import MetricsRegistry
from backend.scripts.backup_database import build_dump_command
from backend.scripts.health_load_test import _percentile
from backend.scripts.verify_database_backup import build_restore_command


def test_metrics_registry_renders_request_and_health_metrics():
    registry = MetricsRegistry()
    registry.observe_request("GET", "/health/live", 200, 0.125)
    registry.set_health("mysql", True)

    rendered = registry.render_prometheus()

    assert 'legal_rag_http_requests_total{method="GET",path="/health/live",status_code="200"} 1' in rendered
    assert 'legal_rag_health_check_status{check="mysql"} 1' in rendered
    assert "0.125000" in rendered


def test_metrics_registry_escapes_prometheus_labels():
    registry = MetricsRegistry()
    registry.observe_request("GET", '/unsafe/"path', 500, 0.1)

    rendered = registry.render_prometheus()

    assert 'path="/unsafe/\\"path"' in rendered


def test_backup_command_keeps_password_out_of_arguments(tmp_path: Path):
    command, environment = build_dump_command(
        "mysql+pymysql://backup:secret@example.test:3307/legal_rag",
        tmp_path / "backup.sql",
    )

    assert "secret" not in command
    assert environment["MYSQL_PWD"] == "secret"
    assert "--single-transaction" in command
    assert "--hex-blob" in command


def test_restore_command_rejects_production_database(tmp_path: Path):
    with pytest.raises(ValueError, match="independent"):
        build_restore_command("mysql+pymysql://backup:secret@example.test/legal_rag", tmp_path / "backup.sql")


def test_restore_command_uses_independent_database(tmp_path: Path):
    command, environment = build_restore_command(
        "mysql+pymysql://backup:secret@example.test:3307/legal_rag_restore",
        tmp_path / "backup.sql",
    )

    assert command[-1] == "legal_rag_restore"
    assert "secret" not in command
    assert environment["MYSQL_PWD"] == "secret"


def test_percentile_uses_sorted_nearest_rank():
    assert _percentile([0.1, 0.2, 0.3, 0.4], 0.95) == 0.3
