"""MySQL-backed knowledge-base and document lifecycle service."""

from __future__ import annotations

import json
from typing import Any


class KnowledgeBaseService:
    """Keep management SQL isolated from HTTP routes."""

    def __init__(self, client: Any) -> None:
        self.client = client

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        settings = json.dumps(payload.get("settings", {}), ensure_ascii=False)
        self.client.execute(
            "INSERT INTO knowledge_bases "
            "(name, description, tenant_id, owner_id, settings) VALUES (%s, %s, %s, %s, %s)",
            (payload["name"], payload.get("description", ""), payload["tenant_id"],
             payload.get("owner_id"), settings),
        )
        return self.client.fetchone(
            "SELECT * FROM knowledge_bases WHERE tenant_id = %s AND name = %s "
            "ORDER BY id DESC LIMIT 1",
            (payload["tenant_id"], payload["name"]),
        ) or {}

    def list(self, tenant_id: int, offset: int, limit: int) -> tuple[list[dict[str, Any]], int]:
        rows = self.client.fetchall(
            "SELECT * FROM knowledge_bases WHERE tenant_id = %s "
            "AND status <> 'deleted' ORDER BY id DESC LIMIT %s OFFSET %s",
            (tenant_id, limit, offset),
        )
        total = self.client.fetchone(
            "SELECT COUNT(*) AS total FROM knowledge_bases "
            "WHERE tenant_id = %s AND status <> 'deleted'",
            (tenant_id,),
        )
        return rows, int((total or {}).get("total", 0))

    def get(self, kb_id: int, tenant_id: int) -> dict[str, Any] | None:
        return self.client.fetchone(
            "SELECT * FROM knowledge_bases WHERE id = %s AND tenant_id = %s "
            "AND status <> 'deleted'",
            (kb_id, tenant_id),
        )

    def update(self, kb_id: int, tenant_id: int, payload: dict[str, Any]) -> bool:
        fields, values = [], []
        for key in ("name", "description", "status"):
            if payload.get(key) is not None:
                fields.append(f"{key} = %s")
                values.append(payload[key])
        if payload.get("settings") is not None:
            fields.append("settings = %s")
            values.append(json.dumps(payload["settings"], ensure_ascii=False))
        if not fields:
            return bool(self.get(kb_id, tenant_id))
        values.extend([kb_id, tenant_id])
        return bool(self.client.execute(
            f"UPDATE knowledge_bases SET {', '.join(fields)} "
            "WHERE id = %s AND tenant_id = %s AND status <> 'deleted'",
            tuple(values),
        ))

    def delete(self, kb_id: int, tenant_id: int) -> bool:
        return bool(self.client.execute(
            "UPDATE knowledge_bases SET status = 'deleted' "
            "WHERE id = %s AND tenant_id = %s AND status <> 'deleted'",
            (kb_id, tenant_id),
        ))

    def documents(self, kb_id: int, tenant_id: int, offset: int, limit: int):
        rows = self.client.fetchall(
            "SELECT d.* FROM documents d JOIN knowledge_bases k "
            "ON k.id = d.knowledge_base_id WHERE d.knowledge_base_id = %s "
            "AND k.tenant_id = %s AND k.status <> 'deleted' "
            "AND d.tenant_id = %s AND d.status <> 'deleted' "
            "ORDER BY d.created_at DESC LIMIT %s OFFSET %s",
            (kb_id, tenant_id, tenant_id, limit, offset),
        )
        total = self.client.fetchone(
            "SELECT COUNT(*) AS total FROM documents WHERE knowledge_base_id = %s "
            "AND tenant_id = %s AND status <> 'deleted'",
            (kb_id, tenant_id),
        )
        return rows, int((total or {}).get("total", 0))

    def document(self, document_id: str, tenant_id: int) -> dict[str, Any] | None:
        return self.client.fetchone(
            "SELECT * FROM documents WHERE id = %s AND tenant_id = %s "
            "AND status <> 'deleted'",
            (document_id, tenant_id),
        )

    def mark_document_deleted(self, document_id: str, tenant_id: int) -> bool:
        return bool(self.client.execute(
            "UPDATE documents SET status = 'deleted' WHERE id = %s "
            "AND tenant_id = %s AND status <> 'deleted'",
            (document_id, tenant_id),
        ))
