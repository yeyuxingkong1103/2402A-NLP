from backend.app.models.knowledge_base import CrawlSnapshot, KnowledgeMaterial, SourceWhitelistEntry

_whitelist_entries: dict[str, SourceWhitelistEntry] = {}
_snapshots: dict[str, CrawlSnapshot] = {}
_materials: dict[str, KnowledgeMaterial] = {}


def reset_store() -> None:
    # 测试使用内存仓储，重置时清空三类治理对象。
    _whitelist_entries.clear()
    _snapshots.clear()
    _materials.clear()


def save_whitelist_entry(entry: SourceWhitelistEntry) -> SourceWhitelistEntry:
    # 白名单按 ID 保存，URL 唯一性由服务层判断。
    _whitelist_entries[entry.id] = entry
    return entry


def get_whitelist_entry(entry_id: str) -> SourceWhitelistEntry | None:
    # 未找到返回 None，由服务层转换为业务错误。
    return _whitelist_entries.get(entry_id)


def find_active_whitelist_for_url(source_url: str) -> SourceWhitelistEntry | None:
    # 抓取源必须完全匹配一个启用中的白名单地址或其子路径。
    for entry in _whitelist_entries.values():
        if entry.active and (source_url == entry.url or source_url.startswith(f"{entry.url.rstrip('/')}/")):
            return entry
    return None


def find_whitelist_by_url(url: str) -> SourceWhitelistEntry | None:
    # 用于创建时避免重复维护同一来源。
    for entry in _whitelist_entries.values():
        if entry.url == url:
            return entry
    return None


def delete_whitelist_entry(entry_id: str) -> SourceWhitelistEntry | None:
    # 白名单删除仅移除来源授权，不删除已形成的治理留痕。
    return _whitelist_entries.pop(entry_id, None)


def save_snapshot(snapshot: CrawlSnapshot) -> CrawlSnapshot:
    # 快照只追加或覆盖同 ID，不会改变 searchable 的默认 false。
    _snapshots[snapshot.id] = snapshot
    return snapshot


def get_snapshot(snapshot_id: str) -> CrawlSnapshot | None:
    # 服务层负责状态迁移检查。
    return _snapshots.get(snapshot_id)


def save_material(material: KnowledgeMaterial) -> KnowledgeMaterial:
    # 材料状态变化后回写同一对象。
    _materials[material.id] = material
    return material


def get_material(material_id: str) -> KnowledgeMaterial | None:
    # 未找到返回 None，API 层不直接访问仓储。
    return _materials.get(material_id)


def list_materials() -> list[KnowledgeMaterial]:
    # 返回治理材料快照，调用方可按状态展示审核队列。
    return list(_materials.values())
