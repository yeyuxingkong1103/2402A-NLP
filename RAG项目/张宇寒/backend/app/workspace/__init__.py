from .api import delete_file, list_files, router, upload_file, upload_files
from .search import WorkspacePrivateSearchMixin
from .service import WorkspaceService

for _name in ("api", "file_utits", "search", "service"):
    globals()[_name] = __import__(f"{__name__}.{_name}", fromlist=["*"])

__all__ = [
    "WorkspaceService", "WorkspacePrivateSearchMixin",
    "router", "upload_file", "upload_files", "list_files", "delete_file",
]
