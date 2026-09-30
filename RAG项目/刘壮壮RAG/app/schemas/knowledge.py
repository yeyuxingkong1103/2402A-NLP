from datetime import datetime

from pydantic import BaseModel, ConfigDict


class KnowledgeFileRead(BaseModel):
    id: int
    filename: str
    file_type: str
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
