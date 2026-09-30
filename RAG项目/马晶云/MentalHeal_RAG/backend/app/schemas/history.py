from datetime import datetime

from pydantic import BaseModel, Field


class HistoryMessageUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class HistoryMessage(BaseModel):
    role: str
    content: str
    sources: list[dict] = []
    safety_intervention: bool = False


class HistoryConversation(BaseModel):
    session_id: str
    title: str
    role_id: str
    role_name: str
    created_at: datetime
    updated_at: datetime
    messages: list[HistoryMessage]


class HistoryResponse(BaseModel):
    conversations: list[HistoryConversation]
