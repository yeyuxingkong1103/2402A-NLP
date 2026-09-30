from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class _CharacterBase(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


class CharacterCreate(_CharacterBase):
    name: str = Field(min_length=1, max_length=64)
    system_prompt: str = ""
    model_name: str = Field(min_length=1, max_length=128)
    base_url: str | None = None
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 2048
    template_key: str | None = None


class CharacterUpdate(_CharacterBase):
    name: str | None = Field(default=None, min_length=1, max_length=64)
    system_prompt: str | None = None
    model_name: str | None = Field(default=None, min_length=1, max_length=128)
    base_url: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None


class CharacterRead(_CharacterBase):
    id: int
    name: str
    system_prompt: str
    model_name: str
    base_url: str | None
    temperature: float
    top_p: float
    max_tokens: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())
