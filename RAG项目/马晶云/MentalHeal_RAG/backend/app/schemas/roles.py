from pydantic import BaseModel, Field


class AIRoleResponse(BaseModel):
    role_id: str
    name: str
    description: str
    is_active: bool


class RoleCreateRequest(BaseModel):
    role_id: str = Field(min_length=2, max_length=64, pattern=r"^[a-z0-9-]+$")
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=255)
    system_instruction: str = Field(min_length=1, max_length=4000)


class RoleUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    description: str | None = Field(default=None, min_length=1, max_length=255)
    system_instruction: str | None = Field(default=None, min_length=1, max_length=4000)
    is_active: bool | None = None


class RoleListResponse(BaseModel):
    roles: list[AIRoleResponse]
