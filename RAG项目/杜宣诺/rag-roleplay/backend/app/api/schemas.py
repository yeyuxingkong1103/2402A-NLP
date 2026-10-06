from pydantic import BaseModel


def ok(data=None):
    return {"code": 0, "data": data, "message": "ok"}


def err(code: int, message: str):
    return {"code": code, "data": None, "message": message}


class RegisterIn(BaseModel):
    username: str
    password: str


class LoginIn(BaseModel):
    username: str
    password: str


class CharacterIn(BaseModel):
    name: str
    avatar: str = ""
    persona: str = ""
    worldview: str = ""
    relationship: str = ""
    hidden_setting: str = ""
    greeting: str = ""
    sample_dialogue: str = ""
    tags: str = ""
