from ..db.models import Character

CHUNK_SIZE = 500


def _chunk(text: str) -> list[str]:
    return [text[i : i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)] if text else []


async def chunk_and_index(milvus, character: Character) -> None:
    await milvus.delete_by_filter("character_settings", f"character_id == {character.id}")
    fields = {
        "persona": character.persona,
        "worldview": character.worldview,
        "relationship": character.relationship,
        "hidden_setting": character.hidden_setting,
        "sample_dialogue": character.sample_dialogue,
    }
    chunks = []
    for setting_type, text in fields.items():
        for ci, piece in enumerate(_chunk(text)):
            chunks.append({"setting_type": setting_type, "text": piece, "chunk_index": ci})
    if chunks:
        await milvus.upsert_settings(character.id, chunks)


def to_public(character: Character, viewer_id: int | None) -> dict:
    data = {
        "id": character.id, "name": character.name, "avatar": character.avatar,
        "persona": character.persona, "worldview": character.worldview,
        "relationship": character.relationship, "greeting": character.greeting,
        "sample_dialogue": character.sample_dialogue, "is_preset": character.is_preset,
        "tags": character.tags,
    }
    is_owner = viewer_id is not None and character.owner_user_id == viewer_id
    if is_owner and not character.is_preset:
        data["hidden_setting"] = character.hidden_setting
    return data
