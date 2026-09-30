from pymilvus import MilvusClient


class VectorStore:
    KNOWLEDGE_COLLECTION = "knowledge_chunks"
    IMAGE_COLLECTION = "image_vectors"
    MEMORY_COLLECTION = "long_term_memory"

    def __init__(self, uri: str, text_dim: int, image_dim: int):
        self._client = MilvusClient(uri=uri)
        self._text_dim = text_dim
        self._image_dim = image_dim

    def ensure_collections(self) -> None:
        self._client.create_collection(self.KNOWLEDGE_COLLECTION, dimension=self._text_dim)
        self._client.create_collection(self.IMAGE_COLLECTION, dimension=self._image_dim)
        self._client.create_collection(self.MEMORY_COLLECTION, dimension=self._text_dim)

    def insert_text_chunks(
        self,
        user_id: int,
        character_id: int,
        file_id: int,
        chunks: list[str],
        vectors: list[list[float]],
    ) -> list[int]:
        rows = [
            {
                "user_id": user_id,
                "character_id": character_id,
                "file_id": file_id,
                "text": text,
                "vector": vector,
            }
            for text, vector in zip(chunks, vectors)
        ]
        result = self._client.insert(self.KNOWLEDGE_COLLECTION, rows)
        return result["ids"] if isinstance(result, dict) else []

    def insert_image_vector(
        self,
        user_id: int,
        character_id: int,
        file_id: int,
        image_path: str,
        vector: list[float],
    ) -> int:
        rows = [
            {
                "user_id": user_id,
                "character_id": character_id,
                "file_id": file_id,
                "image_path": image_path,
                "vector": vector,
            }
        ]
        result = self._client.insert(self.IMAGE_COLLECTION, rows)
        if isinstance(result, dict) and result.get("ids"):
            return result["ids"][0]
        return 0

    def insert_memory(
        self, user_id: int, character_id: int, text: str, vector: list[float]
    ) -> int:
        rows = [
            {
                "user_id": user_id,
                "character_id": character_id,
                "text": text,
                "vector": vector,
            }
        ]
        result = self._client.insert(self.MEMORY_COLLECTION, rows)
        if isinstance(result, dict) and result.get("ids"):
            return result["ids"][0]
        return 0

    def search_text(self, character_id: int, query_vector: list[float], top_k: int) -> list[dict]:
        result = self._client.search(
            self.KNOWLEDGE_COLLECTION,
            data=[query_vector],
            filter=f"character_id == {character_id}",
            limit=top_k,
            output_fields=["text", "file_id"],
        )
        hits = result[0] if result else []
        return [h["entity"] for h in hits]

    def search_memory(
        self, character_id: int, query_vector: list[float], top_k: int
    ) -> list[dict]:
        result = self._client.search(
            self.MEMORY_COLLECTION,
            data=[query_vector],
            filter=f"character_id == {character_id}",
            limit=top_k,
            output_fields=["text"],
        )
        hits = result[0] if result else []
        return [h["entity"] for h in hits]

    def delete_by_file(self, character_id: int, file_id: int) -> None:
        self._client.delete(
            self.KNOWLEDGE_COLLECTION,
            filter=f"character_id == {character_id} and file_id == {file_id}",
        )
        self._client.delete(
            self.IMAGE_COLLECTION,
            filter=f"character_id == {character_id} and file_id == {file_id}",
        )

    def delete_by_character(self, character_id: int) -> None:
        self._client.delete(
            self.KNOWLEDGE_COLLECTION, filter=f"character_id == {character_id}"
        )
        self._client.delete(
            self.IMAGE_COLLECTION, filter=f"character_id == {character_id}"
        )
        self._client.delete(
            self.MEMORY_COLLECTION, filter=f"character_id == {character_id}"
        )
