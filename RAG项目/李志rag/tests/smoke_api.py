import os
from pathlib import Path

import requests

BASE_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000/api/v1")
USERNAME = os.getenv("RAG_TEST_USERNAME", "admin")
PASSWORD = os.getenv("RAG_TEST_PASSWORD", "ChangeMe123!")
PDF = Path("output/pdf/medical_knowledge_sample.pdf")
SOURCE = "https://www.nhc.gov.cn/cms-search/downFiles/63f752a17cfd4b4781f744477561866f.pdf"


def request(method: str, path: str, **kwargs) -> requests.Response:
    response = requests.request(method, f"{BASE_URL}{path}", timeout=300, **kwargs)
    response.raise_for_status()
    return response


def main() -> None:
    login = request(
        "POST", "/auth/login", json={"username": USERNAME, "password": PASSWORD}
    ).json()
    headers = {"Authorization": f"Bearer {login['access_token']}"}
    roles = request("GET", "/roles", headers=headers).json()
    role_id = roles[0]["id"]
    with PDF.open("rb") as stream:
        uploaded = request(
            "POST",
            "/knowledge/documents",
            headers=headers,
            files={"file": (PDF.name, stream, "application/pdf")},
            data={"role_id": role_id, "source_url": SOURCE},
        ).json()
    result = request(
        "POST",
        "/chat",
        headers=headers,
        json={
            "message": "高血压日常饮食有哪些一般原则？",
            "role_id": role_id,
            "session_id": "smoke-test",
        },
    ).json()
    assert uploaded["status"] == "ready"
    assert result["answer"]
    assert result["sources"]
    print(f"Smoke test passed: document={uploaded['id']}, hits={len(result['sources'])}")


if __name__ == "__main__":
    main()
