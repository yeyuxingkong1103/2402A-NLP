"""Validate the deployed doctor UI and one real RAG answer. Standard library only."""

import argparse
import json
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def request(base, path, method="GET", data=None, timeout=180):
    body = json.dumps(data, ensure_ascii=False).encode("utf-8") if data is not None else None
    req = Request(base.rstrip("/") + path, data=body, method=method,
                  headers={"Content-Type": "application/json", "Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if "application/json" in response.headers.get("Content-Type", "") else raw
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path}: HTTP {exc.code}: {detail[:500]}") from exc


def check(base):
    started = time.monotonic()
    print("Checking web page, assets and doctor endpoint...", flush=True)
    page = request(base, "/chat")
    if not isinstance(page, str) or "chat-form" not in page:
        raise RuntimeError("Chat page is missing; upload the latest update.")
    for path, marker in (("/static/app.js", "checkService"), ("/static/app.css", ".composer")):
        if marker not in request(base, path):
            raise RuntimeError(f"Static asset missing: {path}")
    roles = request(base, "/api/roles")
    if not isinstance(roles, list) or len(roles) != 1 or roles[0].get("name") != "医生":
        raise RuntimeError("Expected the doctor-only API; restart it with the updated code.")
    status = request(base, "/api/status")
    if not status.get("model_ready"):
        raise RuntimeError("Model endpoint is not ready. See logs/demo-vllm.log.")

    # A negative, unique ID isolates diagnostic memory from real user accounts.
    user_id, role_id = -time.time_ns(), roles[0]["id"]
    params = f"?user_id={user_id}&role_id={role_id}"
    question = "在家测量血压需要注意什么？"
    print("Checking one real answer and knowledge citations (may take up to 180s)...", flush=True)
    try:
        answer = request(base, "/api/chat", "POST", {"user_id": user_id, "role_id": role_id, "message": question})
        if not isinstance(answer.get("answer"), str) or not answer["answer"].strip():
            raise RuntimeError("Model returned no answer text.")
        sources = answer.get("sources", [])
        if not sources or not all(item.get("text") and item.get("source") for item in sources):
            raise RuntimeError("Answer arrived without the expected knowledge source text.")
        history = request(base, "/api/history" + params)
        if question not in history.get("history", ""):
            raise RuntimeError("Conversation memory did not retain the question.")
        request(base, "/api/history" + params, "DELETE")
        if question in request(base, "/api/history" + params).get("history", ""):
            raise RuntimeError("Conversation clear did not take effect.")
    except Exception:
        try:
            request(base, "/api/history" + params, "DELETE", timeout=10)
        except Exception:
            pass
        raise
    report = {"ok": True, "elapsed_seconds": round(time.monotonic() - started, 2),
              "question": question, "answer": answer["answer"], "sources": sources,
              "checks": ["web", "assets", "doctor", "model", "real_rag_answer", "citations", "memory", "clear"]}
    path = Path(__file__).resolve().parents[1] / "outputs" / "demo-check.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DEMO_CHECK_OK: page, answer, citations and memory passed.", flush=True)
    print("Report: " + str(path), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:6006")
    args = parser.parse_args()
    try:
        check(args.base_url)
    except Exception as exc:
        print(f"DEMO_CHECK_FAILED: {exc}", flush=True)
        raise SystemExit(1)
