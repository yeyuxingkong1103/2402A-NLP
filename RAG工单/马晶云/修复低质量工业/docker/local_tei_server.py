import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_PATH = os.environ.get("TEI_MODEL_PATH", "/Users/majingyun/Documents/models/bge-m3")
MODEL_DEVICE = os.environ.get("TEI_DEVICE", "cpu")
MODEL_BATCH_SIZE = int(os.environ.get("TEI_BATCH_SIZE", "8"))
MODEL = SentenceTransformer(MODEL_PATH, device=MODEL_DEVICE)


class EmbeddingHandler(BaseHTTPRequestHandler):
    def _json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/health"):
            self._json(200, {"status": "ok", "model": MODEL_PATH})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/embed":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            inputs = request.get("inputs")
            texts = [inputs] if isinstance(inputs, str) else inputs
            if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
                raise ValueError("inputs must be a string or a list of strings")
            vectors = MODEL.encode(
                texts,
                batch_size=MODEL_BATCH_SIZE,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            vectors = np.asarray(vectors, dtype=np.float32).tolist()
            self._json(200, vectors)
        except Exception as exc:
            self._json(400, {"error": str(exc)})

    def log_message(self, format, *args):
        print(format % args, flush=True)


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 6380), EmbeddingHandler).serve_forever()
