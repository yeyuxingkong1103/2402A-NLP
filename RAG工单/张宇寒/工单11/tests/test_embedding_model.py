import tempfile
from pathlib import Path
import unittest

import numpy as np

from embedding_model import DomainEmbeddingModel


TEXTS = [
    "甲公司营业收入100亿元",
    "甲公司员工人数500人",
    "乙公司净利润20亿元",
    "乙公司风险管理体系完善",
]

TRIPLETS = [
    {"query": "甲公司的营业收入是多少", "positive": TEXTS[0], "negative": TEXTS[1]},
    {"query": "乙公司的净利润是多少", "positive": TEXTS[2], "negative": TEXTS[3]},
]


class EmbeddingModelTests(unittest.TestCase):
    def test_base_encoder_returns_normalized_vectors(self):
        model = DomainEmbeddingModel(dimension=3, seed=42)
        model.fit_base(TEXTS + [row["query"] for row in TRIPLETS])
        vectors = model.encode(TEXTS)
        self.assertEqual(vectors.shape, (4, 3))
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), np.ones(4), atol=1e-6)

    def test_triplet_training_does_not_increase_loss(self):
        model = DomainEmbeddingModel(dimension=3, seed=42)
        model.fit_base(TEXTS + [row["query"] for row in TRIPLETS])
        history = model.fit_adapter(TRIPLETS, epochs=25, learning_rate=0.05, margin=1.0)
        self.assertEqual(len(history), 25)
        self.assertLessEqual(history[-1], history[0] + 1e-8)

    def test_saved_model_round_trips_embeddings(self):
        model = DomainEmbeddingModel(dimension=3, seed=42)
        model.fit_base(TEXTS + [row["query"] for row in TRIPLETS])
        model.fit_adapter(TRIPLETS, epochs=3)
        expected = model.encode(["甲公司营业收入"], adapted=True)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder) / "model.pkl"
            model.save(path)
            loaded = DomainEmbeddingModel.load(path)
            np.testing.assert_allclose(loaded.encode(["甲公司营业收入"], adapted=True), expected)


if __name__ == "__main__":
    unittest.main()
