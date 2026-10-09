# -*- coding: utf-8 -*-
"""
金融领域 Embedding 推理封装 (微调后)
工单编号: 人工智能 NLP-RAG 项目-Embedding 模型微调任务
"""
import os, sys, json, logging
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_v11 as config

logger = logging.getLogger(__name__)


class FinancialEmbedding:
    """金融领域微调后的 Embedding"""

    def __init__(self, model_path: str = None):
        self.model_path = model_path or config.OUTPUT_DIR
        self.base_model = config.BASE_MODEL
        self.model = None
        self._loaded = False

    def load(self):
        """加载模型 (优先微调后)"""
        fine_path = self.model_path
        base_path = self.base_model

        # 检查是否有微调模型
        has_fine = os.path.exists(os.path.join(fine_path, "modules.json")) or \
                  os.path.exists(os.path.join(fine_path, "config.json"))
        has_skip = os.path.exists(os.path.join(fine_path, "skip_marker.json"))

        if has_fine:
            logger.info(f"加载微调后模型: {fine_path}")
            model_name_or_path = fine_path
        elif has_skip:
            logger.info(f"微调被跳过, 使用基础模型: {base_path}")
            model_name_or_path = base_path
        else:
            logger.info(f"无微调模型, 使用基础模型: {base_path}")
            model_name_or_path = base_path

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name_or_path)
            self._loaded = True
        except ImportError:
            logger.warning("sentence-transformers 未安装, 使用 TF-IDF")
            self._init_tfidf()
        except Exception as e:
            logger.warning(f"模型加载失败: {e}, 使用 TF-IDF")
            self._init_tfidf()

    def _init_tfidf(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        try:
            import jieba
            def tokenizer(text):
                return [t for t in jieba.cut(text) if t.strip()]
        except ImportError:
            def tokenizer(text):
                return list(text)
        self.tfidf = TfidfVectorizer(tokenizer=tokenizer, lowercase=False)
        self._loaded = True

    def encode(self, texts: List[str]):
        """编码文本"""
        if not self._loaded:
            self.load()

        if self.model is not None:
            return self.model.encode(texts, convert_to_tensor=False)
        else:
            return self.tfidf.fit_transform(texts).toarray()

    def get_model_info(self) -> dict:
        model_name = self.base_model
        is_fine = os.path.exists(os.path.join(self.model_path, "modules.json"))
        return {
            "base_model": model_name,
            "fine_tuned": is_fine,
            "fine_model_path": self.model_path if is_fine else None,
            "device": config.DEVICE,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    emb = FinancialEmbedding()
    emb.load()
    texts = ["武汉力源的控股股东是谁?", "注册资本是多少", "销售部有几个下属"]
    vecs = emb.encode(texts)
    print(f"编码完成: {len(vecs)} 条, 维度 {len(vecs[0])}")
    print(f"模型信息: {emb.get_model_info()}")
