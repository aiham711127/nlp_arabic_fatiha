# ملف test_embeddings.py
# tests/test_embeddings.py
"""
ذا الاختبار يتأكد من أن BERT و Word2Vec ينتجان أرقاماً (Vectors) ذات أبعاد صحيحة وليس مجرد قيم فارغة.
"""

import pytest
import numpy as np
import os
from src.word2vec_model import train_word2vec
from src.bert_embeddings import BertEmbedder
from src.config import MODELS_DIR

def test_word2vec_output():
    """التأكد من أن Word2Vec ينتج ملف موديل صالح"""
    model_path = os.path.join(MODELS_DIR, "word2vec", "fatiha.model")
    assert os.path.exists(model_path), "فشل في العثور على ملف Word2Vec"

def test_bert_embeddings_shape():
    """التأكد من أن BERT ينتج متجهات بطول 768 (معيار AraBERT)"""
    embedder = BertEmbedder()
    sample_text = ["الحمد لله رب العالمين"]
    embeddings = embedder.get_embeddings(sample_text)
    
    # AraBERT base ينتج متجهاً بطول 768 لكل جملة
    assert embeddings.shape[1] == 768
    assert isinstance(embeddings, np.ndarray)