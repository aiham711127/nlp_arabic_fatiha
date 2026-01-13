# ملف word2vec_model.py
# src/word2vec_model.py

"""
Word2Vec - أول نموذج ذكي لتمثيل الكلمات

المبدأ: "الكلمات التي تظهر في سياقات متشابهة لها معانٍ متشابهة"

نوعان:
1. CBOW: تخمين الكلمة من السياق
2. Skip-gram: تخمين السياق من الكلمة
"""
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import numpy as np
from typing import List, Dict

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from src.config import CLEANED_FILE, MODELS_DIR, W2V_VECTOR_SIZE, W2V_WINDOW

def train_word2vec():
    print("\n--- Training Word2Vec ---")
    # 1. تحميل البيانات وتقطيعها لكلمات
    with open(CLEANED_FILE, 'r', encoding='utf-8') as f:
        corpus = [word_tokenize(line.strip()) for line in f.readlines() if line.strip()]

    # 2. بناء النموذج
    # min_count=1 لأن البيانات صغيرة جداً، عادة نجعله 5 أو أكثر
    # model = Word2Vec(sentences=corpus, 
    #                  vector_size=W2V_VECTOR_SIZE, 
    #                  window=W2V_WINDOW, 
    #                  min_count=1, 
    #                  workers=4)
    # اجعل min_count = 1 لكي يقبل الكلمات حتى لو تكررت مرة واحدة فقط
    model = Word2Vec(sentences= corpus, vector_size=100, window=5, min_count=1, workers=4)
    # 3. الحفظ
    model_path = MODELS_DIR / "word2vec" / "fatiha.model"
    model.save(str(model_path))
    print(f"✅ Model saved to {model_path}")
    
    return model

if __name__ == "__main__":
    model = train_word2vec()
    # تجربة: المتجه الخاص بكلمة "الله"
    if "الله" in model.wv:
        print("Vector for 'الله':", model.wv["الله"][:10]) # طباعة أول 10 أبعاد
    elif "لله" in model.wv: # بعد المعالجة قد تكون "لله"
         print("Vector for 'لله':", model.wv["لله"][:10])