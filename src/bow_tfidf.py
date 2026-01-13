# ملف bow_tfidf.py
# src/bow_tfidf.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from src.config import CLEANED_FILE

def load_data():
    with open(CLEANED_FILE, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def apply_bow(corpus):
    print("\n--- Bag of Words (BoW) ---")
    # نستخدم محلل بسيط يعتمد على المسافات لأننا نظفنا النص مسبقاً
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    
    # تحويل لجدول للعرض
    df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    return df

def apply_tfidf(corpus):
    print("\n--- TF-IDF ---")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    
    df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    return df

if __name__ == "__main__":
    corpus = load_data()
    bow_df = apply_bow(corpus)
    print("BOW Shape:", bow_df.shape)
    print(bow_df.head())

    tfidf_df = apply_tfidf(corpus)
    print("TF-IDF Shape:", tfidf_df.shape)
    print(tfidf_df.head())