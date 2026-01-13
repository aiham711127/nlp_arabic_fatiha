import sys
from src.preprocessing import ArabicPreprocessor
from src.bow_tfidf import apply_bow, apply_tfidf, load_data
from src.word2vec_model import train_word2vec
from src.bert_embeddings import BertEmbedder
from src.config import FATIHA_FILE, CLEANED_FILE, MODELS_DIR
import numpy as np

def main():
    print("🚀 Starting NLP Pipeline for Surat Al-Fatiha...")

    # 1. Preprocessing
    print("\n[1] Preprocessing...")
    preprocessor = ArabicPreprocessor()
    preprocessor.preprocess_file(FATIHA_FILE, CLEANED_FILE)
    corpus = load_data()

    # 2. BOW & TF-IDF
    print("\n[2] Classical Representations (BOW & TF-IDF)...")
    bow_df = apply_bow(corpus)
    tfidf_df = apply_tfidf(corpus)
    # حفظ النتائج كـ CSV للمراجعة
    bow_df.to_csv(MODELS_DIR / "bow_results.csv", index=False)
    tfidf_df.to_csv(MODELS_DIR / "tfidf_results.csv", index=False)

    # 3. Word2Vec
    print("\n[3] Word2Vec Training...")
    w2v_model = train_word2vec()

    # 4. BERT
    print("\n[4] BERT Embeddings extraction...")
    bert = BertEmbedder()
    embeddings = bert.get_embeddings(corpus)
    
    print("\n🎉 Pipeline Completed Successfully!")
    print(f"Check '{MODELS_DIR}' for outputs.")

if __name__ == "__main__":
    main()