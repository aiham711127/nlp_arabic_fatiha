# ملف bert_embeddings.py
# src/bert_embeddings.py

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from src.config import CLEANED_FILE, MODELS_DIR, BERT_MODEL_NAME

class BertEmbedder:
    def __init__(self):
        print(f"Loading BERT Model: {BERT_MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        self.model = AutoModel.from_pretrained(BERT_MODEL_NAME)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def get_embeddings(self, text_list):
        self.model.eval()
        embeddings = []
        
        print("Generating BERT Embeddings...")
        with torch.no_grad():
            for text in text_list:
                # التجهيز (Tokenization)
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # التمرير للنموذج
                outputs = self.model(**inputs)
                
                # استخراج الطبقة المخفية الأخيرة (CLS Token) لتمثيل الجملة كاملة
                # [0][:, 0, :] تعني: آخر طبقة مخفية، كل الدفعات، التوكن الأول (CLS)، كل الأبعاد
                cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embedding)
        
        return np.vstack(embeddings)
if __name__ == "__main__":
    import os
    # 1. التأكد من قراءة الملف
    if not os.path.exists(CLEANED_FILE):
        print(f"❌ Error: {CLEANED_FILE} not found!")
    else:
        with open(CLEANED_FILE, 'r', encoding='utf-8') as f:
            corpus = [line.strip() for line in f.readlines() if line.strip()]

        if not corpus:
            print("❌ Error: Corpus is empty!")
        else:
            embedder = BertEmbedder()
            emb_matrix = embedder.get_embeddings(corpus)
            
            # 2. التأكد من وجود المجلد (هذا هو السر!)
            save_dir = MODELS_DIR / "embeddings"
            os.makedirs(save_dir, exist_ok=True)
            
            save_path = save_dir / "bert_embeddings.npy"
            
            # 3. الحفظ الفعلي
            np.save(str(save_path), emb_matrix)
            
            print(f"✅ BERT Embeddings shape: {emb_matrix.shape}")
            print(f"✅ File saved successfully. Size: {os.path.getsize(save_path)} bytes")