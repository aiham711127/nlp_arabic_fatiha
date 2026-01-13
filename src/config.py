# ملف config.py
# src/config.py

"""
ملف التكوين المركزي - من أهم ممارسات الهندسة الاحترافية
يحتوي جميع الثوابت والإعدادات في مكان واحد
"""
import os
from pathlib import Path

# تحديد المسار الجذري للمشروع
ROOT_DIR = Path(__file__).resolve().parent.parent

# مسارات البيانات
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"

# التأكد من وجود المجلدات
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
(MODELS_DIR / "word2vec").mkdir(exist_ok=True)
(MODELS_DIR / "embeddings").mkdir(exist_ok=True)

# ملف سورة الفاتحة
FATIHA_FILE = RAW_DATA_DIR / "al_fatiha.txt"
CLEANED_FILE = PROCESSED_DATA_DIR / "cleaned_fatiha.txt"

# إعدادات النماذج
BERT_MODEL_NAME = "aubmindlab/bert-base-arabertv02" # موديل ممتاز للعربية

W2V_VECTOR_SIZE = 100
W2V_WINDOW = 3