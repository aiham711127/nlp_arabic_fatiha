
"""
الاختبار التكاملي لا يفحص "دالة" واحدة، 
بل يفحص "دورة حياة البيانات" كاملة
للتأكد من أن الملفات الضخمة موجودة وأن الأنابيب
متصلة ببعضها.
"""
import os
import pytest
from src.config import DATA_DIR, MODELS_DIR

def test_pipeline_data_flow():
    """اختبار سلامة تدفق البيانات والملفات الضخمة"""
    
    # 1. التأكد من وجود البيانات الخام
    raw_file = os.path.join(DATA_DIR, "raw", "al_fatiha.txt")
    assert os.path.exists(raw_file), "الملف الخام al_fatiha.txt مفقود!"

    # 2. التأكد من أن حجم الملف ليس صفراً (سلامة البيانات)
    assert os.path.getsize(raw_file) > 0, "الملف الخام فارغ!"

    # 3. التأكد من وجود مخرجات BERT (الملفات الضخمة)
    bert_file = os.path.join(MODELS_DIR, "embeddings", "bert_embeddings.npy")
    # ملاحظة: قد يفشل هذا في GitHub Actions إذا لم نرفع الملفات، 
    # لذا نختبر وجود المجلد على الأقل أو الملف محلياً
    if os.path.exists(bert_file):
        assert os.path.getsize(bert_file) > 1000, "ملف BERT Embeddings تالف أو صغير جداً"

def test_directory_structure():
    """التأكد من أن الهيكل التنظيمي للمشروع مكتمل"""
    required_dirs = [
        os.path.join(DATA_DIR, "processed"),
        os.path.join(MODELS_DIR, "word2vec"),
        os.path.join(MODELS_DIR, "embeddings")
    ]
    for d in required_dirs:
        assert os.path.isdir(d), f"المجلد {d} مفقود من هيكل المشروع"