# ملف test_preprocessing.py
# tests/test_preprocessing.py
"""
محتوى الملف
"""
import pytest
from src.preprocessing import ArabicPreprocessor

# ننشئ كائن من المعالج لاستخدامه في الاختبارات
@pytest.fixture
def processor():
    return ArabicPreprocessor()

def test_remove_diacritics(processor):
    """اختبار إزالة الحركات (التشكيل)"""
    text_with_diacritics = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
    expected = "بسم الله الرحمن الرحيم"
    # ملاحظة: دالة re.sub قد تترك مسافات أو رموز صغيرة، نتأكد من التطابق
    result = processor.remove_diacritics(text_with_diacritics)
    assert result.strip() == expected

def test_normalize_arabic(processor):
    """اختبار توحيد الأحرف (الألف والياء والهاء)"""
    text_to_normalize = "إِيَّاكَ نَعْبُدُ"
    # بعد المعالجة: إ -> ا ، حذف التشكيل
    # سنطبق العمليتين معاً كما يفعل الكود الرئيسي
    cleaned = processor.remove_diacritics(text_to_normalize)
    normalized = processor.normalize_arabic(cleaned)
    
    assert "اياك" in normalized
    assert "إ" not in normalized

def test_empty_input(processor):
    """اختبار التعامل مع النصوص الفارغة"""
    assert processor.remove_diacritics("") == ""