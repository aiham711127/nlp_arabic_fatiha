# ملف preprocessing.py
# src/preprocessing.py
"""
معالجة النص العربي - أهم خطوة في أي مشروع NLP
"""

import re
import nltk
from nltk.corpus import stopwords
from src.config import FATIHA_FILE, CLEANED_FILE

# تحميل مكتبات NLTK الضرورية
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    import re

class ArabicPreprocessor:
    def __init__(self):
        # قائمة الحركات والرموز القرآنية بدون مسافات بينها لضمان الدقة
        # تشمل: الفتحة، الضمة، الكسرة، السكون، الشدة، التنوين، الألف الخنجرية، وعلامات الوقف
        self.arabic_diacritics = re.compile(r'[\u064B-\u0652\u0670\u06D6-\u06ED]')

    def remove_diacritics(self, text):
        """إزالة التشكيل والرموز القرآنية بدقة"""
        if not text:
            return ""
        # إزالة الحركات
        text = re.sub(self.arabic_diacritics, '', text)
        # إزالة التطويل (التطويل ليس حركة بل حرف زينة)
        text = re.sub(r'ـ', '', text)
        return text

    def normalize_arabic(self, text):
        """توحيد أشكال الأحرف"""
        if not text:
            return ""
        # توحيد الألفات
        text = re.sub(r'[إأآ]', 'ا', text)
        # توحيد الياء والهاء والهمزات
        text = re.sub(r'ى', 'ي', text)
        text = re.sub(r'ة', 'ه', text)
        text = re.sub(r'[ؤئ]', 'ء', text)
        return text
    
    def preprocess_file(self, input_path, output_path):
        """قراءة الملف، تنظيفه بالكامل، وحفظه"""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if not line: 
                    continue
                
                # 1. إزالة التشكيل أولاً
                text = self.remove_diacritics(line)
                # 2. توحيد الأحرف ثانياً
                text = self.normalize_arabic(text)
                
                cleaned_lines.append(text)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(cleaned_lines))
            
            print(f"✅ Preprocessing complete. Saved to {output_path}")
            return cleaned_lines
            
        except FileNotFoundError:
            print(f"❌ Error: The file at {input_path} was not found.")
            return []
        
if __name__ == "__main__":
    processor = ArabicPreprocessor()
    processor.preprocess_file(FATIHA_FILE, CLEANED_FILE)