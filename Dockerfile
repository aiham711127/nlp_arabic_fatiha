# 1. استخدام صورة بايثون الرسمية الخفيفة
FROM python:3.9-slim

# 2. منع بايثون من كتابة ملفات pyc وتفعيل السجلات
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. تحديد مجلد العمل داخل الحاوية
WORKDIR /app

# 4. تثبيت أدوات النظام الضرورية (لبعض مكتبات الـ ML)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. نسخ ملف المتطلبات وتثبيتها
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt stopwords

# 6. نسخ باقي كود المشروع
COPY . .

# 7. الأمر الافتراضي عند تشغيل الحاوية
CMD ["python", "main.py"]