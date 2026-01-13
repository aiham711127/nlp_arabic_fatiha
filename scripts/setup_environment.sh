#!/bin/bash

echo "🚀 Setting up the environment..."

# 1. إنشاء بيئة افتراضية إذا لم تكن موجودة
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# 2. تفعيل البيئة
source venv/bin/activate

# 3. تحديث pip وتثبيت المكتبات
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. تحميل بيانات NLTK الضرورية
python -m nltk.downloader punkt stopwords

echo "✅ Environment setup complete! Run 'source venv/bin/activate' to start."