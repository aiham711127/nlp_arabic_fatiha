#!/bin/bash

#!/bin/bash

# تفعيل البيئة أولاً
source venv/bin/activate

# التأكد من المسار وتشغيل الكود الرئيسي
echo "🔥 Running NLP Pipeline..."
export PYTHONPATH=$PYTHONPATH:.
python main.py
