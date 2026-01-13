# 🕌 NLP Pipeline for Surat Al-Fatiha
![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)

An advanced, modular Natural Language Processing (NLP) pipeline designed to analyze Quranic texts. This project demonstrates the transition from classical statistical methods to modern deep learning embeddings using **AraBERT**.



## 📂 Project Structure
The project architecture adheres to the **Production-Ready** standards, separating concerns into logical modules:

```text
nlp_arabic_fatiha/
├── data/
│   ├── raw/                # Original Quranic text (with diacritics)
│   └── processed/          # Cleaned and normalized text
├── src/                    # Core logic (Preprocessing, Models)
├── models/                 # Saved Word2Vec and BERT .npy files
├── tests/                  # Automated Unit Tests (Pytest)
├── notebooks/              # Research and Visualization
└── main.py                 # Pipeline Orchestrator
```
### 🚀 Key Features & Techniques
## 🛠 1. Advanced Arabic Preprocessing
Diacritics Removal: Specialized Regex for Quranic marks (Harakaat) and small symbols (Small Alef, etc.).

**Normalization:** Unifying Arabic character variants (e.g., إ، أ، آ -> ا).

**Tokenization:** Segmenting text for statistical and neural analysis.

### 📊 2. Text Representation
**Classical:** Bag of Words (BoW) & TF-IDF for statistical importance.

**Neural:** Word2Vec (Skip-gram/CBOW) for semantic relationships.

State-of-the-Art: AraBERT (v02) for deep contextual embeddings, capturing the spiritual and linguistic depth of the Quranic verses.

### 🛠 Installation & Usage
**Option 1**: Docker (Recommended)
Ensure Docker is installed and run:

docker-compose up --build

**Option 2:** Local Setup
1. Clone the Repo:

git clone [https://github.com/aiham711127/nlp_arabic_fatiha.git]

2. Setup Environment:
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt

3. Run Tests: 
python -m pytest tests/

4. Execute Pipeline:
python main.py

### 🧪 Testing (CI/CD)
This project uses Pytest for ensuring data integrity during preprocessing. The pipeline is ready for integration with GitHub Actions to automate testing on every push.

### 👤 Author
Aiham Albukaiti

Machine Learning & NLP Enthusiast & Deep Learning

https://github.com/aiham711127

www.linkedin.com/in/aiham-albukhaiti-9165693a5


### 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.


