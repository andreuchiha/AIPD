# Phishing Email Detection System

A machine learning-based system that detects phishing emails by analyzing email content, URLs, and writing style. The system uses multiple ML models (Random Forest, Gradient Boosting, Logistic Regression, and Naive Bayes) and provides explanations for its predictions.

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download required models:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words'); nltk.download('wordnet')"
python -m spacy download en_core_web_sm
```

3. Run the web interface:
```bash
cd web
python app.py
```
Then open your browser to `http://localhost:5000`

## Command Line Usage

To analyze an email file:
```bash
python src/run_inf.py
```

## Training Models

To train a new model:
```bash
python src/model_training_rf.py [path_to_data.csv]  # Random Forest
python src/model_training_gb.py [path_to_data.csv]  # Gradient Boosting
python src/model_training_lr.py [path_to_data.csv]  # Logistic Regression
python src/model_training_nb.py [path_to_data.csv]  # Naive Bayes
```

## Input Data Format

The system expects CSV files with:
- `clean_body`: Email text content
- `label`: 0 for legitimate, 1 for phishing 