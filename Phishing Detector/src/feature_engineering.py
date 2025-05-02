"""
Feature Engineering Module

This module implements comprehensive feature extraction and engineering for email classification.
It includes text processing, linguistic analysis, and various feature extraction techniques.

Features:
- TF-IDF Features: Text content analysis using TF-IDF vectorization
- POS Features: Part of Speech tagging (nouns, verbs, adjectives, etc.)
- NER Features: Named Entity Recognition (persons, organizations, locations, etc.)
- Stylometric Features: Writing style analysis (word length, sentence length, etc.)
- URL Analysis Features: URL safety indicators (suspicious patterns, SSL, etc.)
- Urgency Score: Message urgency assessment
"""

import os
import pickle
import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from urllib.parse import urlparse
import tld
from typing import Dict, List, Tuple
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob

# Handle both package and standalone imports
try:
    # Try relative imports (for package usage)
    from .urgency_scoring import extract_urgency_features
    from .url_analysis import analyze_urls_batch
    from .data_preprocessing import clean_text, load_and_clean
except ImportError:
    # Fall back to absolute imports (for standalone usage)
    from urgency_scoring import extract_urgency_features
    from url_analysis import analyze_urls_batch
    from data_preprocessing import clean_text, load_and_clean

# URL regex pattern for detection
URL_PATTERN = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+|www\.(?:[-\w.]|(?:%[\da-fA-F]{2}))+'

# Lazy-load spaCy model with only required components
_nlp = None
def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load('en_core_web_sm', disable=['parser'])
    return _nlp

def extract_spacy_features_batch(texts):
    """
    Extract POS and NER features using spaCy's batch processing.
    Returns a DataFrame with feature vectors for each text.
    """
    nlp = get_nlp()
    pos_tags   = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN']
    ent_labels = ['PERSON', 'ORG', 'GPE', 'URL', 'DATE', 'MONEY']

    rows = []
    for doc in tqdm(nlp.pipe(texts, batch_size=50), total=len(texts)):
        total_tokens = len(doc)
        pos_counts = {pos: 0 for pos in pos_tags}
        for token in doc:
            if token.pos_ in pos_counts:
                pos_counts[token.pos_] += 1
        for pos in pos_counts:
            pos_counts[pos] = pos_counts[pos] / total_tokens if total_tokens else 0.0
        ent_counts = {label: 0 for label in ent_labels}
        for ent in doc.ents:
            if ent.label_ in ent_counts:
                ent_counts[ent.label_] += 1
        rows.append({**pos_counts, **ent_counts})
    return pd.DataFrame(rows)

def extract_stylometric_features_batch(texts):
    """
    Extract stylometric features in batch processing mode.
    Features include uppercase ratio, exclamation count, average word length,
    and URL-related metrics.
    """
    data = []
    for text in texts:
        tokens    = text.split()
        num_tokens= len(tokens)
        uppercase = sum(1 for t in tokens if t.isupper())
        excls     = text.count('!')
        avg_len   = np.mean([len(t) for t in tokens]) if num_tokens else 0.0
        
        urls = re.findall(URL_PATTERN, text)
        url_count = len(urls)
        
        data.append({
            'uppercase_ratio': uppercase / num_tokens if num_tokens else 0.0,
            'exclamation_count': excls,
            'avg_word_length': avg_len,
            'url_count': url_count,
            'url_ratio': url_count / num_tokens if num_tokens else 0.0
        })
    return pd.DataFrame(data)

def extract_urgency_features_batch(texts):
    """
    Extract urgency-related features in batch processing mode.
    """
    data = []
    for text in texts:
        urgency_features = extract_urgency_features(text)
        data.append(urgency_features)
    return pd.DataFrame(data)

def build_tfidf_vectorizer(max_features=521, ngram_range=(1,2), stop_words='english'):
    """
    Create and configure a TF-IDF vectorizer for text feature extraction.
    
    Args:
        max_features: Number of most frequent words to keep
        ngram_range: Range of n-grams to use
        stop_words: Language for stop words
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words=stop_words,
        min_df=2,
        max_df=0.95
    )

def vectorize_train_test(vectorizer, X_train_texts, X_test_texts):
    """
    Perform basic TF-IDF vectorization on training and test texts.
    """
    X_train_tfidf = vectorizer.fit_transform(X_train_texts)
    X_test_tfidf  = vectorizer.transform(X_test_texts)
    return X_train_tfidf, X_test_tfidf

def vectorize_with_extras_batch(vectorizer, X_train_texts, X_test_texts):
    """
    Comprehensive feature extraction pipeline combining:
    1. TF-IDF vectorization
    2. spaCy POS/NER features
    3. Stylometric features
    4. Urgency features
    Returns concatenated feature matrices for training and test sets.
    """
    X_train_tfidf = vectorizer.fit_transform(X_train_texts)
    X_test_tfidf  = vectorizer.transform(X_test_texts)

    spacy_train = extract_spacy_features_batch(X_train_texts)
    spacy_test  = extract_spacy_features_batch(X_test_texts)

    style_train = extract_stylometric_features_batch(X_train_texts)
    style_test  = extract_stylometric_features_batch(X_test_texts)

    urgency_train = extract_urgency_features_batch(X_train_texts)
    urgency_test  = extract_urgency_features_batch(X_test_texts)

    X_train_final = hstack([
        X_train_tfidf,
        spacy_train.values,
        style_train.values,
        urgency_train.values
    ])
    X_test_final  = hstack([
        X_test_tfidf,
        spacy_test.values,
        style_test.values,
        urgency_test.values
    ])

    return X_train_final, X_test_final

def featurize_single(text, vectorizer):
    """
    Extract features for a single text sample.
    Combines TF-IDF, spaCy, stylometric, and urgency features.
    Returns a sparse feature vector.
    """
    tfidf_vec = vectorizer.transform([text])
    spacy_df  = extract_spacy_features_batch([text])
    style_df  = extract_stylometric_features_batch([text])
    urgency_df = extract_urgency_features_batch([text])

    return hstack([
        tfidf_vec,
        spacy_df.values,
        style_df.values,
        urgency_df.values
    ])

def classify_by_max_score(text, vectorizer, clf, chunk_size=500):
    """
    Process text in chunks and classify based on maximum prediction probability.
    Returns 1 if any chunk's probability exceeds 0.5, 0 otherwise.
    """
    tokens     = text.split()
    max_proba  = 0.0
    for i in range(0, len(tokens), chunk_size):
        snippet = " ".join(tokens[i : i + chunk_size])
        X_chunk = featurize_single(snippet, vectorizer)
        proba   = clf.predict_proba(X_chunk)[0, 1]
        max_proba = max(max_proba, proba)
    return 1 if max_proba >= 0.5 else 0

def feature_summary(X_tfidf, feature_names, top_n=10):
    """
    Generate summary of top features by average TF-IDF weight.
    """
    mean_vals = np.asarray(X_tfidf.mean(axis=0)).ravel()
    top_idxs  = mean_vals.argsort()[::-1][:top_n]
    return pd.DataFrame({
        'feature': feature_names[top_idxs],
        'avg_tfidf': mean_vals[top_idxs]
    })

# Initialize NLTK components
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Urgency indicators for feature extraction
URGENCY_INDICATORS = {
    'urgent', 'immediately', 'asap', 'right away', 'hurry', 'quick', 'fast',
    'emergency', 'critical', 'important', 'action required', 'attention needed',
    'verify', 'confirm', 'validate', 'check', 'review', 'update', 'secure',
    'suspicious', 'unusual', 'unexpected', 'warning', 'alert', 'notice',
}

def preprocess_text(text: str) -> str:
    """
    Preprocess text by tokenizing, removing stopwords, and lemmatizing.
    """
    # Tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    return ' '.join(tokens)

def extract_pos_features(text: str) -> Dict[str, int]:
    """
    Extract POS (Part of Speech) features.
    """
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    
    # Count different parts of speech
    pos_counts = {
        'NN': 0,  # Nouns
        'VB': 0,  # Verbs
        'JJ': 0,  # Adjectives
        'RB': 0,  # Adverbs
        'PRP': 0,  # Pronouns
        'DT': 0,  # Determiners
        'IN': 0,  # Prepositions
        'CC': 0,  # Conjunctions
        'CD': 0,  # Numbers
        'SYM': 0  # Symbols
    }
    
    for word, tag in pos_tags:
        if tag.startswith('NN'):
            pos_counts['NN'] += 1
        elif tag.startswith('VB'):
            pos_counts['VB'] += 1
        elif tag.startswith('JJ'):
            pos_counts['JJ'] += 1
        elif tag.startswith('RB'):
            pos_counts['RB'] += 1
        elif tag.startswith('PRP'):
            pos_counts['PRP'] += 1
        elif tag.startswith('DT'):
            pos_counts['DT'] += 1
        elif tag.startswith('IN'):
            pos_counts['IN'] += 1
        elif tag.startswith('CC'):
            pos_counts['CC'] += 1
        elif tag.startswith('CD'):
            pos_counts['CD'] += 1
        elif tag.startswith('SYM'):
            pos_counts['SYM'] += 1
    
    return pos_counts

def extract_ner_features(text: str) -> Dict[str, int]:
    """
    Extract Named Entity Recognition (NER) features.
    """
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    ner_tags = ne_chunk(pos_tags)
    
    # Count different named entities
    ner_counts = {
        'PERSON': 0,
        'ORGANIZATION': 0,
        'GPE': 0,  # Geo-Political Entities
        'MONEY': 0,
        'PERCENT': 0,
        'DATE': 0,
        'TIME': 0
    }
    
    for chunk in ner_tags:
        if hasattr(chunk, 'label'):
            if chunk.label() in ner_counts:
                ner_counts[chunk.label()] += 1
    
    return ner_counts

def calculate_urgency_score(text: str) -> float:
    """
    Calculate urgency score based on presence of urgency indicators.
    """
    words = set(word_tokenize(text.lower()))
    urgency_words = words.intersection(URGENCY_INDICATORS)
    return len(urgency_words) / len(words) if words else 0.0

def extract_stylometric_features(text: str) -> Dict[str, float]:
    """
    Extract stylometric features.
    """
    # Calculate average word length
    words = text.split()
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    
    # Calculate average sentence length
    sentences = text.split('.')
    avg_sentence_length = sum(len(sent.split()) for sent in sentences) / len(sentences) if sentences else 0
    
    # Calculate ratio of uppercase letters
    total_chars = len(text)
    uppercase_chars = sum(1 for c in text if c.isupper())
    uppercase_ratio = uppercase_chars / total_chars if total_chars > 0 else 0
    
    # Calculate ratio of special characters
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    special_char_ratio = special_chars / total_chars if total_chars > 0 else 0
    
    return {
        'avg_word_length': avg_word_length,
        'avg_sentence_length': avg_sentence_length,
        'uppercase_ratio': uppercase_ratio,
        'special_char_ratio': special_char_ratio
    }

def extract_features(texts: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Extract all features from a list of texts.
    """
    # Preprocess texts
    preprocessed_texts = [preprocess_text(text) for text in texts]
    
    # TF-IDF features
    tfidf = TfidfVectorizer(max_features=300)
    tfidf_features = tfidf.fit_transform(preprocessed_texts)
    feature_names = tfidf.get_feature_names_out()
    
    # Additional features
    additional_features = []
    for text in texts:
        # POS features
        pos_features = extract_pos_features(text)
        
        # NER features
        ner_features = extract_ner_features(text)
        
        # Stylometric features
        stylometric_features = extract_stylometric_features(text)
        
        # URL analysis features
        url_features = analyze_urls_batch([text])[0]
        
        # Combine all features
        features = {
            **pos_features,
            **ner_features,
            **stylometric_features,
            **url_features,
            'urgency_score': calculate_urgency_score(text)
        }
        
        additional_features.append(list(features.values()))
    
    # Convert to numpy array
    additional_features = np.array(additional_features)
    
    # Combine TF-IDF and additional features
    combined_features = np.hstack([tfidf_features.toarray(), additional_features])
    
    # Update feature names
    feature_names = list(feature_names) + list(pos_features.keys()) + \
                   list(ner_features.keys()) + list(stylometric_features.keys()) + \
                   list(url_features.keys()) + ['urgency_score']
    
    return combined_features, feature_names

if __name__ == "__main__":
    # Example pipeline
    X_train, X_test, y_train, y_test = load_and_clean(
        './data/Kaggle.csv',
        text_col='clean_body',
        label_col='label',
        test_size=0.2,
        random_state=42,
    )

    # TF–IDF only
    vec = build_tfidf_vectorizer()
    X_tr, X_te = vectorize_train_test(vec,
                                      X_train['clean_body'],
                                      X_test['clean_body'])

    names = vec.get_feature_names_out()
    print("Top TF–IDF features:\n", feature_summary(X_tr, names, top_n=10))

    # Persist vectorizer
    artifacts_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), '..', 'artifacts')
    )
    os.makedirs(artifacts_dir, exist_ok=True)
    vect_path = os.path.join(artifacts_dir, 'tfidf_vectorizer.pkl')
    with open(vect_path, 'wb') as f:
        pickle.dump(vec, f)
    print(f"Vectorizer saved to {vect_path}")
