"""
Inference Module

This module implements the prediction and explanation pipeline for the phishing detection system.
It includes functions for loading model artifacts, making predictions, and generating
explanations using LIME and urgency scoring.
"""

import pickle
import os
from lime.lime_text import LimeTextExplainer
import numpy as np
from nltk.corpus import stopwords
import nltk
from scipy.sparse import hstack

# Handle both package and standalone imports
try:
    # Try relative imports (for package usage)
    from .data_preprocessing import clean_text
    from .feature_engineering import (
        featurize_single, 
        classify_by_max_score, 
        vectorize_with_extras_batch, 
        vectorize_train_test,
        extract_spacy_features_batch,
        extract_stylometric_features_batch,
        extract_urgency_features_batch
    )
    from .urgency_scoring import get_urgency_explanation
except ImportError:
    # Fall back to absolute imports (for standalone usage)
    from data_preprocessing import clean_text
    from feature_engineering import (
        featurize_single, 
        classify_by_max_score, 
        vectorize_with_extras_batch, 
        vectorize_train_test,
        extract_spacy_features_batch,
        extract_stylometric_features_batch,
        extract_urgency_features_batch
    )
    from urgency_scoring import get_urgency_explanation

# Download stopwords if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Get English stopwords
STOP_WORDS = set(stopwords.words('english'))
# Add common email-specific words to stopwords
STOP_WORDS.update(['dear', 'sir', 'madam', 'hello', 'hi', 'regards', 'thanks', 'thank', 'you', 'your', 'yours', 'best', 'kind', 'regards', 'and'])

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths to saved model artifacts
VEC_FILE_RF = os.path.join(PROJECT_ROOT, "artifacts", "tfidf_vectorizer.pkl")
MODEL_FILE_RF = os.path.join(PROJECT_ROOT, "artifacts", "phish_detector.pkl")
SELECTOR_FILE_RF = os.path.join(PROJECT_ROOT, "artifacts", "feature_selector.pkl")
VEC_FILE_NB = os.path.join(PROJECT_ROOT, "artifacts", "tfidf_vectorizer_nb.pkl")
MODEL_FILE_NB = os.path.join(PROJECT_ROOT, "artifacts", "phish_detector_nb.pkl")
SELECTOR_FILE_NB = os.path.join(PROJECT_ROOT, "artifacts", "feature_selector_nb.pkl")
FEATURES_FILE_NB = os.path.join(PROJECT_ROOT, "artifacts", "selected_features_nb.pkl")
VEC_FILE_LR = os.path.join(PROJECT_ROOT, "artifacts", "tfidf_vectorizer_lr.pkl")
MODEL_FILE_LR = os.path.join(PROJECT_ROOT, "artifacts", "phish_detector_lr.pkl")
SELECTOR_FILE_LR = os.path.join(PROJECT_ROOT, "artifacts", "feature_selector_lr.pkl")
FEATURES_FILE_LR = os.path.join(PROJECT_ROOT, "artifacts", "selected_features_lr.pkl")
VEC_FILE_GB = os.path.join(PROJECT_ROOT, "artifacts", "tfidf_vectorizer_gb.pkl")
MODEL_FILE_GB = os.path.join(PROJECT_ROOT, "artifacts", "phish_detector_gb.pkl")
SELECTOR_FILE_GB = os.path.join(PROJECT_ROOT, "artifacts", "feature_selector_gb.pkl")
FEATURES_FILE_GB = os.path.join(PROJECT_ROOT, "artifacts", "selected_features_gb.pkl")

def load_artifacts(model_type='rf'):
    """
    Load the fitted TF-IDF vectorizer and trained classifier.
    
    Args:
        model_type: Model type to load ('rf', 'nb', 'lr', or 'gb')
    
    Returns:
        tuple: (vectorizer, classifier, feature_selector, selected_features)
    """
    if model_type.lower() == 'rf':
        vec_file = VEC_FILE_RF
        model_file = MODEL_FILE_RF
        selector_file = SELECTOR_FILE_RF
        selected_features = None
    elif model_type.lower() == 'nb':
        vec_file = VEC_FILE_NB
        model_file = MODEL_FILE_NB
        selector_file = SELECTOR_FILE_NB
        features_file = FEATURES_FILE_NB
        with open(features_file, "rb") as f:
            selected_features = pickle.load(f)
    elif model_type.lower() == 'lr':
        vec_file = VEC_FILE_LR
        model_file = MODEL_FILE_LR
        selector_file = SELECTOR_FILE_LR
        features_file = FEATURES_FILE_LR
        with open(features_file, "rb") as f:
            selected_features = pickle.load(f)
    elif model_type.lower() == 'gb':
        vec_file = VEC_FILE_GB
        model_file = MODEL_FILE_GB
        selector_file = SELECTOR_FILE_GB
        features_file = FEATURES_FILE_GB
        with open(features_file, "rb") as f:
            selected_features = pickle.load(f)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    with open(vec_file, "rb") as f:
        vec = pickle.load(f)
    with open(model_file, "rb") as f:
        clf = pickle.load(f)
    with open(selector_file, "rb") as f:
        selector = pickle.load(f)
    
    return vec, clf, selector, selected_features

def predict_proba(texts, vec, clf, selector=None, model_type='rf'):
    """
    Get prediction probabilities for input texts.
    Handles both single strings and lists of strings.
    
    Args:
        texts: Input text or list of texts
        vec: TF-IDF vectorizer
        clf: Trained classifier
        selector: Feature selector (optional)
        model_type: Model type ('rf', 'nb', 'lr', or 'gb')
    
    Returns:
        numpy.ndarray: Prediction probabilities
    """
    if isinstance(texts, str):
        texts = [texts]
    
    cleaned_texts = [clean_text(text) for text in texts]
    
    # Generate features based on model type
    if model_type.lower() == 'lr':
        features = [vec.transform([text]) for text in cleaned_texts]
        stacked_features = np.vstack([f.toarray() for f in features])
    else:
        # Extract all feature types
        tfidf_feat = vec.transform(cleaned_texts)
        spacy_feat = extract_spacy_features_batch(cleaned_texts)
        style_feat = extract_stylometric_features_batch(cleaned_texts)
        urgency_feat = extract_urgency_features_batch(cleaned_texts)
        
        # Combine all features
        stacked_features = hstack([
            tfidf_feat,
            spacy_feat.values,
            style_feat.values,
            urgency_feat.values
        ]).toarray()
    
    # Apply feature selection if provided
    if selector is not None:
        try:
            # Check if feature count matches selector's expected input
            if stacked_features.shape[1] != selector.n_features_in_:
                print(f"Warning: Feature count mismatch. Got {stacked_features.shape[1]} features, "
                      f"but selector expects {selector.n_features_in_} features.")
                
                # If we have more features than expected, select only the needed ones
                if stacked_features.shape[1] > selector.n_features_in_:
                    stacked_features = stacked_features[:, :selector.n_features_in_]
                # If we have fewer features, pad with zeros
                else:
                    padding = np.zeros((stacked_features.shape[0], 
                                      selector.n_features_in_ - stacked_features.shape[1]))
                    stacked_features = np.hstack([stacked_features, padding])
            
            # Apply feature selection
            stacked_features = selector.transform(stacked_features)
        except Exception as e:
            print(f"Warning: Feature selection failed with error: {str(e)}")
            # Continue with original features if selection fails
            pass
    
    return clf.predict_proba(stacked_features)

def filter_stop_words(explanation):
    """
    Filter out stop words from LIME explanation.
    
    Args:
        explanation: List of (word, weight) tuples
    
    Returns:
        list: Filtered explanation with stop words removed
    """
    return [(word, weight) for word, weight in explanation 
            if word.lower() not in STOP_WORDS]

def explain_prediction(raw_email: str, vec, clf, selector=None, selected_features=None, model_type='rf', num_features=10, num_samples=500):
    """
    Generate LIME explanation for model prediction.
    
    Args:
        raw_email: Email text to explain
        vec: TF-IDF vectorizer
        clf: Trained classifier
        selector: Feature selector (optional)
        selected_features: List of selected feature names (optional)
        model_type: Model type ('rf', 'nb', 'lr', or 'gb')
        num_features: Number of features to show in explanation
        num_samples: Number of samples for LIME (lower = faster but less accurate)
    
    Returns:
        list: Top features contributing to the prediction
    """
    explainer = LimeTextExplainer(class_names=['Legitimate', 'Phishing'])
    
    exp = explainer.explain_instance(
        raw_email,
        lambda x: predict_proba(x, vec, clf, selector, model_type),
        num_features=num_features * 2,
        num_samples=num_samples
    )
    
    explanation = exp.as_list()
    filtered_explanation = filter_stop_words(explanation)
    filtered_explanation.sort(key=lambda x: abs(x[1]), reverse=True)
    
    if len(filtered_explanation) < num_features:
        original_words = {word for word, _ in filtered_explanation}
        additional = [(word, weight) for word, weight in explanation 
                     if word not in original_words][:num_features - len(filtered_explanation)]
        filtered_explanation.extend(additional)
    
    return filtered_explanation[:num_features]

def predict_email(
    raw_email: str,
    vec,
    clf,
    selector=None,
    selected_features=None,
    model_type='rf',
    explain: bool = False,
    num_samples: int = 500
):
    """
    Make prediction on a single email with optional explanations.
    
    Args:
        raw_email: Email text to analyze
        vec: TF-IDF vectorizer
        clf: Trained classifier
        selector: Feature selector (optional)
        selected_features: List of selected feature names (optional)
        model_type: Model type ('rf', 'nb', 'lr', or 'gb')
        explain: Whether to generate explanations
        num_samples: Number of samples for LIME explanation
    
    Returns:
        tuple: (prediction, confidence, lime_explanation, urgency_explanation)
    """
    cleaned = clean_text(raw_email)
    
    if model_type.lower() in ['rf', 'nb', 'lr', 'gb']:
        tfidf_feat = vec.transform([cleaned])
        spacy_feat = extract_spacy_features_batch([cleaned])
        style_feat = extract_stylometric_features_batch([cleaned])
        urgency_feat = extract_urgency_features_batch([cleaned])
        
        feat = hstack([
            tfidf_feat,
            spacy_feat.values,
            style_feat.values,
            urgency_feat.values
        ])
    
    if selector is not None:
        try:
            feat = selector.transform(feat)
        except:
            pass
    
    probs = clf.predict_proba(feat)[0]
    pred = clf.predict(feat)[0]
    confidence = probs[1] if pred == 1 else probs[0]
    
    lime_explanation = None
    urgency_explanation = None
    
    if explain:
        lime_explanation = explain_prediction(
            raw_email, vec, clf, selector, selected_features,
            model_type, num_features=10, num_samples=num_samples
        )
        urgency_explanation = get_urgency_explanation(raw_email)
    
    return pred, confidence, lime_explanation, urgency_explanation
