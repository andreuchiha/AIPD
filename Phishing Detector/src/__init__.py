"""
Phishing Detection Package

This package provides a comprehensive solution for detecting phishing emails
using machine learning. It includes modules for data preprocessing, feature
engineering, model training, and inference with explanations.
"""

# Make src directory a Python package
from .inference import predict_email, load_artifacts
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