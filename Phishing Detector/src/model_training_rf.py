"""
Random Forest Model Training Module

This module implements the training pipeline for a Random Forest classifier
for phishing email detection. It handles data loading, feature engineering,
model training, evaluation, and artifact saving.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, chi2
import os

from data_preprocessing import load_and_clean
from feature_engineering import build_tfidf_vectorizer, vectorize_with_extras_batch

def train_and_evaluate(
    data_path,
    model_output_dir='artifacts',
    test_size=0.2,
    random_state=42,
    n_features=521
):
    # Load and split data
    X_train, X_test, y_train, y_test = load_and_clean(
        data_path,
        test_size=test_size,
        random_state=random_state,
    )

    # Initialize TF-IDF vectorizer and extract features
    vec = build_tfidf_vectorizer(
        max_features=521,
        ngram_range=(1, 3),
        stop_words='english'
    )
    X_train_feats, X_test_feats = vectorize_with_extras_batch(
        vec,
        X_train['clean_body'],
        X_test['clean_body']
    )

    # Perform feature selection
    print("\nPerforming feature selection...")
    n_actual_features = X_train_feats.shape[1]
    n_features = min(n_features, n_actual_features)
    print(f"Total features available: {n_actual_features}")
    print(f"Selecting top {n_features} features")
    
    selector = SelectKBest(chi2, k=n_features)
    X_train_selected = selector.fit_transform(X_train_feats, y_train)
    X_test_selected = selector.transform(X_test_feats)

    # Initialize and train Random Forest classifier
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=random_state,
        max_features='sqrt'
    )
    clf.fit(X_train_selected, y_train)

    # Evaluate model performance
    train_preds = clf.predict(X_train_selected)
    print("\nTRAIN:\n", classification_report(y_train, train_preds))
    test_preds = clf.predict(X_test_selected)
    print("\nTEST:\n", classification_report(y_test, test_preds))

    # Save model artifacts
    os.makedirs(model_output_dir, exist_ok=True)
    vec_path = os.path.join(model_output_dir, 'tfidf_vectorizer.pkl')
    model_path = os.path.join(model_output_dir, 'phish_detector.pkl')
    selector_path = os.path.join(model_output_dir, 'feature_selector.pkl')

    with open(vec_path, 'wb') as f:
        pickle.dump(vec, f)
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    with open(selector_path, 'wb') as f:
        pickle.dump(selector, f)

    print(f"\nSaved artifacts to {model_output_dir}:")
    print(f"- Vectorizer: {vec_path}")
    print(f"- Model: {model_path}")
    print(f"- Feature selector: {selector_path}")

if __name__ == "__main__":
    import sys
    data_csv = sys.argv[1] if len(sys.argv) > 1 else './data/Kaggle.csv'
    train_and_evaluate(data_csv)
