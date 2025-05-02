"""
Gradient Boosting Model Training Module

This module implements the training pipeline for a Gradient Boosting classifier
for phishing email detection. It includes hyperparameter tuning, feature selection,
and comprehensive model evaluation across training, validation, and test sets.
"""

import os
import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd
from tqdm import tqdm

from data_preprocessing import load_and_clean
from feature_engineering import build_tfidf_vectorizer, vectorize_train_test, vectorize_with_extras_batch

def evaluate_model(clf, X, y, cv=3):
    """
    Evaluate model using cross-validation and print detailed metrics.
    """
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring='f1')
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    clf.fit(X, y)
    y_pred = clf.predict(X)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))
    
    return clf

def train_and_evaluate_gb(
    data_path,
    model_output_dir='artifacts',
    test_size=0.2,
    val_size=0.1,
    random_state=42,
    n_features=521
):
    print("Loading and preprocessing data...")
    # Load and split data into train, validation, and test sets
    X_train, X_test, y_train, y_test = load_and_clean(
        data_path,
        test_size=test_size + val_size,
        random_state=random_state,
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test,
        test_size=test_size/(test_size + val_size),
        random_state=random_state,
        stratify=y_test
    )

    print("Building TF-IDF vectorizer...")
    # Initialize TF-IDF vectorizer
    vec = build_tfidf_vectorizer(
        max_features=521,
        ngram_range=(1, 3),
        stop_words='english'
    )
    
    print("Vectorizing text data...")
    # Extract features including URL analysis and urgency scoring
    X_train_feats, X_test_feats = vectorize_with_extras_batch(
        vec,
        X_train['clean_body'],
        X_test['clean_body']
    )
    X_val_feats, _ = vectorize_with_extras_batch(
        vec,
        X_val['clean_body'],
        X_val['clean_body']
    )
    
    print("Performing feature selection...")
    # Select most informative features
    n_actual_features = X_train_feats.shape[1]
    n_features = min(n_features, n_actual_features)
    print(f"Total features available: {n_actual_features}")
    print(f"Selecting top {n_features} features")
    
    feature_names = vec.get_feature_names_out()
    selector = SelectKBest(chi2, k=n_features)
    X_train_selected = selector.fit_transform(X_train_feats, y_train)
    X_test_selected = selector.transform(X_test_feats)
    X_val_selected = selector.transform(X_val_feats)
    
    selected_indices = selector.get_support(indices=True)
    selected_indices = selected_indices[selected_indices < len(feature_names)]
    selected_features = feature_names[selected_indices]
    print(f"Selected {len(selected_features)} most informative features")

    # Perform hyperparameter tuning
    print("\nPerforming hyperparameter tuning...")
    param_grid = {
        'n_estimators': [1000],
        'learning_rate': [0.1],
        'max_depth': [3],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'subsample': [0.8]
    }
    
    base_clf = GradientBoostingClassifier(
        random_state=random_state,
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-4
    )
    grid_search = GridSearchCV(
        base_clf,
        param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    with tqdm(total=1, desc="Grid Search") as pbar:
        grid_search.fit(X_train_selected, y_train)
        pbar.update(1)
    
    print(f"Best parameters: {grid_search.best_params_}")
    clf = grid_search.best_estimator_
    
    # Train final model with best parameters
    print("\nTraining final model with best parameters and early stopping...")
    with tqdm(total=1, desc="Final Training") as pbar:
        clf.fit(X_train_selected, y_train)
        pbar.update(1)
    
    # Evaluate model performance across all sets
    print("\nEvaluating on training data:")
    y_train_pred = clf.predict(X_train_selected)
    print("\nTraining Set Classification Report:")
    print(classification_report(y_train, y_train_pred))
    print("\nTraining Set Confusion Matrix:")
    print(confusion_matrix(y_train, y_train_pred))
    
    print("\nEvaluating on validation data:")
    y_val_pred = clf.predict(X_val_selected)
    print("\nValidation Set Classification Report:")
    print(classification_report(y_val, y_val_pred))
    print("\nValidation Set Confusion Matrix:")
    print(confusion_matrix(y_val, y_val_pred))
    
    print("\nEvaluating on test data:")
    y_test_pred = clf.predict(X_test_selected)
    print("\nTest Set Classification Report:")
    print(classification_report(y_test, y_test_pred))
    print("\nTest Set Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))

    # Save model artifacts
    print("\nSaving model artifacts...")
    os.makedirs(model_output_dir, exist_ok=True)
    vec_path = os.path.join(model_output_dir, 'tfidf_vectorizer_gb.pkl')
    model_path = os.path.join(model_output_dir, 'phish_detector_gb.pkl')
    selector_path = os.path.join(model_output_dir, 'feature_selector_gb.pkl')
    features_path = os.path.join(model_output_dir, 'selected_features_gb.pkl')

    with open(vec_path, 'wb') as f:
        pickle.dump(vec, f)
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    with open(selector_path, 'wb') as f:
        pickle.dump(selector, f)
    with open(features_path, 'wb') as f:
        pickle.dump(selected_features, f)

    print(f"\nSaved artifacts to {model_output_dir}:")
    print(f"- Vectorizer: {vec_path}")
    print(f"- Model: {model_path}")
    print(f"- Feature selector: {selector_path}")
    print(f"- Selected features: {features_path}")

if __name__ == "__main__":
    import sys
    data_csv = sys.argv[1] if len(sys.argv) > 1 else './data/Kaggle.csv'
    train_and_evaluate_gb(data_csv) 