# src/model_training.py
from feature_engineering import build_tfidf_vectorizer, vectorize_train_test
from data_preprocessing import load_and_clean
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

"""
Module: model_training.py
Loads data, vectorizes text, trains and evaluates a model, then saves artifacts.
"""

def train_and_evaluate(
    data_path,
    model_output_dir='artifacts',
    test_size=0.2,
    random_state=42
):
    # 1. Load and split
    X_train, X_test, y_train, y_test = load_and_clean(
        data_path,
        test_size=test_size,
        random_state=random_state,
    )
    # 2. Vectorize
    vec = build_tfidf_vectorizer()
    X_train_tfidf, X_test_tfidf = vectorize_train_test(
        vec,
        X_train['clean_body'],
        X_test['clean_body']
    )
    # 3. Train
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_train_tfidf, y_train)
    # 4. Evaluate
    preds = clf.predict(X_test_tfidf)
    print(classification_report(y_test, preds))
    # 5. Save artifacts
    import os
    os.makedirs(model_output_dir, exist_ok=True)
    with open(f"{model_output_dir}/tfidf_vectorizer.pkl", 'wb') as f:
        pickle.dump(vec, f)
    with open(f"{model_output_dir}/phish_detector.pkl", 'wb') as f:
        pickle.dump(clf, f)

if __name__ == "__main__":
    import sys
    data_csv = sys.argv[1] if len(sys.argv) > 1 else './data/Kaggle.csv'
    train_and_evaluate(data_csv)
