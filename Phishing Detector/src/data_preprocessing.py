"""
Data Preprocessing Module

This module handles data loading, cleaning, and preparation for the phishing email
detection system. It includes functions for text cleaning, missing data handling,
and train-test splitting with stratification.
"""

import pandas as pd
from bs4 import BeautifulSoup
import re
from sklearn.model_selection import train_test_split

#IMPORTS
# bs4 clean_text() function to strip out any HTML tags so you're left with plain text.


def load_data(path):
    """
    Load data from a CSV file.
    """
    return pd.read_csv(path)

def inspect_missing(df, cols):
    """
    Count missing values in specified columns.
    """
    return df[cols].isnull().sum()

def drop_missing(df, cols):
    """
    Remove rows with missing values in specified columns.
    """
    return df.dropna(subset=cols).copy()

def clean_text(text):
    """
    Clean and normalize text by:
    1. Removing HTML tags
    2. Converting to lowercase
    3. Removing URLs
    4. Removing non-alphanumeric characters
    5. Normalizing whitespace
    """
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess(df):
    """
    Preprocess the dataset by:
    1. Removing rows with missing values
    2. Cleaning text content
    """
    df = drop_missing(df, ['body','label'])
    df['clean_body'] = df['body'].apply(clean_text)
    return df

def split_data(df, text_col, label_col, **kwargs):
    """
    Split data into features and labels, then into train-test sets.
    """
    X = df[[text_col]]
    y = df[label_col]
    return train_test_split(X, y, **kwargs)

def load_and_clean(path,
                   text_col: str = 'clean_body',
                   label_col: str = 'label',
                   test_size: float = 0.2,
                   random_state: int = 42):
    """
    Complete data loading and preprocessing pipeline:
    1. Load data from CSV
    2. Preprocess (handle missing values and clean text)
    3. Split into train-test sets with stratification
    
    Returns:
        X_train, X_test: Feature DataFrames
        y_train, y_test: Label Series
    """
    df = load_data(path)
    df_clean = preprocess(df)  

    X = df_clean[[text_col]]
    y = df_clean[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test

def main():
    """
    Example usage of the preprocessing pipeline.
    """
    df = load_data("./data/Kaggle.csv")
    print("Missing before drop:", inspect_missing(df, ['sender','receiver','subject','body','label']))
    df_clean = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df_clean[['clean_body']],
        df_clean['label'],
        test_size=0.2,
        random_state=42,
        stratify=df_clean['label']
    )
    
    print(f"\nOriginal rows: {df.shape[0]}")
    print(f"Rows after drop: {df_clean.shape[0]}\n")

    print("Sample cleaned training emails:")
    print(pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1).head(10))

if __name__ == "__main__":
    main()

