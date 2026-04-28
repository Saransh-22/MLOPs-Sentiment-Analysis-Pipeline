
import os
import re
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from config import (
    DATA_PATH, TEXT_COLUMN, LABEL_COLUMN,
    TEST_SIZE, RANDOM_STATE, MAX_FEATURES,
    PROCESSED_DATA_DIR, X_TRAIN_PATH, X_TEST_PATH,
    Y_TRAIN_PATH, Y_TEST_PATH, VECTORIZER_PATH
)

STOP_WORDS = set(stopwords.words('english'))


def clean_tweet(tweet):
    
    tweet = tweet.lower()
    
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet)
    
    tweet = re.sub(r'#(\w+)', r'\1', tweet)
    
    tweet = re.sub(r'@\w+', '', tweet)
    
    tweet = re.sub(r'[^\w\s]', '', tweet)
    
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    
    words = tweet.split()
    words = [word for word in words if word not in STOP_WORDS and len(word) > 1]
    tweet = ' '.join(words)
    
    return tweet


def load_dataset(filepath):
    
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
        print("Dropped 'id' column")
    
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    
    return df


def preprocess_data():
    
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    df = load_dataset(DATA_PATH)
    
    if TEXT_COLUMN not in df.columns or LABEL_COLUMN not in df.columns:
        raise ValueError(f"Dataset must have '{TEXT_COLUMN}' and '{LABEL_COLUMN}' columns")
    
    print(f"\nChecking for missing values...")
    print(f"Missing in {TEXT_COLUMN}: {df[TEXT_COLUMN].isna().sum()}")
    print(f"Missing in {LABEL_COLUMN}: {df[LABEL_COLUMN].isna().sum()}")
    
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
    print(f"Dataset shape after removing NaNs: {df.shape[0]} rows")
    
    print(f"\nCleaning tweets...")
    df['tweet_cleaned'] = df[TEXT_COLUMN].apply(clean_tweet)
    
    df = df[df['tweet_cleaned'].str.len() > 0]
    print(f"Dataset shape after removing empty tweets: {df.shape[0]} rows")
    
    X_text = df['tweet_cleaned'].values
    y = df[LABEL_COLUMN].values
    
    print(f"\nClass distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Label {label}: {count} samples ({count/len(y)*100:.1f}%)")
    
    print(f"\nVectorizing with TF-IDF (max_features={MAX_FEATURES})...")
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, lowercase=False)
    X = vectorizer.fit_transform(X_text)
    
    print(f"Vectorized shape: {X.shape}")
    print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")
    
    print(f"\nSplitting data (test_size={TEST_SIZE}, random_state={RANDOM_STATE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    print(f"\nSaving processed data...")
    from scipy.sparse import save_npz
    
    save_npz(X_TRAIN_PATH, X_train)
    save_npz(X_TEST_PATH, X_test)
    np.save(Y_TRAIN_PATH, y_train)
    np.save(Y_TEST_PATH, y_test)
    
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("\n[*] Preprocessing complete!")
    print(f"[*] Saved: {X_TRAIN_PATH}")
    print(f"[*] Saved: {X_TEST_PATH}")
    print(f"[*] Saved: {Y_TRAIN_PATH}")
    print(f"[*] Saved: {Y_TEST_PATH}")
    print(f"[*] Saved: {VECTORIZER_PATH}")


if __name__ == "__main__":
    preprocess_data()
