
import os
import pickle
import numpy as np
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

DATA_DIR = "data/processed"
MODELS_DIR = "models"

os.makedirs(MODELS_DIR, exist_ok=True)


def load_data():
    
    print("Loading preprocessed data...")
    
    X_train = load_npz(f"{DATA_DIR}/X_train.npz")
    X_test = load_npz(f"{DATA_DIR}/X_test.npz")
    y_train = np.load(f"{DATA_DIR}/y_train.npy")
    y_test = np.load(f"{DATA_DIR}/y_test.npy")
    
    print(f"✓ Data loaded")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  y_test shape: {y_test.shape}\n")
    
    return X_train, X_test, y_train, y_test


def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test):
    
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"✓ {model_name} trained")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}\n")
    
    return {
        "name": model_name,
        "model": model,
        "accuracy": accuracy,
        "f1": f1,
        "y_pred": y_pred
    }


def train_all_models(X_train, X_test, y_train, y_test):
    
    results = []
    
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_result = train_and_evaluate(lr_model, "Logistic Regression", X_train, X_test, y_train, y_test)
    results.append(lr_result)
    
    nb_model = MultinomialNB()
    nb_result = train_and_evaluate(nb_model, "Naive Bayes", X_train, X_test, y_train, y_test)
    results.append(nb_result)
    
    svm_model = LinearSVC(max_iter=2000, random_state=42)
    svm_result = train_and_evaluate(svm_model, "SVM", X_train, X_test, y_train, y_test)
    results.append(svm_result)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_result = train_and_evaluate(rf_model, "Random Forest", X_train, X_test, y_train, y_test)
    results.append(rf_result)
    
    xgb_model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_result = train_and_evaluate(xgb_model, "XGBoost", X_train, X_test, y_train, y_test)
    results.append(xgb_result)
    
    return results


def print_comparison(results):
    
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Model':<20} {'Accuracy':<15} {'F1 Score':<15}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['name']:<20} {result['accuracy']:<15.4f} {result['f1']:<15.4f}")
    
    print("=" * 60)


def save_models(results):
    
    print("\nSaving models...")
    
    for result in results:
        model_name = result['name'].lower().replace(" ", "_")
        filepath = f"{MODELS_DIR}/{model_name}.pkl"
        
        with open(filepath, 'wb') as f:
            pickle.dump(result['model'], f)
        
        print(f"✓ Saved: {filepath}")


def save_best_model(results):
    
    best_result = max(results, key=lambda x: x['f1'])
    
    best_model_path = f"{MODELS_DIR}/best_model.pkl"
    
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_result['model'], f)
    
    print(f"\n✓ Best model saved: {best_model_path}")
    print(f"  Model: {best_result['name']}")
    print(f"  F1 Score: {best_result['f1']:.4f}")


def main():
    
    print("=" * 60)
    print("SENTIMENT ANALYSIS - MODEL TRAINING")
    print("=" * 60 + "\n")
    
    X_train, X_test, y_train, y_test = load_data()
    
    results = train_all_models(X_train, X_test, y_train, y_test)
    
    print_comparison(results)
    
    save_models(results)
    
    save_best_model(results)
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
