
import os
import pickle
import numpy as np
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mlflow.set_tracking_uri(f"file:{BASE_DIR}/mlruns")

DATA_DIR = "data/processed"
MODELS_DIR = "models"
MLFLOW_EXPERIMENT = "sentiment-analysis"

os.makedirs(MODELS_DIR, exist_ok=True)


mlflow.set_experiment(MLFLOW_EXPERIMENT)


def load_data():
    print("Loading preprocessed data...")
    
    X_train = load_npz(f"{DATA_DIR}/X_train.npz")
    X_test = load_npz(f"{DATA_DIR}/X_test.npz")
    y_train = np.load(f"{DATA_DIR}/y_train.npy")
    y_test = np.load(f"{DATA_DIR}/y_test.npy")
    
    print(f"[*] Data loaded")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  y_test shape: {y_test.shape}\n")
    
    return X_train, X_test, y_train, y_test


def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test):
    print(f"Training {model_name}...")
    
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        mlflow.log_param("model_name", model_name)
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        
        mlflow.sklearn.log_model(model, "model")
        
        print(f"[*] {model_name} trained")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  [*] Logged to MLflow\n")
        
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
    results.append(train_and_evaluate(lr_model, "Logistic Regression", X_train, X_test, y_train, y_test))
    
    nb_model = MultinomialNB()
    results.append(train_and_evaluate(nb_model, "Naive Bayes", X_train, X_test, y_train, y_test))
    
    svm_model = LinearSVC(max_iter=2000, random_state=42)
    results.append(train_and_evaluate(svm_model, "SVM", X_train, X_test, y_train, y_test))
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    results.append(train_and_evaluate(rf_model, "Random Forest", X_train, X_test, y_train, y_test))
    
    xgb_model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    results.append(train_and_evaluate(xgb_model, "XGBoost", X_train, X_test, y_train, y_test))
    
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


def save_best_model(results):
    best_result = max(results, key=lambda x: x['f1'])
    
    best_model_path = f"{MODELS_DIR}/best_model.pkl"
    
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_result['model'], f)
    
    print(f"\n[*] Best model saved: {best_model_path}")
    print(f"  Model: {best_result['name']}")
    print(f"  Accuracy: {best_result['accuracy']:.4f}")
    print(f"  F1 Score: {best_result['f1']:.4f}")
    
    return best_result


def main():
    print("=" * 60)
    print("SENTIMENT ANALYSIS - MODEL TRAINING WITH MLflow")
    print("=" * 60 + "\n")
    
    X_train, X_test, y_train, y_test = load_data()
    
    results = train_all_models(X_train, X_test, y_train, y_test)
    
    print_comparison(results)
    
    save_best_model(results)
    
    print("\n" + "=" * 60)
    print("[*] Training complete!")
    print("=" * 60)
    print("\nTo view MLflow UI:")
    print("  Run: mlflow ui")
    print("  Open: http://localhost:5000")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()