
import re
import pickle
import nltk
from nltk.corpus import stopwords

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

STOP_WORDS = set(stopwords.words('english'))
MODEL_PATH = "models/best_model.pkl"
VECTORIZER_PATH = "data/processed/vectorizer.pkl"


def load_model_and_vectorizer():
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    
    return model, vectorizer


def clean_text(text):
    
    text = text.lower()
    
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    text = re.sub(r'#(\w+)', r'\1', text)
    
    text = re.sub(r'@\w+', '', text)
    
    text = re.sub(r'[^\w\s]', '', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    words = text.split()
    words = [word for word in words if word not in STOP_WORDS and len(word) > 1]
    text = ' '.join(words)
    
    return text


def predict(input_text):
    
    model, vectorizer = load_model_and_vectorizer()
    
    cleaned_text = clean_text(input_text)
    
    if not cleaned_text:
        return {
            "prediction": "Invalid input - text too short after cleaning",
            "confidence": 0.0
        }
    
    X = vectorizer.transform([cleaned_text])
    
    prediction = model.predict(X)[0]
    
    label = "Positive" if prediction == 0 else "Negative"
    
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)[0]
        confidence = float(max(probabilities))
    else:
        confidence = 1.0
    
    return {
        "prediction": label,
        "confidence": round(confidence, 4)
    }


if __name__ == "__main__":
    test_texts = [
        "I love this phone! Amazing product",
        "This is terrible, worst purchase ever",
        "It's okay, nothing special"
    ]
    
    print("Testing prediction logic...\n")
    for text in test_texts:
        result = predict(text)
        print(f"Text: {text}")
        print(f"Result: {result}\n")
