# Sentiment Analysis MLOps Project

A complete end-to-end sentiment analysis machine learning pipeline with MLflow tracking, model training, and FastAPI deployment.

---

## Project Structure

```
sentiment-mlops/
├── data/
│   ├── data.csv                 # Raw dataset
│   └── processed/               # Preprocessed data
│       ├── X_train.npz
│       ├── X_test.npz
│       ├── y_train.npy
│       ├── y_test.npy
│       └── vectorizer.pkl       # TF-IDF vectorizer
├── models/
│   ├── best_model.pkl           # Best trained model
│   └── [other models]
├── src/
│   ├── preprocess.py            # Data preprocessing
│   ├── train_mlflow.py          # Model training with MLflow
│   ├── predict.py               # Prediction logic
│   └── config.py                # Configuration
├── app/
│   └── main.py                  # FastAPI application
│   └── ui.py                     # Streamlit UI
├── run_pipeline.py              # Pipeline orchestrator
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker containerization
├── docker-compose.yml            # FastAPI + Streamlit orchestration
└── README.md                    # This file
```

---

## Quick Start

### 1. Setup Environment

**Create virtual environment:**
```bash
python -m venv venv
.\venv\Scripts\activate          # Windows
source venv/bin/activate         # Linux/Mac
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## How to Run Each Component

### 2. Data Preprocessing

Clean and vectorize the raw data:

```bash
python src/preprocess.py
```

**Output:**
- `data/processed/X_train.npz` - Training features
- `data/processed/X_test.npz` - Test features
- `data/processed/y_train.npy` - Training labels
- `data/processed/y_test.npy` - Test labels
- `data/processed/vectorizer.pkl` - TF-IDF vectorizer

---

### 3. Model Training with MLflow

Train all 5 models and track with MLflow:

```bash
python src/train_mlflow.py
```

**Output:**
- `models/best_model.pkl` - Best performing model
- MLflow runs in `mlruns/` directory
- Metrics: Accuracy, F1 Score

**View MLflow Dashboard:**
```bash
mlflow ui
```
Then open http://localhost:5000 in your browser

---

### 4. Run Complete Pipeline

Execute preprocessing + training in sequence:

```bash
python run_pipeline.py
```

This will:
1. Preprocess data
2. Train models
3. Save best model
4. Log to MLflow

---

### 5. Start FastAPI Server

Run the prediction API:

```bash
uvicorn app.main:app --reload
```

**API will be available at:**
- Main: http://localhost:8000
- Interactive Docs (Swagger): http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

**Test the endpoint:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"I love this product!"}'
```

**Response:**
```json
{
  "prediction": "Positive",
  "confidence": 0.8651
}
```

---

### 6. Start Streamlit UI

Run the Streamlit web UI (requires the FastAPI backend running):

```bash
streamlit run app/ui.py
```

**UI will be available at:**
- http://localhost:8501

**Optional: point UI to a different API URL**
```bash
# Windows (PowerShell)
$env:SENTIMENT_API_URL = "http://localhost:8000"
streamlit run app/ui.py

# Linux/Mac
SENTIMENT_API_URL=http://localhost:8000 streamlit run app/ui.py
```

---

### 7. Docker Deployment (FastAPI only)

**Build Docker image:**
```bash
docker build -t sentiment-api .
```

**Run container:**
```bash
docker run -p 8000:8000 sentiment-api
```

**Test:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"I love this product!"}'
```

---

### 8. Docker Compose (FastAPI + Streamlit)

Build and start both services:

```bash
docker-compose up --build
```

**Access:**
- FastAPI: http://localhost:8000
- Swagger Docs: http://localhost:8000/docs
- Streamlit UI: http://localhost:8501

Stop services:
```bash
docker-compose down
```

---

## Component Details

### Data Preprocessing (src/preprocess.py)

- Loads raw CSV data
- Text cleaning:
  - Lowercase
  - Remove URLs
  - Remove hashtags (keep words)
  - Remove mentions (@username)
  - Remove punctuation
  - Remove stopwords
- TF-IDF vectorization (5000 features)
- Train-test split (80-20)

### Model Training (src/train_mlflow.py)

- Trains 5 models:
  1. Logistic Regression
  2. Naive Bayes
  3. SVM
  4. Random Forest
  5. XGBoost
- Logs metrics (Accuracy, F1 Score) to MLflow
- Saves best model based on F1 score
- All runs tracked and versioned

### Prediction API (src/predict.py + app/main.py)

- Loads best model and vectorizer
- Same preprocessing as training
- FastAPI REST endpoint
- Returns prediction + confidence score

### Streamlit UI (app/ui.py)

- Simple web interface for sentiment prediction
- Calls FastAPI `/predict` endpoint
- Shows prediction with confidence
- Uses `SENTIMENT_API_URL` environment variable (defaults to http://localhost:8000)

---

## API Endpoints

### GET `/`

Health check endpoint

**Response:**
```json
{
  "message": "Sentiment Analysis API is running",
  "version": "1.0.0"
}
```

### POST `/predict`

Predict sentiment for text

**Request:**
```json
{
  "text": "I love this product!"
}
```

**Response:**
```json
{
  "prediction": "Positive",
  "confidence": 0.8651
}
```

---

## Testing

### Using Swagger UI (Easiest)

1. Start API: `uvicorn app.main:app --reload`
2. Open: http://localhost:8000/docs
3. Click `/predict` → "Try it out"
4. Enter text
5. Click "Execute"

### Using Python

```python
import requests

url = "http://localhost:8000/predict"
data = {"text": "I love this phone!"}
response = requests.post(url, json=data)
print(response.json())
```

### Using PowerShell

```powershell
$uri = "http://localhost:8000/predict"
$body = @{"text"="I love this product!"} | ConvertTo-Json
Invoke-RestMethod -Uri $uri -Method Post -Body $body -ContentType "application/json"
```

---

## Configuration

Edit `src/config.py` to customize:

```python
DATA_PATH = "data/data.csv"           # Raw data
TEXT_COLUMN = "tweet"                 # Text column name
LABEL_COLUMN = "label"                # Label column name
TEST_SIZE = 0.2                       # Test split ratio
MAX_FEATURES = 5000                   # TF-IDF features
```

---

## Troubleshooting

### Issue: "No such file or directory"
- Make sure you're in the project directory: `cd sentiment-mlops`
- Verify data files exist in `data/` folder

### Issue: "Module not found"
- Reinstall dependencies: `pip install -r requirements.txt`
- Verify virtual environment is activated

### Issue: MLflow UI showing no data
- Stop and restart MLflow: `mlflow ui`
- Hard refresh browser: `Ctrl + Shift + R`

### Issue: API returning opposite predictions
- This was fixed! Update `src/predict.py` (0 = Positive, 1 = Negative)

---

## Next Steps

1. **Improve model accuracy** - Try different preprocessing or hyperparameters
2. **Add model versioning** - Use MLflow model registry
3. **Add unit tests** - Test preprocessing and prediction logic
4. **Deploy to cloud** - Use AWS, GCP, or Azure
5. **Add monitoring** - Track API performance and model drift

---

## License

This is an educational MLOps project.

---

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review logs in `mlruns/` folder
3. Test individual components (preprocess.py, train_mlflow.py, predict.py)
