
from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict

app = FastAPI(
    title="Sentiment Analysis API",
    description="API for predicting sentiment of text",
    version="1.0.0"
)


class PredictionRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float


@app.get("/")
def read_root():
    
    return {
        "message": "Sentiment Analysis API is running",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Get sentiment prediction for text"
        }
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: PredictionRequest):
    
    result = predict(request.text)
    
    return PredictionResponse(
        prediction=result["prediction"],
        confidence=result["confidence"]
    )


if __name__ == "__main__":
    import uvicorn
    
    print("Starting Sentiment Analysis API...")
    print("Run with: uvicorn app.main:app --reload")
    print("API Docs: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
