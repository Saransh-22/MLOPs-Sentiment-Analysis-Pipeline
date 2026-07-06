# 🚀 Sentiment Analysis MLOps Pipeline

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.47-red?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)
![AWS](https://img.shields.io/badge/AWS-EC2-orange?logo=amazonaws)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue)

</p>

An **end-to-end MLOps pipeline** for sentiment analysis built using **Scikit-Learn, MLflow, FastAPI, Streamlit, Docker, and AWS EC2**.

The project covers the complete machine learning lifecycle—from preprocessing and model training to experiment tracking, model serving, containerization, and cloud deployment.

---

# 📌 Features

- NLP text preprocessing
- TF-IDF vectorization
- Train multiple ML models
- MLflow experiment tracking
- Automatic best model selection
- REST API using FastAPI
- Interactive Streamlit interface
- Dockerized deployment
- Docker Compose orchestration
- AWS EC2 deployment

---

# 🏗 Architecture

```text
                    User
                      │
                      ▼
              Streamlit Web UI
                      │
             HTTP Request (/predict)
                      │
                      ▼
              FastAPI Backend
                      │
          Text Preprocessing
                      │
          TF-IDF Vectorization
                      │
             Trained ML Model
                      │
                      ▼
             Sentiment Prediction
```

---

# 🛠 Tech Stack

| Category | Technologies |
|------------|-------------------------------|
| Language | Python |
| Machine Learning | Scikit-Learn |
| NLP | NLTK, TF-IDF |
| Experiment Tracking | MLflow |
| Backend | FastAPI |
| Frontend | Streamlit |
| Containerization | Docker, Docker Compose |
| Cloud | AWS EC2 |
| Version Control | Git, GitHub |

---

# 📂 Project Structure

```text
sentiment-mlops/
│
├── app/
│   ├── main.py
│   └── ui.py
│
├── src/
│   ├── preprocess.py
│   ├── train_mlflow.py
│   ├── predict.py
│   └── config.py
│
├── models/
│
├── data/
│   ├── data.csv
│   └── processed/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-train.txt
├── run_pipeline.py
└── README.md
```

---

# ⚙ Machine Learning Pipeline

```text
Dataset
   │
   ▼
Preprocessing
   │
   ▼
TF-IDF Vectorization
   │
   ▼
Train Multiple Models
   │
   ▼
MLflow Tracking
   │
   ▼
Select Best Model
   │
   ▼
Save Model (.pkl)
   │
   ▼
FastAPI
   │
   ▼
Streamlit
   │
   ▼
AWS EC2 Deployment
```

---

# 🚀 Local Setup

## Clone Repository

```bash
git clone https://github.com/Saransh-22/MLOPs-Sentiment-Analysis-Pipeline.git

cd MLOPs-Sentiment-Analysis-Pipeline
```

## Create Virtual Environment

```bash
python -m venv venv
```

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🧠 Train the Model

Run preprocessing

```bash
python src/preprocess.py
```

Train models

```bash
python src/train_mlflow.py
```

Run complete pipeline

```bash
python run_pipeline.py
```

---

# 🌐 Run the Application

## FastAPI

```bash
uvicorn app.main:app --reload
```

Swagger Documentation

```
http://localhost:8000/docs
```

---

## Streamlit

```bash
streamlit run app/ui.py
```

```
http://localhost:8501
```

---

# 🐳 Docker Deployment

Build and run

```bash
docker compose up --build
```

Services

| Service | URL |
|---------|-----------------------------|
| FastAPI | http://localhost:8000 |
| Swagger | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |

Stop containers

```bash
docker compose down
```

---

# ☁ AWS EC2 Deployment

This project is deployed on an **AWS EC2 Ubuntu instance** using Docker Compose.

Deployment workflow:

1. Launch EC2 Instance
2. Install Docker
3. Install Docker Compose
4. Clone Repository
5. Build Docker Images
6. Run Containers
7. Configure Security Groups
8. Access the application via Public IP

---

# 📡 API Documentation

## POST /predict

Request

```json
{
  "text": "I love this product!"
}
```

Response

```json
{
  "prediction": "Positive",
  "confidence": 0.98
}
```

---

# 📷 Project Screenshots

> Add screenshots here after deployment.

### Streamlit UI

```
screenshots/home.png
```

### Prediction

```
screenshots/prediction.png
```

### Swagger Documentation

```
screenshots/swagger.png
```

### AWS Deployment

```
screenshots/aws.png
```

---

# 📈 Future Improvements

- GitHub Actions CI/CD
- Nginx Reverse Proxy
- HTTPS using Let's Encrypt
- Docker Hub Integration
- Portainer Dashboard
- MLflow Model Registry
- Monitoring & Logging
- Model Drift Detection

---

# 👨‍💻 Author

**Saransh Neema**

GitHub: https://github.com/Saransh-22

LinkedIn: *(Add your LinkedIn profile here)*

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.