@echo off
cls
echo.
echo ====================================================
echo Sentiment Analysis MLOps System
echo ====================================================
echo.
echo Starting FastAPI and Streamlit services...
echo.
echo Services:
echo   Streamlit UI: http://localhost:8501
echo   FastAPI API:  http://localhost:8000
echo   FastAPI Docs: http://localhost:8000/docs
echo.
echo ====================================================
echo.
docker-compose up
pause
