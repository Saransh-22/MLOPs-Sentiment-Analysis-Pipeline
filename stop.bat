@echo off
cls
echo.
echo ====================================================
echo Stopping Sentiment Analysis System...
echo ====================================================
echo.
docker-compose down
echo.
echo System stopped successfully!
echo.
pause
