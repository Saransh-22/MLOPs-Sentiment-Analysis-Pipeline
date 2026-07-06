# Use Python 3.11 slim image
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
# Set working directory
WORKDIR /app

# Copy requirements first (for better Docker caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Download NLTK stopwords
RUN python -c "import nltk; nltk.download('stopwords')"

# Expose port for FastAPI
EXPOSE 8000

# Command to run FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
