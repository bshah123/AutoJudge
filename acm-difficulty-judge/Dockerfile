# Use Python 3.12 (matches your local/Colab environment)
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies for LightGBM
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Hugging Face Spaces runs on port 7860 by default
ENV FLASK_RUN_PORT=7860
EXPOSE 7860

# Command to run the app
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app_production:app"]