FROM python:3.12-slim

WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Hugging Face Spaces default port is 7860
EXPOSE 7860

# Run uvicorn pointing to your app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]