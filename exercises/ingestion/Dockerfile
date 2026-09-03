FROM python:3.12-slim

WORKDIR /app

# Dependencies first, in their own layer: this step is cached and only re-runs
# when requirements.txt changes, not every time you edit the code.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the code explicitly rather than `COPY . .`, so .env never ends up baked
# into the image. Compose supplies the environment at run time instead.
COPY ingestion.py .

CMD ["uvicorn", "ingestion:app", "--host", "0.0.0.0", "--port", "8000"]