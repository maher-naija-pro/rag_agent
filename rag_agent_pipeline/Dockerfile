FROM python:3.11-slim

# System deps: tesseract for OCR, curl for healthchecks
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-fra \
        tesseract-ocr-eng \
        curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

ARG REQUIREMENTS=requirements.txt
COPY requirements*.txt .
RUN pip install --no-cache-dir -r ${REQUIREMENTS}

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
