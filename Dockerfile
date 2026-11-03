# syntax=docker/dockerfile:1
FROM python:3.11-slim

LABEL maintainer="semdiff"
LABEL description="Cross-Lingual Semantic Diff for Regulatory Documents"

# System dependencies for PDF parsing and Indic fonts
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpoppler-cpp-dev \
    poppler-utils \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY data/ ./data/
COPY evaluation/ ./evaluation/
COPY pyproject.toml .

# Pre-download LaBSE model weights (cached at build time for offline use)
# Uncomment to bake in model weights (increases image size by ~1.9GB):
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/LaBSE')"

# Create directories
RUN mkdir -p .cache reports models/classifier

# Default: run FastAPI server
EXPOSE 8000
ENV PYTHONPATH=/app/src

CMD ["uvicorn", "semdiff.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# Alternative: run Streamlit UI
# CMD ["streamlit", "run", "src/semdiff/ui.py", "--server.port=8501", "--server.address=0.0.0.0"]

