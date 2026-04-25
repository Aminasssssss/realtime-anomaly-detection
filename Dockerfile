FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY dashboard/ dashboard/

RUN mkdir -p models reports data/SMAP_MSL

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
