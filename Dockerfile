FROM python:3.11-slim

WORKDIR /app

# System dependencies for sentence-transformers / pypdf
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PORT=8000
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]