FROM python:3.11-slim

WORKDIR /app

# install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy application code
COPY app/ ./app/
COPY data/ ./data/
COPY shl_product_catalog.json .
COPY scripts/ ./scripts/

# build the embedding index at image build time
# so the container starts ready-to-serve
RUN python scripts/build_index.py

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
