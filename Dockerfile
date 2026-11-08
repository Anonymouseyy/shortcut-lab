FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU PyTorch & dependencies first for fast layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code and sample assets
COPY assets/ ./assets/
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY README.md ./

# Pre-cache default ViT-B/16 and DeepLabV3 weights during build
ENV TORCH_HOME=/app/weights
RUN python -c "import torchvision.models as m; m.vit_b_16(weights=m.ViT_B_16_Weights.DEFAULT); m.resnet50(weights=m.ResNet50_Weights.DEFAULT); m.segmentation.deeplabv3_mobilenet_v3_large(weights=m.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)"

EXPOSE 7860

CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
