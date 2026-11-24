FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch wheels first so cloud CPU builders avoid downloading ~2.5 GB CUDA bloat
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face Spaces runs Docker containers as non-root UID 1000
RUN useradd -m -u 1000 user \
    && mkdir -p /home/user/.cache /home/user/.config /home/user/.EasyOCR \
    && chown -R user:user /app /home/user

ENV HOME=/home/user \
    PATH="/home/user/.local/bin:$PATH" \
    PORT=7860 \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/home/user/.config/Ultralytics \
    DEEPFACE_HOME=/home/user/.cache/deepface \
    HF_HOME=/home/user/.cache/huggingface \
    TORCH_HOME=/home/user/.cache/torch \
    MPLCONFIGDIR=/home/user/.cache/matplotlib

COPY --chown=user:user . .

USER user

# Pre-bake YOLO11m weights (~40 MB) for instant startup
RUN python -c "from ultralytics import YOLO; YOLO('yolo11m.pt')"

EXPOSE 7860

CMD ["python", "main.py"]
