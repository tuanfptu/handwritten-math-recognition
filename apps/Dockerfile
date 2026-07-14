# Use PyTorch official image with CUDA 12.1 support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for Qwen3-VL
RUN pip install --no-cache-dir \
    peft \
    accelerate \
    bitsandbytes \
    sentencepiece \
    protobuf \
    einops \
    timm

# Copy application code
COPY main.py .

# Create necessary directories
RUN mkdir -p outputs_datagen_continue plots_datagen_continue

# Expose port for the server
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Run the application
CMD ["python", "main.py"]
