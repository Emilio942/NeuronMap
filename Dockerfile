# Multi-stage Docker build for NeuronMap
FROM python:3.13.7-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install pytest pytest-cov black flake8 mypy jupyter notebook

# Copy source code
COPY . .

# Set up the package in development mode
RUN pip install -e .

# Expose port for Jupyter notebook
EXPOSE 8888

# Default command for development
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Production stage
FROM base as production

# Copy only necessary files
COPY src/ ./src/
COPY configs/ ./configs/
COPY main.py .
COPY setup.py .
COPY README.md .

# Install the package
RUN pip install .

# Create non-root user
RUN useradd --create-home --shell /bin/bash neuronmap
USER neuronmap

# Create data directories
RUN mkdir -p /home/neuronmap/data/{raw,processed,outputs}
RUN mkdir -p /home/neuronmap/logs

# Set working directory to user home
WORKDIR /home/neuronmap

# Copy configs to user directory
COPY --chown=neuronmap:neuronmap configs/ ./configs/

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import src.utils.monitoring; print('Health check passed')" || exit 1

# Default command
CMD ["python", "-m", "src.main", "--help"]

# GPU-enabled stage
FROM nvidia/cuda:11.8-runtime-ubuntu20.04 as gpu

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV CUDA_VISIBLE_DEVICES=0

# Create app directory
WORKDIR /app

# Copy requirements with GPU support
COPY requirements.txt .

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install the package
RUN pip install .

# Create non-root user
RUN useradd --create-home --shell /bin/bash neuronmap
USER neuronmap

# Create data directories
RUN mkdir -p /home/neuronmap/data/{raw,processed,outputs}
RUN mkdir -p /home/neuronmap/logs

# Set working directory
WORKDIR /home/neuronmap

# Copy configs
COPY --chown=neuronmap:neuronmap configs/ ./configs/

# Default command for GPU version
CMD ["python", "-c", "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"]
