# Installation Guide

This guide provides comprehensive installation instructions for NeuronMap across different platforms and environments.

## 🚀 Quick Install (Recommended)

The fastest way to get started with NeuronMap:

```bash
# Clone the repository
git clone https://github.com/Emilio942/NeuronMap.git
cd NeuronMap

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from src.utils.config import get_config_manager; print('✅ Installation successful!')"
```

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.8+ (3.9+ recommended)
- **RAM**: 8GB (16GB+ recommended for large models)
- **Storage**: 2GB free space
- **OS**: Linux, macOS, or Windows

### Recommended Requirements
- **Python**: 3.10+
- **RAM**: 32GB+ 
- **GPU**: NVIDIA GPU with 8GB+ VRAM (for CUDA acceleration)
- **Storage**: 10GB+ free space (for models and data)
- **OS**: Ubuntu 20.04+ or similar

## 🐧 Linux Installation

### Ubuntu/Debian

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3 python3-pip python3-venv git build-essential -y

# Install additional dependencies for scientific computing
sudo apt install python3-dev libffi-dev libssl-dev -y

# Clone and setup NeuronMap
git clone https://github.com/Emilio942/NeuronMap.git
cd NeuronMap

# Create virtual environment
python3 -m venv neuronmap_env
source neuronmap_env/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

# Install NeuronMap in development mode
pip install -e .

# Verify installation
python -m src.utils.config --validate
```

### CentOS/RHEL/Fedora

```bash
# For CentOS/RHEL 8+
sudo dnf install python3 python3-pip python3-devel git gcc openssl-devel libffi-devel -y

# For older versions
# sudo yum install python3 python3-pip python3-devel git gcc openssl-devel libffi-devel -y

# Follow the same steps as Ubuntu from cloning onwards
git clone https://github.com/Emilio942/NeuronMap.git
cd NeuronMap
python3 -m venv neuronmap_env
source neuronmap_env/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install -e .
```

### Arch Linux

```bash
# Install dependencies
sudo pacman -S python python-pip git base-devel

# Follow standard installation
git clone https://github.com/Emilio942/NeuronMap.git
cd NeuronMap
python -m venv neuronmap_env
source neuronmap_env/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install -e .
```

## 🍎 macOS Installation

### Using Homebrew (Recommended)

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and Git
brew install python git

# Clone and setup NeuronMap
git clone https://github.com/Emilio942/NeuronMap.git
cd NeuronMap

# Create virtual environment
python3 -m venv neuronmap_env
source neuronmap_env/bin/activate

# Install dependencies
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install -e .

# Verify installation
python -c "from src.utils.config import get_config_manager; print('✅ Installation successful!')"
```

### Using MacPorts

```bash
# Install Python via MacPorts
sudo port install python310 py310-pip git

# Create symlinks
sudo port select --set python3 python310
sudo port select --set pip pip310

# Follow standard installation
git clone https://github.com/Emilio942/NeuronMap.git
cd NeuronMap
python3 -m venv neuronmap_env
source neuronmap_env/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install -e .
```

### macOS Apple Silicon (M1/M2)

```bash
# Install Homebrew (ARM64 version)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and dependencies
brew install python git

# Install additional dependencies for scientific computing
brew install openblas lapack

# Set environment variables for Apple Silicon
export OPENBLAS=$(brew --prefix openblas)
export LDFLAGS="-L$OPENBLAS/lib"
export CPPFLAGS="-I$OPENBLAS/include"

# Follow standard installation
git clone https://github.com/Emilio942/NeuronMap.git
cd NeuronMap
python3 -m venv neuronmap_env
source neuronmap_env/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install -e .
```

## 🪟 Windows Installation

### Using Git Bash (Recommended)

```bash
# Install Git for Windows (includes Git Bash)
# Download from: https://git-scm.com/download/win

# Install Python from python.org or Microsoft Store
# Download from: https://www.python.org/downloads/windows/

# Open Git Bash and clone repository
git clone https://github.com/Emilio942/NeuronMap.git
cd NeuronMap

# Create virtual environment
python -m venv neuronmap_env
source neuronmap_env/Scripts/activate

# Install dependencies
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install -e .

# Verify installation
python -c "from src.utils.config import get_config_manager; print('✅ Installation successful!')"
```

### Using PowerShell

```powershell
# Install Python from python.org (ensure "Add to PATH" is checked)
# Install Git for Windows

# Clone repository
git clone https://github.com/Emilio942/NeuronMap.git
cd NeuronMap

# Create virtual environment
python -m venv neuronmap_env
neuronmap_env\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install -e .

# Verify installation
python -c "from src.utils.config import get_config_manager; print('✅ Installation successful!')"
```

### Using WSL (Windows Subsystem for Linux)

```bash
# Install WSL2 with Ubuntu
wsl --install -d Ubuntu

# Open Ubuntu terminal and follow Linux installation instructions
sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-pip python3-venv git -y

git clone https://github.com/Emilio942/NeuronMap.git
cd NeuronMap
python3 -m venv neuronmap_env
source neuronmap_env/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install -e .
```

## 🐳 Docker Installation

### Using Docker Compose (Recommended)

```bash
# Clone repository
git clone https://github.com/Emilio942/NeuronMap.git
cd NeuronMap

# Build and run with Docker Compose
docker-compose up --build

# Access the container
docker-compose exec neuronmap bash

# Verify installation inside container
python -c "from src.utils.config import get_config_manager; print('✅ Installation successful!')"
```

### Using Dockerfile

```bash
# Build Docker image
docker build -t neuronmap:latest .

# Run container
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  neuronmap:latest

# For GPU support (requires nvidia-docker)
docker run -it --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  neuronmap:latest
```

## ⚡ GPU Support

### NVIDIA CUDA Setup

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Install CUDA-enabled PyTorch (if not already installed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA installation
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"

# Configure NeuronMap for GPU
export NEURONMAP_ENV=production
python -m src.utils.config --environment production --hardware-check
```

### AMD ROCm Support

```bash
# Install ROCm (Linux only)
sudo apt install rocm-dev

# Install ROCm-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2

# Verify ROCm installation
python -c "import torch; print(f'ROCm available: {torch.cuda.is_available()}')"
```

## 🔧 Development Installation

For contributing to NeuronMap development:

```bash
# Clone with development branch
git clone -b develop https://github.com/Emilio942/NeuronMap.git
cd NeuronMap

# Create development environment
python -m venv dev_env
source dev_env/bin/activate  # On Windows: dev_env\Scripts\activate

# Install development dependencies
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install -r requirements_dev.txt

# Install pre-commit hooks
pre-commit install

# Install in editable mode
pip install -e .

# Run tests to verify setup
python -m pytest tests/
python validate_section_1_1.py
python validate_section_1_2.py
```

## 📦 Installing Additional Dependencies

### For Advanced Visualization

```bash
pip install plotly dash bokeh altair
```

### For Large Model Support

```bash
pip install accelerate bitsandbytes
```

### For Distributed Computing

```bash
pip install ray dask distributed
```

### For Database Integration

```bash
pip install sqlalchemy psycopg2-binary pymongo
```

## ✅ Verification

After installation, verify everything works correctly:

```bash
# Basic functionality test
python -c "
from src.utils.config import get_config_manager
from src.analysis.activation_extractor import ActivationExtractor
from src.visualization.core_visualizer import CoreVisualizer

config = get_config_manager()
print('✅ Configuration system working')

extractor = ActivationExtractor()
print('✅ Analysis module working')

visualizer = CoreVisualizer()
print('✅ Visualization module working')

print('🎉 All systems operational!')
"

# Run comprehensive validation
python validate_section_1_1.py
python validate_section_1_2.py

# Test CLI interface
python -m src.analysis.activation_extractor --help
python -m src.data_generation.question_generator --help
```

## 🚨 Troubleshooting

### Common Issues

**ImportError: No module named 'src'**
```bash
# Make sure you're in the project root directory
pwd  # Should show .../neuronmap
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**CUDA out of memory**
```bash
# Reduce batch size in configuration
python -c "
from src.utils.config import get_config_manager
config = get_config_manager()
analysis_config = config.get_analysis_config()
analysis_config.batch_size = 4  # Reduce from default
"
```

**Permission denied errors**
```bash
# Fix permissions
chmod +x setup.sh
sudo chown -R $USER:$USER neuronmap/
```

### Getting Help

- **GitHub Issues**: [Report installation problems](https://github.com/Emilio942/NeuronMap/issues)
- **Discussions**: [Community support](https://github.com/Emilio942/NeuronMap/discussions)
- **Documentation**: [Complete guides](https://neuronmap.readthedocs.io)

## 🔄 Updating NeuronMap

```bash
# Navigate to project directory
cd NeuronMap

# Pull latest changes
git pull origin main

# Update dependencies
pip install --upgrade -r requirements.txt

# Re-install in development mode
pip install -e .

# Verify update
python -c "import src; print(f'Version: {src.__version__}')"
```

## 🗑 Uninstallation

```bash
# Remove virtual environment
rm -rf neuronmap_env/

# Remove project directory
rm -rf neuronmap/

# Remove any global installations
pip uninstall neuronmap
```
