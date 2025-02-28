# PABKit: Process-Aware Benchmarking Toolkit

## Overview
PABKit is a multi-framework benchmarking toolkit for evaluating machine learning models based on their learning **process** rather than just final accuracy. It supports **PyTorch**, **TensorFlow**, and **JAX**, allowing users to track learning stability, generalization efficiency, and rule evolution across different models.

## Features
- **Multi-Framework Support:** Compatible with PyTorch, TensorFlow, and JAX.
- **Process-Aware Metrics:** Tracks learning stability, memorization vs. generalization, and rule evolution.
- **Flexible Experiment Runner:** Unified API for training and evaluation across different frameworks.
- **Visualization Tools:** Generates learning curve plots and generalization comparisons.
- **CI/CD Integration:** Supports automated testing and publishing via GitHub Actions.
- **Modular Codebase:** Backend-specific implementations for each framework.
- **Package Management via Poetry & setup.py:** Hybrid installation support.

## Installation

### Option 1: Install Core Package
```sh
pip install pabkit
```

### Option 2: Install with Specific Frameworks
```sh
pip install pabkit[pytorch]      # PyTorch support
pip install pabkit[tensorflow]   # TensorFlow support
pip install pabkit[jax]          # JAX support
```

### Option 3: Install via Poetry
```sh
poetry add pabkit
```

## Usage

### Train a Model
```python
from pab.experiment import train_model
from torchvision import models

# Load a PyTorch model
model = models.resnet50(pretrained=True)
train_loader, val_loader = load_dataset("pytorch")

# Train with PABKit
trained_model = train_model(model, train_loader, val_loader, framework="pytorch", epochs=10)
```

### Track Learning Progress
```python
from pab.metrics import ProcessAwareMetrics

stability = ProcessAwareMetrics.learning_stability(training_checkpoints)
print(f"Learning Stability: {stability}")
```

### Run Experiments via CLI
```sh
pabkit-train --framework pytorch --epochs 10
```

## Development Setup

### Clone Repository
```sh
git clone https://github.com/yourgithub/pabkit.git
cd pabkit
```

### Install Dependencies
```sh
poetry install  # For Poetry users
# OR
pip install -e .  # For Pip users
```

### Run Tests
```sh
pytest tests/
```

## Project Structure
```
pabkit/
│── pab/                        # Main package directory
│   │── __init__.py             # Exposes package functionality
│   │── core.py                 # Core benchmarking functions
│   │── metrics.py              # Learning stability, rule evolution
│   │── visualization.py        # Learning curves & plots
│   │── dataset.py              # Data loading utilities
│   │── experiment.py           # Experiment manager
│   │── backends/               # Framework-specific implementations
│   │   │── pytorch_backend.py
│   │   │── tensorflow_backend.py
│   │   │── jax_backend.py
│── scripts/                    # CLI tools
│── tests/                      # Unit tests
│── docs/                       # Documentation
│── .github/workflows/