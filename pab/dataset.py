import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state
from torchvision import datasets, transforms, models
from tqdm import tqdm
from loguru import logger

from pab.backends.pytorch_backend import train_pytorch_model
from pab.backends.tensorflow_backend import train_tensorflow_model
from pab.backends.jax_backend import train_jax_model
from pab.dataset import load_dataset
from pab.metrics import ProcessAwareMetrics
from pab.visualization import plot_learning_curve

#Unified Experiment Runner

def train_model(model: any, train_loader: any, val_loader: any, framework: str = "pytorch", epochs: int = 10, device: str = 'cuda'):
    """Train model using specified framework (PyTorch, TensorFlow, or JAX)."""
    logger.info(f"Training model using {framework} framework for {epochs} epochs.")
    if framework == "pytorch":
        return train_pytorch_model(model, train_loader, val_loader, epochs, device)
    elif framework == "tensorflow":
        return train_tensorflow_model(model, train_loader, val_loader, epochs)
    elif framework == "jax":
        return train_jax_model(model, train_loader, val_loader, epochs)
    else:
        raise ValueError(f"Unsupported framework: {framework}")

if __name__ == '__main__':
    framework = os.getenv("PAB_FRAMEWORK", "pytorch")  # Default to PyTorch
    train_loader, val_loader = load_dataset(framework)

    if framework == "pytorch":
        model = models.resnet50(pretrained=True)
    elif framework == "tensorflow":
        model = tf.keras.applications.ResNet50(weights='imagenet')
    elif framework == "jax":
        class JaxModel(nn.Module):
            @nn.compact
            def __call__(self, x):
                x = nn.Conv(features=32, kernel_size=(3, 3))(x)
                x = nn.relu(x)
                x = nn.Dense(features=10)(x)
                return x
        model = JaxModel()

    trained_model = train_model(model, train_loader, val_loader, framework, epochs=10)
    logger.info("Training Complete.")

