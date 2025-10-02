"""
Sparse Autoencoder (SAE) Training Pipeline

This module implements a complete pipeline for training Sparse Autoencoders
on transformer model activations to discover interpretable features.
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from .model_integration import ModelManager
from ..data_processing.dataset_loader import OpenWebTextLoader

# Import config from utils
try:
    from ..utils.config import AnalysisConfig
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.config import AnalysisConfig

logger = logging.getLogger(__name__)


@dataclass
class SAEConfig:
    """Configuration for Sparse Autoencoder training."""
    # Target model and layer
    model_name: str
    layer: int
    component: str  # 'mlp', 'attention', 'residual'

    # Model architecture
    input_dim: int
    hidden_dim: int
    sparsity_penalty: float = 1e-3
    
    # Training parameters
    learning_rate: float = 1e-3
    batch_size: int = 1024
    num_epochs: int = 100
    warmup_steps: int = 1000
    
    # Data parameters
    max_sequences: int = 10000
    sequence_length: int = 128
    
    # Regularization
    weight_decay: float = 1e-4
    dropout_rate: float = 0.1
    grad_clip_norm: float = 1.0
    
    # Logging and checkpointing
    log_interval: int = 100
    checkpoint_interval: int = 1000
    validation_split: float = 0.1
    
    # Output
    output_dir: str = "outputs/sae_models"
    model_name_prefix: str = "sae"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'model_name': self.model_name,
            'layer': self.layer,
            'component': self.component,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'sparsity_penalty': self.sparsity_penalty,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'warmup_steps': self.warmup_steps,
            'max_sequences': self.max_sequences,
            'sequence_length': self.sequence_length,
            'weight_decay': self.weight_decay,
            'dropout_rate': self.dropout_rate,
            'grad_clip_norm': self.grad_clip_norm,
            'log_interval': self.log_interval,
            'checkpoint_interval': self.checkpoint_interval,
            'validation_split': self.validation_split,
            'output_dir': self.output_dir,
            'model_name_prefix': self.model_name_prefix
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SAEConfig':
        """Create from dictionary."""
        # Filter out activation_collection_layers if it exists in old configs
        data.pop('activation_collection_layers', None)
        return cls(**data)


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder for learning interpretable features from activations."""
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Encoder: maps input to sparse features
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Decoder: reconstructs input from sparse features
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout_rate)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse feature representation."""
        return self.encoder(x)
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode features back to input space."""
        return self.decoder(features)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.
        
        Returns:
            Tuple of (reconstructed_input, sparse_features)
        """
        features = self.encode(x)
        reconstruction = self.decode(features)
        return reconstruction, features
    
    def get_feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Get sparse feature activations for input."""
        with torch.no_grad():
            features = self.encode(x)
        return features


class ActivationDataset(Dataset):
    """Dataset for model activations."""
    
    def __init__(self, activations: torch.Tensor):
        """
        Initialize dataset with activations.
        
        Args:
            activations: Tensor of shape [num_samples, feature_dim]
        """
        self.activations = activations
    
    def __len__(self) -> int:
        return len(self.activations)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.activations[idx]


@dataclass
class SAETrainingResult:
    """Result of SAE training."""
    model: SparseAutoencoder
    config: SAEConfig
    training_history: Dict[str, List[float]]
    final_metrics: Dict[str, float]
    model_path: str
    training_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding the model)."""
        return {
            'config': self.config.to_dict(),
            'training_history': self.training_history,
            'final_metrics': self.final_metrics,
            'model_path': self.model_path,
            'training_time': self.training_time
        }


class SAETrainer:
    """Trainer for Sparse Autoencoders."""
    
    def __init__(self, config: SAEConfig, device: Optional[torch.device] = None):
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model: Optional[SparseAutoencoder] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'train_reconstruction_loss': [],
            'train_sparsity_loss': [],
            'val_loss': [],
            'val_reconstruction_loss': [],
            'val_sparsity_loss': [],
            'learning_rate': []
        }
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def collect_activations(
        self,
        model_manager: ModelManager,
        texts: List[str],
    ) -> torch.Tensor:
        """
        Collect activations from the model for training data.
        
        Args:
            model_manager: ModelManager instance
            texts: List of texts to process
            
        Returns:
            Tensor of activations [num_samples, feature_dim]
        """
        logger.info(f"Collecting activations from {len(texts)} texts for model {self.config.model_name}, layer {self.config.layer}, component {self.config.component}")
        
        # Load the model
        adapter = model_manager.load_model(self.config.model_name)
        model = adapter.model
        
        all_activations = []
        activations_cache = {}
        hooks = []
        
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                activations_cache[name] = output.detach()
            return hook
        
        # Determine the module to hook based on component and layer
        target_module = None
        layer_idx = self.config.layer
        
        if hasattr(model, 'transformer'):  # GPT-style models
            if self.config.component == 'residual':
                # Output of the transformer block (residual stream)
                target_module = model.transformer.h[layer_idx]
            elif self.config.component == 'mlp':
                # Output of the MLP layer (after activation function)
                target_module = model.transformer.h[layer_idx].mlp
            elif self.config.component == 'attention':
                # Output of the attention layer
                target_module = model.transformer.h[layer_idx].attn
            else:
                raise ValueError(f"Unsupported component for GPT-style model: {self.config.component}")
        elif hasattr(model, 'encoder'):  # BERT-style models
            if self.config.component == 'residual':
                target_module = model.encoder.layer[layer_idx]
            elif self.config.component == 'mlp':
                target_module = model.encoder.layer[layer_idx].intermediate
            elif self.config.component == 'attention':
                target_module = model.encoder.layer[layer_idx].attention.output
            else:
                raise ValueError(f"Unsupported component for BERT-style model: {self.config.component}")
        else:
            raise ValueError(f"Unknown model architecture for {self.config.model_name}")

        if target_module is None:
            raise ValueError(f"Could not find target module for {self.config.model_name}, layer {self.config.layer}, component {self.config.component}")

        hook = target_module.register_forward_hook(make_hook(f'layer_{layer_idx}_{self.config.component}'))
        hooks.append(hook)
        
        try:
            # Process texts in batches
            batch_size = 8  # Small batch size to avoid memory issues
            
            for i in tqdm(range(0, len(texts), batch_size), desc="Collecting activations"):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                inputs = adapter.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.sequence_length
                ).to(self.device)
                
                # Forward pass
                with torch.no_grad():
                    _ = model(**inputs)
                
                # Collect activations
                for layer_name, activations in activations_cache.items():
                    # Flatten sequence dimension: [batch, seq, hidden] -> [batch*seq, hidden]
                    flat_activations = activations.view(-1, activations.size(-1))
                    all_activations.append(flat_activations.cpu())
                
                activations_cache.clear()
                
                # Memory cleanup
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
        
        # Concatenate all activations
        if all_activations:
            combined_activations = torch.cat(all_activations, dim=0)
            logger.info(f"Collected {combined_activations.shape[0]} activation samples with dimension {combined_activations.shape[1]}")
            
            # Verify input_dim
            if self.config.input_dim != combined_activations.shape[1]:
                logger.warning(f"Configured input_dim ({self.config.input_dim}) does not match collected activation dimension ({combined_activations.shape[1]}). Adjusting SAEConfig.input_dim.")
                self.config.input_dim = combined_activations.shape[1]

            return combined_activations
        else:
            raise ValueError("No activations collected")
    
    def create_data_loaders(
        self,
        activations: torch.Tensor
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Create training and validation data loaders."""
        # Shuffle and split data
        num_samples = len(activations)
        indices = torch.randperm(num_samples)
        
        if self.config.validation_split > 0:
            val_size = int(num_samples * self.config.validation_split)
            train_indices = indices[val_size:]
            val_indices = indices[:val_size]
            
            train_dataset = ActivationDataset(activations[train_indices])
            val_dataset = ActivationDataset(activations[val_indices])
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            
            return train_loader, val_loader
        else:
            train_dataset = ActivationDataset(activations)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            return train_loader, None
    
    def compute_loss(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute SAE loss: reconstruction + sparsity penalty.
        
        Returns:
            Tuple of (total_loss, reconstruction_loss, sparsity_loss)
        """
        # Reconstruction loss (MSE)
        reconstruction_loss = nn.functional.mse_loss(reconstruction, target)
        
        # Sparsity loss (L1 penalty on features)
        sparsity_loss = torch.mean(torch.abs(features))
        
        # Total loss
        total_loss = reconstruction_loss + self.config.sparsity_penalty * sparsity_loss
        
        return total_loss, reconstruction_loss, sparsity_loss
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_sparsity_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(self.device)
            
            # Forward pass
            reconstruction, features = self.model(batch)
            
            # Compute loss
            loss, recon_loss, sparsity_loss = self.compute_loss(
                reconstruction, batch, features
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip_norm
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_sparsity_loss += sparsity_loss.item()
            num_batches += 1
            
            # Logging
            if batch_idx % self.config.log_interval == 0:
                logger.info(
                    f"Batch {batch_idx}/{len(train_loader)}: "
                    f"Loss={loss.item():.4f}, "
                    f"Recon={recon_loss.item():.4f}, "
                    f"Sparsity={sparsity_loss.item():.4f}"
                )
        
        return {
            'loss': total_loss / num_batches,
            'reconstruction_loss': total_recon_loss / num_batches,
            'sparsity_loss': total_sparsity_loss / num_batches
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_sparsity_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                
                # Forward pass
                reconstruction, features = self.model(batch)
                
                # Compute loss
                loss, recon_loss, sparsity_loss = self.compute_loss(
                    reconstruction, batch, features
                )
                
                # Accumulate losses
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_sparsity_loss += sparsity_loss.item()
                num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'reconstruction_loss': total_recon_loss / num_batches,
            'sparsity_loss': total_sparsity_loss / num_batches
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> str:
        """Save model checkpoint."""
        checkpoint_path = Path(self.config.output_dir) / f"{self.config.model_name_prefix}_epoch_{epoch}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.to_dict(),
            'metrics': metrics,
            'training_history': self.training_history
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)
    
    def train(
        self,
        model_manager: ModelManager,
    ) -> SAETrainingResult:
        """
        Complete training pipeline.
        
        Args:
            model_manager: ModelManager instance
            
        Returns:
            Training result with model and metrics
        """
        start_time = time.time()
        logger.info("Starting SAE training pipeline")
        
        # Load texts using OpenWebTextLoader
        data_loader = OpenWebTextLoader(num_samples=self.config.max_sequences)
        texts = data_loader.stream_texts()

        # Collect activations
        activations = self.collect_activations(model_manager)
        
        # Initialize model with correct input_dim
        self.model = SparseAutoencoder(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            dropout_rate=self.config.dropout_rate
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(activations)

        # Learning rate scheduler
        total_steps = self.config.num_epochs * len(train_loader)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=0.1
        )
        
        # Training loop
        best_val_loss = float('inf')
        best_model_path = None
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
            
            # Update history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['train_reconstruction_loss'].append(train_metrics['reconstruction_loss'])
            self.training_history['train_sparsity_loss'].append(train_metrics['sparsity_loss'])
            self.training_history['learning_rate'].append(self.scheduler.get_last_lr()[0])
            
            if val_metrics:
                self.training_history['val_loss'].append(val_metrics['loss'])
                self.training_history['val_reconstruction_loss'].append(val_metrics['reconstruction_loss'])
                self.training_history['val_sparsity_loss'].append(val_metrics['sparsity_loss'])
            
            # Log epoch results
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
            if val_metrics:
                logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % (self.config.checkpoint_interval // len(train_loader)) == 0:
                checkpoint_path = self.save_checkpoint(epoch, {**train_metrics, **val_metrics})
                
                # Update best model
                current_val_loss = val_metrics.get('loss', train_metrics['loss'])
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    best_model_path = checkpoint_path
        
        # Save final model
        final_model_path = Path(self.config.output_dir) / f"{self.config.model_name_prefix}_final.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.to_dict(),
            'training_history': self.training_history
        }, final_model_path)
        
        training_time = time.time() - start_time
        
        # Prepare final metrics
        final_metrics = {
            'best_val_loss': best_val_loss,
            'final_train_loss': self.training_history['train_loss'][-1],
            'training_time': training_time,
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'sparsity_achieved': self._compute_sparsity_metrics(activations[:1000])  # Sample for efficiency
        }
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return SAETrainingResult(
            model=self.model,
            config=self.config,
            training_history=self.training_history,
            final_metrics=final_metrics,
            model_path=str(final_model_path),
            training_time=training_time
        )
    
    def _compute_sparsity_metrics(self, sample_activations: torch.Tensor) -> Dict[str, float]:
        """Compute sparsity metrics on sample activations."""
        self.model.eval()
        with torch.no_grad():
            sample_activations = sample_activations.to(self.device)
            _, features = self.model(sample_activations)
            
            # Compute sparsity metrics
            mean_activation = torch.mean(torch.abs(features))
            fraction_active = torch.mean((torch.abs(features) > 1e-6).float())
            
            return {
                'mean_feature_activation': float(mean_activation),
                'fraction_features_active': float(fraction_active),
                'effective_sparsity': float(1.0 - fraction_active)
            }
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss curves
        axes[0, 0].plot(self.training_history['train_loss'], label='Train')
        if self.training_history['val_loss']:
            axes[0, 0].plot(self.training_history['val_loss'], label='Validation')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Reconstruction loss
        axes[0, 1].plot(self.training_history['train_reconstruction_loss'], label='Train')
        if self.training_history['val_reconstruction_loss']:
            axes[0, 1].plot(self.training_history['val_reconstruction_loss'], label='Validation')
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Sparsity loss
        axes[1, 0].plot(self.training_history['train_sparsity_loss'], label='Train')
        if self.training_history['val_sparsity_loss']:
            axes[1, 0].plot(self.training_history['val_sparsity_loss'], label='Validation')
        axes[1, 0].set_title('Sparsity Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        
        # Learning rate
        axes[1, 1].plot(self.training_history['learning_rate'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('LR')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved: {save_path}")
        
        return fig


# Utility functions
def load_sae_model(model_path: str, device: Optional[torch.device] = None) -> Tuple[SparseAutoencoder, SAEConfig]:
    """Load a trained SAE model."""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(model_path, map_location=device)
    config = SAEConfig.from_dict(checkpoint['config'])
    
    model = SparseAutoencoder(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        dropout_rate=config.dropout_rate
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config


def create_default_sae_config(model_name: str, layer: int, component: str, input_dim: Optional[int] = None, hidden_dim: Optional[int] = None) -> SAEConfig:
    """Create a default SAE configuration."""
    if input_dim is None:
        input_dim = 768  # Default to common hidden size, will be updated dynamically
    if hidden_dim is None:
        hidden_dim = input_dim * 4  # Standard expansion factor
    
    return SAEConfig(
        model_name=model_name,
        layer=layer,
        component=component,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        sparsity_penalty=1e-3,
        learning_rate=1e-3,
        batch_size=1024,
        num_epochs=50,
        max_sequences=5000
    )
