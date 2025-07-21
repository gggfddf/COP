"""
Autonomous Gold Trading AI - Hybrid Neural Architecture
Combines Transformer, CNN, and LSTM with auxiliary modules:
- Contrastive Learning Block
- Quantile Regression/MC Dropout
- Outlier Movement Classifier
- Pattern Memory Encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from typing import Dict, List, Tuple, Optional
import yaml
import logging
from transformers import TransformerModel
import math

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for Transformer"""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Linear transformations and reshape
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)
        
        # Reshape and apply output linear layer
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_dim
        )
        
        return self.output_linear(attended)

class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    
    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attended = self.attention(x, mask)
        x = self.norm1(x + attended)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x

class CNNFeatureExtractor(nn.Module):
    """CNN component for pattern detection over raw OHLCV windows"""
    
    def __init__(self, input_channels: int, filters: List[int], kernel_sizes: List[int]):
        super().__init__()
        
        self.conv_layers = nn.ModuleList()
        
        in_channels = input_channels
        for i, (out_channels, kernel_size) in enumerate(zip(filters, kernel_sizes)):
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.MaxPool1d(2)
            )
            self.conv_layers.append(conv_block)
            in_channels = out_channels
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, features)
        x = x.transpose(1, 2)  # (batch_size, features, seq_len)
        
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Global average pooling
        x = self.global_pool(x).squeeze(-1)  # (batch_size, filters[-1])
        
        return x

class LSTMEncoder(nn.Module):
    """LSTM component for sequential dependencies and cycles"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 dropout: float = 0.2, bidirectional: bool = True):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        hidden = self.dropout(hidden)
        
        return lstm_out, hidden

class ContrastiveLearningBlock(nn.Module):
    """Contrastive Learning Block for high move vs low move classification"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 64)
        )
        
        self.temperature = nn.Parameter(torch.tensor(0.07))
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # Project embeddings to contrastive space
        projections = self.projector(embeddings)
        
        # L2 normalize
        projections = F.normalize(projections, dim=1)
        
        return projections
    
    def contrastive_loss(self, projections: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss"""
        batch_size = projections.shape[0]
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(projections, projections.T) / self.temperature
        
        # Create mask for positive pairs
        labels = labels.unsqueeze(1)
        positive_mask = torch.eq(labels, labels.T).float()
        
        # Remove diagonal (self-similarity)
        positive_mask = positive_mask - torch.eye(batch_size, device=projections.device)
        
        # Compute loss
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # Average over positive pairs
        loss = -(positive_mask * log_prob).sum(dim=1) / (positive_mask.sum(dim=1) + 1e-8)
        
        return loss.mean()

class QuantileRegressionHead(nn.Module):
    """Quantile regression head for uncertainty estimation"""
    
    def __init__(self, input_dim: int, num_quantiles: int = 5):
        super().__init__()
        
        self.num_quantiles = num_quantiles
        self.quantiles = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
        
        self.quantile_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(input_dim // 2, 1)
            ) for _ in range(num_quantiles)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quantile_preds = []
        
        for layer in self.quantile_layers:
            quantile_preds.append(layer(x))
        
        return torch.cat(quantile_preds, dim=1)  # (batch_size, num_quantiles)
    
    def quantile_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute quantile loss"""
        losses = []
        
        for i, quantile in enumerate(self.quantiles):
            pred = predictions[:, i]
            error = targets - pred
            loss = torch.max(quantile * error, (quantile - 1) * error)
            losses.append(loss.mean())
        
        return torch.stack(losses).mean()

class OutlierMovementClassifier(nn.Module):
    """Classifier for detecting 5%+ upcoming moves"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 2)  # Binary: outlier or not
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

class PatternMemoryEncoder(nn.Module):
    """Pattern Memory Encoder for identifying recurring candle structures"""
    
    def __init__(self, input_dim: int, memory_size: int = 1000, pattern_dim: int = 64):
        super().__init__()
        
        self.memory_size = memory_size
        self.pattern_dim = pattern_dim
        
        # Pattern encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, pattern_dim)
        )
        
        # Memory bank
        self.register_buffer('memory_bank', torch.randn(memory_size, pattern_dim))
        self.register_buffer('memory_labels', torch.zeros(memory_size))
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
        
        # Pattern classifier
        self.pattern_classifier = nn.Sequential(
            nn.Linear(pattern_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 50)  # 50 pattern classes
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode pattern
        pattern_encoding = self.encoder(x)
        pattern_encoding = F.normalize(pattern_encoding, dim=1)
        
        # Classify pattern
        pattern_logits = self.pattern_classifier(pattern_encoding)
        
        return pattern_encoding, pattern_logits
    
    def update_memory(self, encodings: torch.Tensor, labels: torch.Tensor):
        """Update memory bank with new patterns"""
        batch_size = encodings.shape[0]
        
        ptr = int(self.memory_ptr)
        
        if ptr + batch_size <= self.memory_size:
            self.memory_bank[ptr:ptr + batch_size] = encodings.detach()
            self.memory_labels[ptr:ptr + batch_size] = labels.detach()
            ptr = (ptr + batch_size) % self.memory_size
        else:
            # Wrap around
            remaining = self.memory_size - ptr
            self.memory_bank[ptr:] = encodings[:remaining].detach()
            self.memory_labels[ptr:] = labels[:remaining].detach()
            
            overflow = batch_size - remaining
            self.memory_bank[:overflow] = encodings[remaining:].detach()
            self.memory_labels[:overflow] = labels[remaining:].detach()
            ptr = overflow
        
        self.memory_ptr[0] = ptr

class HybridTransformerCNNLSTM(pl.LightningModule):
    """
    Hybrid neural architecture combining Transformer, CNN, and LSTM
    with all auxiliary modules for autonomous Gold trading
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        super().__init__()
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.training_config = self.config['training']
        
        # Model dimensions
        self.input_dim = 100  # Will be set based on feature engineering output
        self.sequence_length = self.model_config['input_sequence_length']
        self.hidden_dim = self.model_config['transformer']['hidden_dim']
        
        # Input projection
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        
        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(
            self.sequence_length, self.hidden_dim
        )
        
        # TRANSFORMER COMPONENT
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.model_config['transformer']['num_heads'],
                ff_dim=self.hidden_dim * 4,
                dropout=self.model_config['transformer']['dropout']
            ) for _ in range(self.model_config['transformer']['num_layers'])
        ])
        
        # CNN COMPONENT
        self.cnn_extractor = CNNFeatureExtractor(
            input_channels=self.input_dim,
            filters=self.model_config['cnn']['filters'],
            kernel_sizes=self.model_config['cnn']['kernel_sizes']
        )
        
        # LSTM COMPONENT
        self.lstm_encoder = LSTMEncoder(
            input_size=self.hidden_dim,
            hidden_size=self.model_config['lstm']['hidden_size'],
            num_layers=self.model_config['lstm']['num_layers'],
            dropout=self.model_config['lstm']['dropout'],
            bidirectional=self.model_config['lstm']['bidirectional']
        )
        
        # Feature fusion
        lstm_output_dim = (
            self.model_config['lstm']['hidden_size'] * 2 
            if self.model_config['lstm']['bidirectional'] 
            else self.model_config['lstm']['hidden_size']
        )
        cnn_output_dim = self.model_config['cnn']['filters'][-1]
        
        self.fusion_dim = self.hidden_dim + lstm_output_dim + cnn_output_dim
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.fusion_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # AUXILIARY MODULES
        self.contrastive_block = ContrastiveLearningBlock(self.hidden_dim)
        self.quantile_head = QuantileRegressionHead(self.hidden_dim)
        self.outlier_classifier = OutlierMovementClassifier(self.hidden_dim)
        self.pattern_memory = PatternMemoryEncoder(self.hidden_dim)
        
        # OUTPUT HEADS
        self.direction_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, self.model_config['output']['direction_classes'])
        )
        
        self.magnitude_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, len(self.model_config['prediction_horizon']))
        )
        
        self.volatility_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, self.model_config['output']['volatility_classes'])
        )
        
        # Loss weights
        self.loss_weights = {
            'direction': 1.0,
            'magnitude': 1.0,
            'volatility': 0.5,
            'contrastive': 0.3,
            'quantile': 0.5,
            'outlier': 0.7,
            'pattern': 0.4
        }
        
        # Metrics storage
        self.save_hyperparameters()
        
    def _create_positional_encoding(self, seq_len: int, hidden_dim: int) -> torch.Tensor:
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(seq_len, hidden_dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * 
                           (-math.log(10000.0) / hidden_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # (1, seq_len, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hybrid architecture
        
        Args:
            x: Input tensor (batch_size, seq_len, features)
            
        Returns:
            Dictionary with all predictions and embeddings
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x_proj = self.input_projection(x)
        
        # Add positional encoding
        if self.positional_encoding.device != x.device:
            self.positional_encoding = self.positional_encoding.to(x.device)
        x_proj = x_proj + self.positional_encoding[:, :seq_len, :]
        
        # TRANSFORMER PATH
        transformer_out = x_proj
        for transformer_layer in self.transformer_layers:
            transformer_out = transformer_layer(transformer_out)
        
        # Take last sequence output for transformer
        transformer_features = transformer_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # CNN PATH
        cnn_features = self.cnn_extractor(x)  # (batch_size, cnn_output_dim)
        
        # LSTM PATH
        lstm_out, lstm_hidden = self.lstm_encoder(transformer_out)
        lstm_features = lstm_hidden  # (batch_size, lstm_output_dim)
        
        # FEATURE FUSION
        fused_features = torch.cat([
            transformer_features, 
            lstm_features, 
            cnn_features
        ], dim=1)
        
        fused_features = self.feature_fusion(fused_features)
        
        # AUXILIARY MODULES
        contrastive_proj = self.contrastive_block(fused_features)
        quantile_preds = self.quantile_head(fused_features)
        outlier_preds = self.outlier_classifier(fused_features)
        pattern_encoding, pattern_logits = self.pattern_memory(fused_features)
        
        # MAIN PREDICTIONS
        direction_logits = self.direction_head(fused_features)
        magnitude_preds = self.magnitude_head(fused_features)
        volatility_logits = self.volatility_head(fused_features)
        
        return {
            'direction_logits': direction_logits,
            'magnitude_preds': magnitude_preds,
            'volatility_logits': volatility_logits,
            'contrastive_proj': contrastive_proj,
            'quantile_preds': quantile_preds,
            'outlier_preds': outlier_preds,
            'pattern_encoding': pattern_encoding,
            'pattern_logits': pattern_logits,
            'embeddings': fused_features
        }
    
    def training_step(self, batch, batch_idx):
        """Training step with multi-task loss"""
        x, targets = batch
        
        # Forward pass
        outputs = self(x)
        
        # Compute losses
        losses = self._compute_losses(outputs, targets)
        
        # Total loss
        total_loss = sum(
            self.loss_weights[key] * loss 
            for key, loss in losses.items()
        )
        
        # Log losses
        for key, loss in losses.items():
            self.log(f'train_loss_{key}', loss, prog_bar=True)
        self.log('train_loss_total', total_loss, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        x, targets = batch
        outputs = self(x)
        losses = self._compute_losses(outputs, targets)
        
        total_loss = sum(
            self.loss_weights[key] * loss 
            for key, loss in losses.items()
        )
        
        # Log validation losses
        for key, loss in losses.items():
            self.log(f'val_loss_{key}', loss)
        self.log('val_loss_total', total_loss)
        
        # Compute metrics
        metrics = self._compute_metrics(outputs, targets)
        for key, metric in metrics.items():
            self.log(f'val_{key}', metric)
        
        return total_loss
    
    def _compute_losses(self, outputs: Dict[str, torch.Tensor], 
                       targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute all losses"""
        losses = {}
        
        # Direction loss (cross entropy)
        if 'direction' in targets:
            losses['direction'] = F.cross_entropy(
                outputs['direction_logits'], 
                targets['direction']
            )
        
        # Magnitude loss (MSE)
        if 'magnitude' in targets:
            losses['magnitude'] = F.mse_loss(
                outputs['magnitude_preds'], 
                targets['magnitude']
            )
        
        # Volatility loss (cross entropy)
        if 'volatility' in targets:
            losses['volatility'] = F.cross_entropy(
                outputs['volatility_logits'], 
                targets['volatility']
            )
        
        # Contrastive loss
        if 'high_move_labels' in targets:
            losses['contrastive'] = self.contrastive_block.contrastive_loss(
                outputs['contrastive_proj'], 
                targets['high_move_labels']
            )
        
        # Quantile loss
        if 'magnitude' in targets:
            losses['quantile'] = self.quantile_head.quantile_loss(
                outputs['quantile_preds'], 
                targets['magnitude'][:, 0]  # Use first horizon
            )
        
        # Outlier loss
        if 'outlier_labels' in targets:
            losses['outlier'] = F.cross_entropy(
                outputs['outlier_preds'], 
                targets['outlier_labels']
            )
        
        # Pattern loss
        if 'pattern_labels' in targets:
            losses['pattern'] = F.cross_entropy(
                outputs['pattern_logits'], 
                targets['pattern_labels']
            )
        
        return losses
    
    def _compute_metrics(self, outputs: Dict[str, torch.Tensor], 
                        targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute evaluation metrics"""
        metrics = {}
        
        # Direction accuracy
        if 'direction' in targets:
            direction_preds = torch.argmax(outputs['direction_logits'], dim=1)
            metrics['direction_accuracy'] = (
                direction_preds == targets['direction']
            ).float().mean()
        
        # Magnitude MAE
        if 'magnitude' in targets:
            metrics['magnitude_mae'] = F.l1_loss(
                outputs['magnitude_preds'], 
                targets['magnitude']
            )
        
        # Outlier F1 score (simplified)
        if 'outlier_labels' in targets:
            outlier_preds = torch.argmax(outputs['outlier_preds'], dim=1)
            tp = ((outlier_preds == 1) & (targets['outlier_labels'] == 1)).float().sum()
            fp = ((outlier_preds == 1) & (targets['outlier_labels'] == 0)).float().sum()
            fn = ((outlier_preds == 0) & (targets['outlier_labels'] == 1)).float().sum()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            metrics['outlier_f1'] = 2 * precision * recall / (precision + recall + 1e-8)
        
        return metrics
    
    def predict_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        """Prediction step for inference"""
        x = batch if isinstance(batch, torch.Tensor) else batch[0]
        
        with torch.no_grad():
            outputs = self(x)
            
            # Convert logits to probabilities
            direction_probs = F.softmax(outputs['direction_logits'], dim=1)
            volatility_probs = F.softmax(outputs['volatility_logits'], dim=1)
            outlier_probs = F.softmax(outputs['outlier_preds'], dim=1)
            pattern_probs = F.softmax(outputs['pattern_logits'], dim=1)
            
            predictions = {
                'direction_probs': direction_probs,
                'direction_pred': torch.argmax(direction_probs, dim=1),
                'magnitude_pred': outputs['magnitude_preds'],
                'volatility_probs': volatility_probs,
                'volatility_pred': torch.argmax(volatility_probs, dim=1),
                'quantile_preds': outputs['quantile_preds'],
                'outlier_probs': outlier_probs,
                'outlier_pred': torch.argmax(outlier_probs, dim=1),
                'pattern_probs': pattern_probs,
                'pattern_pred': torch.argmax(pattern_probs, dim=1),
                'confidence': torch.max(direction_probs, dim=1)[0],
                'embeddings': outputs['embeddings']
            }
        
        return predictions
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.training_config['learning_rate'],
            weight_decay=self.training_config['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.training_config['epochs']
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss_total'
            }
        }
    
    def get_pattern_similarities(self, embeddings: torch.Tensor, 
                                top_k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get most similar patterns from memory bank"""
        with torch.no_grad():
            # Normalize embeddings
            embeddings = F.normalize(embeddings, dim=1)
            
            # Compute similarities with memory bank
            similarities = torch.matmul(embeddings, self.pattern_memory.memory_bank.T)
            
            # Get top-k similar patterns
            top_similarities, top_indices = torch.topk(similarities, top_k, dim=1)
            
        return top_similarities, top_indices


# Example usage and testing
if __name__ == "__main__":
    # Create model
    model = HybridTransformerCNNLSTM()
    
    # Test forward pass
    batch_size = 32
    seq_len = 100
    features = 85  # From comprehensive feature engineering
    
    x = torch.randn(batch_size, seq_len, features)
    
    # Update input dimension
    model.input_dim = features
    model.input_projection = nn.Linear(features, model.hidden_dim)
    
    outputs = model(x)
    
    print("🔥 Hybrid Model Architecture Created Successfully!")
    print(f"✅ Transformer layers: {len(model.transformer_layers)}")
    print(f"✅ CNN feature extractor: {model.cnn_extractor}")
    print(f"✅ LSTM encoder: {model.lstm_encoder}")
    print(f"✅ Contrastive learning: {model.contrastive_block}")
    print(f"✅ Quantile regression: {model.quantile_head}")
    print(f"✅ Outlier classifier: {model.outlier_classifier}")
    print(f"✅ Pattern memory: {model.pattern_memory}")
    
    print(f"\nOutput shapes:")
    for key, tensor in outputs.items():
        print(f"  {key}: {tensor.shape}")
    
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")