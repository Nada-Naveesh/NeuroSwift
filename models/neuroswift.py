import torch
import torch.nn as nn
from models.multiscale import MultiScaleCNN
from models.attention import ECABlock

DEFAULT_CONFIG = {
    'input_channels': 64,
    'input_time': 640,
    'num_classes': 5,
    'kernel_sizes': [3, 5, 7],
    'attention_gamma': 2,
    'attention_b': 1,
}

class NeuroSwiftModel(nn.Module):
    """
    NeuroSwift: Motor Imagery Classification Model
    - Multi-scale feature extraction
    - Efficient Channel Attention (ECA)
    - Optimized for 90-95% accuracy on PhysioNet
    """
    def __init__(self, config=None, **kwargs):
        super().__init__()
        cfg = dict(DEFAULT_CONFIG)
        if config is not None:
            cfg.update(config)
        cfg.update(kwargs)
        self.config = cfg
        
        self.multiscale = MultiScaleCNN(
            in_channels=cfg['input_channels'],
            out_channels=192,
            kernel_sizes=cfg['kernel_sizes']
        )
        
        self.attention = ECABlock(
            channels=192,
            gamma=cfg['attention_gamma'],
            b=cfg['attention_b']
        )
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(192, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(256, cfg['num_classes'])
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.multiscale(x)
        x = self.attention(x)
        x = self.gap(x).squeeze(-1)
        
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x

    @torch.inference_mode()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)
