"""
Model implementation for OpenLRM integration with TripoSR.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf


class TokenizerModel(nn.Module):
    """
    Tokenizer model for TripoSR with OpenLRM integration.
    This replaces the DINOSingleImageTokenizer and Triplane1DTokenizer.
    """
    
    def __init__(self, cfg):
        """
        Initialize the tokenizer model.
        
        Args:
            cfg (OmegaConf): Configuration object.
        """
        super().__init__()
        self.cfg = cfg
        
        # Tokenizer configuration
        token_dim = cfg.model.tokenizer.dim
        hidden_dim = cfg.model.tokenizer.hidden_dim
        num_heads = cfg.model.tokenizer.heads
        num_tokens = cfg.model.tokenizer.num_tokens
        
        # Backbone configuration
        backbone_dim = cfg.model.backbone.dim
        backbone_depth = cfg.model.backbone.depth
        backbone_heads = cfg.model.backbone.heads
        mlp_ratio = cfg.model.backbone.mlp_ratio
        dropout = cfg.model.backbone.dropout
        attention_dropout = cfg.model.backbone.attention_dropout
        
        # Image encoder (using a simple CNN for demonstration)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Projection from CNN features to token dimension
        self.projection = nn.Linear(512, token_dim)
        
        # Token embeddings
        self.token_embedding = nn.Parameter(torch.randn(num_tokens, token_dim))
        self.pos_embedding = nn.Parameter(torch.randn(num_tokens, token_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=backbone_depth
        )
        
        # Output projection
        self.output_projection = nn.Linear(token_dim, token_dim)
        
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (dict): Input dictionary containing 'rgb' tensor.
            
        Returns:
            dict: Output dictionary containing tokens and features.
        """
        # Get RGB image from input
        rgb = x['rgb']  # [B, 3, H, W]
        batch_size = rgb.shape[0]
        
        # Extract image features
        img_features = self.image_encoder(rgb)  # [B, 512, 1, 1]
        img_features = img_features.view(batch_size, 512)  # [B, 512]
        
        # Project to token dimension
        img_tokens = self.projection(img_features)  # [B, token_dim]
        
        # Expand image tokens to match token embedding size
        img_tokens = img_tokens.unsqueeze(1).expand(-1, self.token_embedding.shape[0], -1)  # [B, num_tokens, token_dim]
        
        # Add token embeddings and positional embeddings
        tokens = img_tokens + self.token_embedding + self.pos_embedding  # [B, num_tokens, token_dim]
        
        # Apply transformer
        transformed_tokens = self.transformer(tokens)  # [B, num_tokens, token_dim]
        
        # Apply output projection
        output_tokens = self.output_projection(transformed_tokens)  # [B, num_tokens, token_dim]
        
        # Create output dictionary
        output = {
            'tokens': output_tokens,
            'features': transformed_tokens
        }
        
        return output


def build_model(cfg):
    """
    Build the model for TripoSR with OpenLRM integration.
    
    Args:
        cfg (OmegaConf): Configuration object.
        
    Returns:
        nn.Module: Model instance.
    """
    model = TokenizerModel(cfg)
    return model
