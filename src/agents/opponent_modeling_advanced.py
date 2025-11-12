"""
Advanced Opponent Modeling Network with Attention Mechanisms.
Uses transformer-like attention and gated fusion for sophisticated opponent modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for opponent feature processing.
    Allows the model to focus on different aspects of opponent behavior.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize multi-head attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through multi-head attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism for combining state and opponent features.
    Uses learnable gates to control the contribution of each feature type.
    """
    
    def __init__(self, state_dim: int, opponent_dim: int, hidden_dim: int):
        """
        Initialize gated fusion.
        
        Args:
            state_dim: Dimension of state features
            opponent_dim: Dimension of opponent features
            hidden_dim: Dimension of hidden/fused representation
        """
        super(GatedFusion, self).__init__()
        
        # Projections
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.opponent_proj = nn.Linear(opponent_dim, hidden_dim)
        
        # Gates
        self.state_gate = nn.Sequential(
            nn.Linear(state_dim + opponent_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.opponent_gate = nn.Sequential(
            nn.Linear(state_dim + opponent_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Fusion
        self.fusion = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, state_features: torch.Tensor, opponent_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse state and opponent features using gated mechanism.
        
        Args:
            state_features: State features [batch_size, state_dim]
            opponent_features: Opponent features [batch_size, opponent_dim]
            
        Returns:
            Fused features [batch_size, hidden_dim]
        """
        # Project features
        state_proj = self.state_proj(state_features)
        opponent_proj = self.opponent_proj(opponent_features)
        
        # Compute gates
        combined = torch.cat([state_features, opponent_features], dim=1)
        state_gate = self.state_gate(combined)
        opponent_gate = self.opponent_gate(combined)
        
        # Gated fusion
        fused = state_gate * state_proj + opponent_gate * opponent_proj
        fused = self.fusion(fused)
        
        return fused


class AdvancedOpponentModelingNetwork(nn.Module):
    """
    Advanced opponent modeling network with attention and temporal modeling.
    Uses transformer-like architecture to process opponent action sequences.
    """
    
    def __init__(
        self,
        feature_size: int,
        hidden_layers: List[int] = [256, 128, 64],
        strategy_dim: int = 64,
        num_attention_heads: int = 4,
        use_attention: bool = True,
        use_temporal: bool = True,
        dropout: float = 0.1
    ):
        """
        Initialize advanced opponent modeling network.
        
        Args:
            feature_size: Size of input opponent features
            hidden_layers: List of hidden layer sizes
            strategy_dim: Dimension of output strategy representation
            num_attention_heads: Number of attention heads
            use_attention: Whether to use attention mechanism
            use_temporal: Whether to use temporal modeling
            dropout: Dropout probability
        """
        super(AdvancedOpponentModelingNetwork, self).__init__()
        
        self.feature_size = feature_size
        self.strategy_dim = strategy_dim
        self.use_attention = use_attention
        self.use_temporal = use_temporal
        
        # Input projection
        self.input_proj = nn.Linear(feature_size, hidden_layers[0])
        self.layer_norm = nn.LayerNorm(hidden_layers[0])
        
        # Attention mechanism (if enabled)
        if use_attention and use_temporal:
            # For temporal sequences, we need to reshape features
            # Assume we can create sequences from recent history
            self.attention = MultiHeadAttention(
                embed_dim=hidden_layers[0],
                num_heads=num_attention_heads,
                dropout=dropout
            )
            self.attention_norm = nn.LayerNorm(hidden_layers[0])
        
        # Feature processing layers
        layers = []
        prev_size = hidden_layers[0]
        
        for hidden_size in hidden_layers[1:]:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        self.feature_layers = nn.ModuleList(layers)
        
        # Strategy representation head
        self.strategy_head = nn.Sequential(
            nn.Linear(prev_size, strategy_dim),
            nn.LayerNorm(strategy_dim),
            nn.Tanh()  # Tanh for bounded representation
        )
        
        # Auxiliary prediction heads (for multi-task learning)
        self.hand_size_predictor = nn.Linear(prev_size, 1)
        self.action_predictor = nn.Linear(prev_size, 61)  # UNO has 61 actions
        
    def forward(
        self,
        opponent_features: torch.Tensor,
        sequence_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through opponent modeling network.
        
        Args:
            opponent_features: Opponent features [batch_size, feature_size]
            sequence_features: Optional sequence features [batch_size, seq_len, feature_size]
            
        Returns:
            Tuple of (strategy_representation, hand_size_pred, action_pred)
        """
        batch_size = opponent_features.shape[0]
        
        # Input projection
        x = self.input_proj(opponent_features)
        x = self.layer_norm(x)
        
        # Temporal attention (if enabled and sequence provided)
        if self.use_attention and self.use_temporal and sequence_features is not None:
            # Process sequence with attention
            seq_proj = self.input_proj(sequence_features)  # [batch, seq_len, hidden]
            seq_proj = self.attention_norm(seq_proj)
            seq_attn = self.attention(seq_proj)
            
            # Aggregate sequence (mean pooling)
            seq_aggregated = seq_attn.mean(dim=1)  # [batch, hidden]
            
            # Combine with current features
            x = x + seq_aggregated  # Residual connection
            x = self.layer_norm(x)
        
        # Feature processing
        for layer in self.feature_layers:
            x = layer(x)
        
        # Strategy representation
        strategy = self.strategy_head(x)
        
        # Auxiliary predictions (for multi-task learning)
        hand_size_pred = self.hand_size_predictor(x) if self.training else None
        action_pred = self.action_predictor(x) if self.training else None
        
        return strategy, hand_size_pred, action_pred


class AdvancedDMCNetworkWithOpponent(nn.Module):
    """
    Advanced DMC Network with sophisticated opponent modeling integration.
    Uses gated fusion and residual connections for better feature combination.
    """
    
    def __init__(
        self,
        state_size: int,
        opponent_strategy_dim: int,
        action_size: int,
        hidden_layers: List[int] = [512, 256, 128],
        activation: str = 'relu',
        dropout: float = 0.1,
        use_gated_fusion: bool = True
    ):
        """
        Initialize advanced DMC network with opponent modeling.
        
        Args:
            state_size: Dimension of input state features
            opponent_strategy_dim: Dimension of opponent strategy representation
            action_size: Dimension of action space
            hidden_layers: List of hidden layer sizes
            activation: Activation function
            dropout: Dropout probability
            use_gated_fusion: Whether to use gated fusion
        """
        super(AdvancedDMCNetworkWithOpponent, self).__init__()
        
        self.state_size = state_size
        self.opponent_strategy_dim = opponent_strategy_dim
        self.action_size = action_size
        self.use_gated_fusion = use_gated_fusion
        
        # Choose activation
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        else:
            self.activation = F.relu
        
        # State and opponent feature processing
        self.state_proj = nn.Sequential(
            nn.Linear(state_size, hidden_layers[0]),
            nn.LayerNorm(hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.opponent_proj = nn.Sequential(
            nn.Linear(opponent_strategy_dim, hidden_layers[0]),
            nn.LayerNorm(hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fusion mechanism
        if use_gated_fusion:
            self.fusion = GatedFusion(
                state_dim=hidden_layers[0],
                opponent_dim=hidden_layers[0],
                hidden_dim=hidden_layers[0]
            )
        else:
            # Simple concatenation + projection
            self.fusion = nn.Sequential(
                nn.Linear(hidden_layers[0] * 2, hidden_layers[0]),
                nn.LayerNorm(hidden_layers[0]),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        # Main network layers with residual connections
        self.main_layers = nn.ModuleList()
        for i in range(len(hidden_layers) - 1):
            layer = nn.Sequential(
                nn.Linear(hidden_layers[i], hidden_layers[i + 1]),
                nn.LayerNorm(hidden_layers[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.main_layers.append(layer)
        
        final_dim = hidden_layers[-1]
        
        # Output heads with separate processing
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU(),
            nn.Linear(final_dim // 2, 1)
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU(),
            nn.Linear(final_dim // 2, action_size)
        )
        
        # Monte Carlo head
        self.monte_carlo_head = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU(),
            nn.Linear(final_dim // 2, action_size)
        )
    
    def forward(
        self,
        state_features: torch.Tensor,
        opponent_strategy: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            state_features: State features [batch_size, state_size]
            opponent_strategy: Opponent strategy [batch_size, opponent_strategy_dim]
            
        Returns:
            Tuple of (value, policy_logits, mc_values)
        """
        # Project state and opponent features
        state_proj = self.state_proj(state_features)
        opponent_proj = self.opponent_proj(opponent_strategy)
        
        # Fusion
        if self.use_gated_fusion:
            x = self.fusion(state_proj, opponent_proj)
        else:
            combined = torch.cat([state_proj, opponent_proj], dim=1)
            x = self.fusion(combined)
        
        # Main network with residual connections
        for layer in self.main_layers:
            residual = x
            x = layer(x)
            # Residual connection if dimensions match
            if x.shape == residual.shape:
                x = x + residual
        
        # Output heads
        value = self.value_head(x)
        policy_logits = self.policy_head(x)
        mc_values = self.monte_carlo_head(x)
        
        return value, policy_logits, mc_values

