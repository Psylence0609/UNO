"""
Opponent modeling network for learning opponent strategy representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class OpponentModelingNetwork(nn.Module):
    """
    Neural network for modeling opponent strategies.
    Processes opponent features and outputs a strategy representation.
    """
    
    def __init__(
        self,
        feature_size: int,
        hidden_layers: List[int] = [128, 64],
        strategy_dim: int = 32,
        activation: str = 'relu',
        dropout: float = 0.1
    ):
        """
        Initialize opponent modeling network.
        
        Args:
            feature_size: Size of input opponent features
            hidden_layers: List of hidden layer sizes
            strategy_dim: Dimension of output strategy representation
            activation: Activation function ('relu', 'tanh', 'leaky_relu')
            dropout: Dropout probability
        """
        super(OpponentModelingNetwork, self).__init__()
        
        self.feature_size = feature_size
        self.strategy_dim = strategy_dim
        
        # Choose activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network layers
        layers = []
        prev_size = feature_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        self.feature_layers = nn.ModuleList(layers)
        
        # Output layer: strategy representation
        self.strategy_head = nn.Linear(prev_size, strategy_dim)
        
    def forward(self, opponent_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through opponent modeling network.
        
        Args:
            opponent_features: Opponent features tensor [batch_size, feature_size]
            
        Returns:
            Strategy representation tensor [batch_size, strategy_dim]
        """
        x = opponent_features
        
        # Pass through feature layers
        for i in range(0, len(self.feature_layers), 2):
            x = self.feature_layers[i](x)  # Linear layer
            x = self.activation(x)
            if i + 1 < len(self.feature_layers):
                x = self.feature_layers[i + 1](x)  # Dropout layer
        
        # Output strategy representation
        strategy = self.strategy_head(x)
        
        # Normalize strategy representation (optional, can help with training)
        strategy = F.normalize(strategy, p=2, dim=1)
        
        return strategy
    
    def predict_strategy(self, opponent_features: torch.Tensor) -> torch.Tensor:
        """
        Predict opponent strategy from features.
        Alias for forward method.
        
        Args:
            opponent_features: Opponent features tensor
            
        Returns:
            Strategy representation tensor
        """
        return self.forward(opponent_features)


class OpponentStrategyClassifier(nn.Module):
    """
    Optional: Classify opponent into strategy types (aggressive, conservative, strategic).
    Can be used for interpretability and analysis.
    """
    
    def __init__(
        self,
        strategy_dim: int,
        num_strategies: int = 3,
        hidden_layers: List[int] = [64, 32]
    ):
        """
        Initialize strategy classifier.
        
        Args:
            strategy_dim: Dimension of strategy representation
            num_strategies: Number of strategy classes
            hidden_layers: List of hidden layer sizes
        """
        super(OpponentStrategyClassifier, self).__init__()
        
        self.strategy_dim = strategy_dim
        self.num_strategies = num_strategies
        
        # Build classifier layers
        layers = []
        prev_size = strategy_dim
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        self.classifier_layers = nn.ModuleList(layers)
        
        # Output layer: strategy class probabilities
        self.output_layer = nn.Linear(prev_size, num_strategies)
        
    def forward(self, strategy_representation: torch.Tensor) -> torch.Tensor:
        """
        Classify strategy representation into strategy types.
        
        Args:
            strategy_representation: Strategy representation tensor [batch_size, strategy_dim]
            
        Returns:
            Strategy class logits [batch_size, num_strategies]
        """
        x = strategy_representation
        
        # Pass through classifier layers
        for layer in self.classifier_layers:
            x = layer(x)
        
        # Output strategy class logits
        logits = self.output_layer(x)
        
        return logits
    
    def predict_strategy_class(self, strategy_representation: torch.Tensor) -> torch.Tensor:
        """
        Predict strategy class from strategy representation.
        
        Args:
            strategy_representation: Strategy representation tensor
            
        Returns:
            Strategy class probabilities [batch_size, num_strategies]
        """
        logits = self.forward(strategy_representation)
        probabilities = F.softmax(logits, dim=1)
        return probabilities

