"""
Opponent feature extraction for UNO game.
Tracks opponent actions, hand sizes, and play patterns to extract features for opponent modeling.
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple


class OpponentFeatureExtractor:
    """
    Extract features about opponent behavior for opponent modeling.
    Tracks action history, hand sizes, and play patterns.
    """
    
    def __init__(self, history_size: int = 20, num_actions: int = 61):
        """
        Initialize opponent feature extractor.
        
        Args:
            history_size: Number of recent actions to track
            num_actions: Total number of actions in action space
        """
        self.history_size = history_size
        self.num_actions = num_actions
        
        # Track opponent information
        # opponent_id -> history
        self.opponent_actions = {}  # Dict[opponent_id, deque]
        self.opponent_hand_sizes = {}  # Dict[opponent_id, deque]
        self.opponent_action_counts = {}  # Dict[opponent_id, Dict[action_type, count]]
        self.game_length = 0
        
        # Action type categories (UNO-specific)
        self.action_categories = {
            'number_card': list(range(0, 40)),  # 0-39: number cards
            'skip': list(range(40, 44)),  # 40-43: skip cards
            'reverse': list(range(44, 48)),  # 44-47: reverse cards
            'draw_2': list(range(48, 52)),  # 48-51: draw 2 cards
            'wild': list(range(52, 56)),  # 52-55: wild cards
            'wild_draw_4': list(range(56, 60)),  # 56-59: wild draw 4 cards
            'draw': [60]  # 60: draw card action
        }
        
    def reset(self, num_players: int = 2):
        """
        Reset feature extractor for a new game.
        
        Args:
            num_players: Number of players in the game
        """
        self.opponent_actions = {}
        self.opponent_hand_sizes = {}
        self.opponent_action_counts = {}
        self.game_length = 0
        
        # Initialize tracking for each opponent
        for player_id in range(num_players):
            if player_id != 0:  # Skip player 0 (our agent)
                self.opponent_actions[player_id] = deque(maxlen=self.history_size)
                self.opponent_hand_sizes[player_id] = deque(maxlen=self.history_size)
                self.opponent_action_counts[player_id] = {
                    'number_card': 0,
                    'skip': 0,
                    'reverse': 0,
                    'draw_2': 0,
                    'wild': 0,
                    'wild_draw_4': 0,
                    'draw': 0,
                    'total': 0
                }
    
    def update_history(self, action: int, opponent_id: int, hand_size: Optional[int] = None):
        """
        Update opponent history with new action and hand size.
        
        Args:
            action: Action taken by opponent
            opponent_id: ID of the opponent player
            hand_size: Current hand size of opponent (if available)
        """
        if opponent_id == 0:  # Skip our agent
            return
        
        if opponent_id not in self.opponent_actions:
            self.opponent_actions[opponent_id] = deque(maxlen=self.history_size)
            self.opponent_hand_sizes[opponent_id] = deque(maxlen=self.history_size)
            self.opponent_action_counts[opponent_id] = {
                'number_card': 0,
                'skip': 0,
                'reverse': 0,
                'draw_2': 0,
                'wild': 0,
                'wild_draw_4': 0,
                'draw': 0,
                'total': 0
            }
        
        # Record action
        self.opponent_actions[opponent_id].append(action)
        
        # Record hand size if available
        if hand_size is not None:
            self.opponent_hand_sizes[opponent_id].append(hand_size)
        
        # Update action counts by category
        action_category = self._categorize_action(action)
        if action_category:
            self.opponent_action_counts[opponent_id][action_category] += 1
            self.opponent_action_counts[opponent_id]['total'] += 1
        
        self.game_length += 1
    
    def _categorize_action(self, action: int) -> Optional[str]:
        """
        Categorize action into card type.
        
        Args:
            action: Action ID
            
        Returns:
            Action category string or None
        """
        for category, action_range in self.action_categories.items():
            if action in action_range:
                return category
        return None
    
    def extract_features(self, state: Dict, opponent_id: int) -> np.ndarray:
        """
        Extract opponent features from current state and history.
        
        Args:
            state: Current game state (RLCard format)
            opponent_id: ID of the opponent player
            
        Returns:
            Feature vector as numpy array
        """
        if opponent_id == 0:
            # Return zero features for our agent
            return np.zeros(self._get_feature_size())
        
        features = []
        
        # 1. Current hand size (if available)
        if 'raw_obs' in state and 'num_cards' in state['raw_obs']:
            num_cards = state['raw_obs']['num_cards']
            # num_cards can be a list or dict
            if isinstance(num_cards, (list, np.ndarray)):
                if opponent_id < len(num_cards):
                    hand_size = num_cards[opponent_id]
                else:
                    hand_size = 7  # Default
            elif isinstance(num_cards, dict):
                hand_size = num_cards.get(opponent_id, 7)
            else:
                hand_size = 7  # Default
            features.append(float(hand_size))
        else:
            # Use last known hand size
            if opponent_id in self.opponent_hand_sizes and len(self.opponent_hand_sizes[opponent_id]) > 0:
                hand_size = float(self.opponent_hand_sizes[opponent_id][-1])
            else:
                hand_size = 7.0  # Default starting hand size
            features.append(hand_size)
        
        # 2. Hand size change rate
        hand_size_change = self._compute_hand_size_change(opponent_id)
        features.append(hand_size_change)
        
        # 3. Recent action history (one-hot encoded)
        recent_actions = self._get_recent_actions(opponent_id, n=10)
        action_history_features = np.zeros(self.num_actions)
        for action in recent_actions:
            if 0 <= action < self.num_actions:
                action_history_features[action] += 1
        # Normalize by number of actions
        if len(recent_actions) > 0:
            action_history_features = action_history_features / len(recent_actions)
        features.extend(action_history_features.tolist())
        
        # 4. Action type frequencies
        action_freqs = self._get_action_frequencies(opponent_id)
        features.extend(action_freqs)
        
        # 5. Play style indicators
        play_style = self._infer_play_style(opponent_id)
        features.extend(play_style)
        
        # 6. Game progress features
        game_progress = self._get_game_progress_features()
        features.extend(game_progress)
        
        return np.array(features, dtype=np.float32)
    
    def extract_sequence_features(self, opponent_id: int) -> np.ndarray:
        """
        Extract sequence of recent actions and hand sizes.
        
        Args:
            opponent_id: ID of the opponent
            
        Returns:
            Sequence features as numpy array [seq_len, sequence_feature_size]
        """
        if opponent_id == 0:
            # Return zero sequence for our agent
            return np.zeros((self.history_size, self.get_sequence_feature_size()), dtype=np.float32)
            
        # Get recent actions and hand sizes
        actions = list(self.opponent_actions.get(opponent_id, []))
        hand_sizes = list(self.opponent_hand_sizes.get(opponent_id, []))
        
        # Pad or truncate to history_size
        seq_len = self.history_size
        
        # Create sequence features
        # Each step: [one_hot_action (num_actions), hand_size (1)]
        sequence_features = np.zeros((seq_len, self.get_sequence_feature_size()), dtype=np.float32)
        
        # Fill from the end (most recent)
        current_len = len(actions)
        for i in range(min(current_len, seq_len)):
            # Index in history (0 is oldest stored, -1 is most recent)
            # We want to fill the sequence such that the last element is the most recent
            hist_idx = current_len - 1 - i
            seq_idx = seq_len - 1 - i
            
            if hist_idx >= 0:
                action = actions[hist_idx]
                hand_size = hand_sizes[hist_idx] if hist_idx < len(hand_sizes) else 7.0
                
                # One-hot action
                if 0 <= action < self.num_actions:
                    sequence_features[seq_idx, action] = 1.0
                
                # Hand size
                sequence_features[seq_idx, self.num_actions] = float(hand_size)
                
        return sequence_features

    def get_sequence_feature_size(self) -> int:
        """
        Get the size of the sequence feature vector (per step).
        
        Returns:
            Sequence feature vector size
        """
        # One-hot action + hand size
        return self.num_actions + 1
    
    def _compute_hand_size_change(self, opponent_id: int) -> float:
        """
        Compute hand size change rate (average change per turn).
        
        Args:
            opponent_id: ID of the opponent
            
        Returns:
            Average hand size change rate
        """
        if opponent_id not in self.opponent_hand_sizes:
            return 0.0
        
        hand_sizes = list(self.opponent_hand_sizes[opponent_id])
        if len(hand_sizes) < 2:
            return 0.0
        
        changes = []
        for i in range(1, len(hand_sizes)):
            change = hand_sizes[i] - hand_sizes[i-1]
            changes.append(change)
        
        if len(changes) == 0:
            return 0.0
        
        return np.mean(changes)
    
    def _get_recent_actions(self, opponent_id: int, n: int = 10) -> List[int]:
        """
        Get recent N actions from opponent.
        
        Args:
            opponent_id: ID of the opponent
            n: Number of recent actions to return
            
        Returns:
            List of recent action IDs
        """
        if opponent_id not in self.opponent_actions:
            return []
        
        recent_actions = list(self.opponent_actions[opponent_id])
        return recent_actions[-n:] if len(recent_actions) > n else recent_actions
    
    def _get_action_frequencies(self, opponent_id: int) -> List[float]:
        """
        Get action type frequencies for opponent.
        
        Args:
            opponent_id: ID of the opponent
            
        Returns:
            List of action type frequencies (normalized)
        """
        if opponent_id not in self.opponent_action_counts:
            return [0.0] * len(self.action_categories)
        
        counts = self.opponent_action_counts[opponent_id]
        total = counts['total']
        
        if total == 0:
            return [0.0] * len(self.action_categories)
        
        frequencies = [
            counts['number_card'] / total,
            counts['skip'] / total,
            counts['reverse'] / total,
            counts['draw_2'] / total,
            counts['wild'] / total,
            counts['wild_draw_4'] / total,
            counts['draw'] / total
        ]
        
        return frequencies
    
    def _infer_play_style(self, opponent_id: int) -> List[float]:
        """
        Infer opponent playing style (aggressive vs conservative).
        
        Args:
            opponent_id: ID of the opponent
            
        Returns:
            List of play style indicators:
            - Aggressiveness score (0-1): tendency to play cards quickly
            - Conservativeness score (0-1): tendency to hold cards
            - Special card usage (0-1): frequency of special cards
        """
        if opponent_id not in self.opponent_action_counts:
            return [0.5, 0.5, 0.0]
        
        counts = self.opponent_action_counts[opponent_id]
        total = counts['total']
        
        if total == 0:
            return [0.5, 0.5, 0.0]
        
        # Aggressiveness: high ratio of number cards and low ratio of draws
        number_card_ratio = counts['number_card'] / total if total > 0 else 0
        draw_ratio = counts['draw'] / total if total > 0 else 0
        aggressiveness = number_card_ratio * (1 - draw_ratio)
        
        # Conservativeness: high ratio of draws and low card play rate
        # Approximated by inverse of aggressiveness
        conservativeness = 1 - aggressiveness
        
        # Special card usage: ratio of special cards (skip, reverse, draw_2, wild)
        special_card_count = (
            counts['skip'] + counts['reverse'] + 
            counts['draw_2'] + counts['wild'] + counts['wild_draw_4']
        )
        special_card_usage = special_card_count / total if total > 0 else 0
        
        return [aggressiveness, conservativeness, special_card_usage]
    
    def _get_game_progress_features(self) -> List[float]:
        """
        Get game progress features.
        
        Returns:
            List of game progress indicators:
            - Game length (normalized)
            - Turn number (normalized)
        """
        # Normalize game length (assuming max ~200 turns)
        max_game_length = 200
        normalized_length = min(self.game_length / max_game_length, 1.0)
        
        # Turn number (relative to game length)
        turn_progress = normalized_length
        
        return [normalized_length, turn_progress]
    
    def _get_feature_size(self) -> int:
        """
        Get the size of the feature vector.
        
        Returns:
            Feature vector size
        """
        # Hand size: 1
        # Hand size change: 1
        # Action history (one-hot): num_actions
        # Action frequencies: 7 (number of action categories)
        # Play style: 3
        # Game progress: 2
        return 1 + 1 + self.num_actions + 7 + 3 + 2
    
    def get_feature_size(self) -> int:
        """
        Public method to get feature vector size.
        
        Returns:
            Feature vector size
        """
        return self._get_feature_size()

