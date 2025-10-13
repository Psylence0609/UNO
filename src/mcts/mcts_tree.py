"""
Monte Carlo Tree Search implementation for UNO reward shaping.
Based on the 2024 research paper for intermediate reward calculation.
"""

import math
import copy
import numpy as np
from collections import defaultdict
import random

class MCTSNode:
    """Node in the MCTS tree."""
    
    def __init__(self, state, parent=None, action=None, player_id=None):
        """
        Initialize MCTS node.
        
        Args:
            state: Game state
            parent: Parent node
            action: Action that led to this state
            player_id: Current player ID
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.player_id = player_id
        
        # MCTS statistics
        self.visits = 0
        self.wins = 0.0
        self.children = {}
        self.untried_actions = None
        
        # Game termination
        self.is_terminal = False
        self.terminal_value = 0.0
    
    def is_fully_expanded(self):
        """Check if all actions have been tried."""
        if self.untried_actions is None:
            return False
        return len(self.untried_actions) == 0
    
    def initialize_actions(self, legal_actions):
        """Initialize untried actions list."""
        if self.untried_actions is None:
            self.untried_actions = list(legal_actions)
    
    def has_untried_actions(self):
        """Check if there are untried actions."""
        return self.untried_actions is not None and len(self.untried_actions) > 0
    
    def best_child(self, exploration_weight=1.414):
        """
        Select best child using UCB1 formula.
        
        Args:
            exploration_weight: Exploration parameter (√2 is theoretical optimum)
        
        Returns:
            MCTSNode: Best child node
        """
        choices_weights = []
        for child in self.children.values():
            if child.visits == 0:
                weight = float('inf')
            else:
                # UCB1 formula: exploitation + exploration
                exploitation = child.wins / child.visits
                exploration = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
                weight = exploitation + exploration
            choices_weights.append(weight)
        
        return list(self.children.values())[np.argmax(choices_weights)]
    
    def add_child(self, action, state, player_id):
        """Add a child node for the given action."""
        child = MCTSNode(state, parent=self, action=action, player_id=player_id)
        self.children[action] = child
        if self.untried_actions is not None:
            self.untried_actions.remove(action)
        return child
    
    def update(self, result):
        """Update node statistics with simulation result."""
        self.visits += 1
        self.wins += result
    
    def get_win_rate(self):
        """Get win rate for this node."""
        return self.wins / self.visits if self.visits > 0 else 0.0


class MCTSRewardShaper:
    """
    Monte Carlo Tree Search for reward shaping in UNO.
    Provides intermediate rewards based on position evaluation.
    """
    
    def __init__(self, env, config=None):
        """
        Initialize MCTS reward shaper.
        
        Args:
            env: UNO environment
            config: MCTS configuration parameters
        """
        self.env = env
        self.config = config or {}
        
        # MCTS parameters
        self.num_simulations = self.config.get('num_simulations', 100)
        self.exploration_constant = self.config.get('exploration_constant', 1.414)
        self.max_depth = self.config.get('max_depth', 50)
        self.rollout_policy = self.config.get('rollout_policy', 'random')
        
        # Reward shaping parameters
        self.reward_scale = self.config.get('reward_scale', 0.1)
        self.discount_factor = self.config.get('discount_factor', 0.95)
        
    def get_shaped_reward(self, state, action, next_state, player_id, base_reward):
        """
        Calculate shaped reward using MCTS evaluation.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            player_id: Current player ID
            base_reward: Original sparse reward
            
        Returns:
            float: Shaped reward combining base reward and MCTS evaluation
        """
        # If game ended, return base reward
        if self.env.is_over():
            return base_reward
        
        # Run MCTS to evaluate position
        mcts_value = self._run_mcts(next_state, player_id)
        
        # Combine base reward with MCTS evaluation
        shaped_reward = base_reward + self.reward_scale * mcts_value
        
        return shaped_reward
    
    def _run_mcts(self, state, player_id):
        """
        Run MCTS simulation from given state.
        
        Args:
            state: Current state
            player_id: Current player ID
            
        Returns:
            float: MCTS evaluation value
        """
        # Create root node
        root = MCTSNode(state, player_id=player_id)
        
        # Initialize legal actions
        if isinstance(state, dict) and 'legal_actions' in state:
            legal_actions = list(state['legal_actions'].keys())
        else:
            legal_actions = list(range(self.env.num_actions))
        
        root.untried_actions = legal_actions.copy()
        
        # Run simulations
        for _ in range(self.num_simulations):
            # Selection and expansion
            node = self._select_and_expand(root)
            
            # Simulation
            result = self._simulate(node)
            
            # Backpropagation
            self._backpropagate(node, result, player_id)
        
        # Return value of root position
        return root.get_win_rate()
    
    def _select_and_expand(self, root):
        """
        Selection and expansion phase of MCTS.
        
        Args:
            root: Root node
            
        Returns:
            MCTSNode: Selected/expanded node
        """
        node = root
        depth = 0
        
        while not node.is_terminal and depth < self.max_depth:
            # Initialize actions if not done yet
            if node.untried_actions is None:
                # Get legal actions for this state
                temp_env = copy.deepcopy(self.env)
                # Restore state to the node's state
                temp_env.set_state(node.state, node.player_id)
                legal_actions = temp_env.get_legal_actions()
                node.initialize_actions(legal_actions)
            
            if node.has_untried_actions():
                # Expansion: add a new child
                action = random.choice(node.untried_actions)
                
                # Simulate taking the action
                try:
                    # Create a copy of the environment for simulation
                    temp_env = copy.deepcopy(self.env)
                    
                    # Take the action in the temporary environment
                    next_state, next_player_id = temp_env.step(action)
                    
                    # Create child node
                    child = node.add_child(action, next_state, next_player_id)
                    
                    # Check if terminal
                    if temp_env.is_over():
                        child.is_terminal = True
                        payoffs = temp_env.get_payoffs()
                        child.terminal_value = 1.0 if payoffs[node.player_id] > 0 else -1.0
                    
                    return child
                    
                except Exception:
                    # If action fails, mark as tried and continue
                    if action in node.untried_actions:
                        node.untried_actions.remove(action)
                    continue
            else:
                # Selection: choose best child
                node = node.best_child(self.exploration_constant)
                depth += 1
        
        return node
    
    def _simulate(self, node):
        """
        Simulation phase - random rollout from current position.
        
        Args:
            node: Starting node for simulation
            
        Returns:
            float: Simulation result from perspective of original player
        """
        if node.is_terminal:
            return node.terminal_value
        
        try:
            # Create temporary environment for rollout
            temp_env = copy.deepcopy(self.env)
            
            # Set environment to node's state (simplified)
            current_player = node.player_id
            rollout_depth = 0
            
            # Random rollout until game ends or max depth
            while not temp_env.is_over() and rollout_depth < self.max_depth:
                # Get legal actions
                state = temp_env.get_state(current_player)
                
                if isinstance(state, dict) and 'legal_actions' in state:
                    legal_actions = list(state['legal_actions'].keys())
                else:
                    legal_actions = list(range(self.env.num_actions))
                
                if not legal_actions:
                    break
                
                # Random policy
                action = random.choice(legal_actions)
                
                # Take action
                _, current_player = temp_env.step(action)
                rollout_depth += 1
            
            # Evaluate final position
            if temp_env.is_over():
                payoffs = temp_env.get_payoffs()
                return 1.0 if payoffs[node.player_id] > 0 else -1.0
            else:
                # If didn't finish, use heuristic evaluation
                return self._heuristic_evaluation(temp_env, node.player_id)
                
        except Exception:
            # If simulation fails, return neutral result
            return 0.0
    
    def _heuristic_evaluation(self, env, player_id):
        """
        Heuristic evaluation for unfinished games.
        
        Args:
            env: Current environment state
            player_id: Player to evaluate for
            
        Returns:
            float: Heuristic value (-1 to 1)
        """
        try:
            # Get perfect information for heuristic
            perfect_info = env.get_perfect_information()
            
            if 'hand_sizes' in perfect_info:
                hand_sizes = perfect_info['hand_sizes']
                
                if len(hand_sizes) > player_id:
                    my_cards = hand_sizes[player_id]
                    opponent_cards = [hand_sizes[i] for i in range(len(hand_sizes)) if i != player_id]
                    
                    if opponent_cards:
                        avg_opponent_cards = sum(opponent_cards) / len(opponent_cards)
                        
                        # Simple heuristic: fewer cards is better
                        if my_cards < avg_opponent_cards:
                            return 0.5  # Advantage
                        elif my_cards > avg_opponent_cards:
                            return -0.5  # Disadvantage
                        else:
                            return 0.0  # Neutral
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _backpropagate(self, node, result, original_player):
        """
        Backpropagation phase - update all nodes on path to root.
        
        Args:
            node: Starting node
            result: Simulation result
            original_player: Original player perspective
        """
        while node is not None:
            # Adjust result based on player perspective
            if node.player_id == original_player:
                adjusted_result = result
            else:
                adjusted_result = -result  # Opposite perspective
            
            node.update(adjusted_result)
            node = node.parent
    
    def get_action_values(self, state, player_id):
        """
        Get MCTS-based action values for all legal actions.
        
        Args:
            state: Current state
            player_id: Current player ID
            
        Returns:
            dict: Action -> value mapping
        """
        action_values = {}
        
        if isinstance(state, dict) and 'legal_actions' in state:
            legal_actions = list(state['legal_actions'].keys())
        else:
            legal_actions = list(range(self.env.num_actions))
        
        # Run MCTS for each action
        for action in legal_actions:
            try:
                # Simulate taking the action
                temp_env = copy.deepcopy(self.env)
                next_state, next_player_id = temp_env.step(action)
                
                # Evaluate resulting position
                value = self._run_mcts(next_state, player_id)
                action_values[action] = value
                
            except Exception:
                action_values[action] = 0.0
        
        return action_values


def calculate_mcts_reward(env, state, action, next_state, player_id, base_reward, mcts_shaper):
    """
    Calculate MCTS-shaped reward using simplified heuristic approach.
    
    Args:
        env: UNO environment
        state: Current state
        action: Action taken
        next_state: Next state
        player_id: Player ID
        base_reward: Base sparse reward
        mcts_shaper: MCTS reward shaper (can be None for simple heuristic)
        
    Returns:
        float: Shaped reward
    """
    # Use simplified heuristic reward shaping instead of full MCTS
    # This avoids the complexity of environment copying
    
    # Base reward weight
    shaped_reward = base_reward
    
    if base_reward == 0.0:  # Only add shaping for non-terminal states
        # Simple heuristic rewards based on game state
        heuristic_reward = 0.0
        
        # Reward for playing cards (making progress)
        heuristic_reward += 0.01
        
        # Extract hand size if possible
        try:
            if 'obs' in state and len(state['obs']) > 7:
                current_hand_size = state['obs'][7] if state['obs'][7] > 0 else 10  # Estimate
                if 'obs' in next_state and len(next_state['obs']) > 7:
                    next_hand_size = next_state['obs'][7] if next_state['obs'][7] > 0 else 10
                    
                    # Reward for reducing hand size
                    if next_hand_size < current_hand_size:
                        heuristic_reward += 0.1
                    
                    # Bonus for small hand size
                    if next_hand_size <= 3:
                        heuristic_reward += 0.05
                        
                    # Penalty for large hand size
                    if next_hand_size >= 10:
                        heuristic_reward -= 0.02
        except (IndexError, KeyError):
            pass
        
        shaped_reward = heuristic_reward
    
    return shaped_reward