# Opponent Modeling for UNO: Research and Implementation Plan

## Overview

Opponent modeling is a stretch goal for this project that aims to enable UNO agents to predict and adapt to opponent strategies, improving performance against diverse opponents. This document provides a comprehensive research summary, recommended approaches, and implementation plan for integrating opponent modeling into our UNO reinforcement learning framework.

---

## Motivation

UNO is an imperfect information game where players cannot see opponents' hands. This creates opportunities for opponent modeling:

1. **Hidden Information**: Opponents' hands are unknown, but we can infer information from their actions
2. **Action Patterns**: Opponents' playing styles (aggressive, conservative, strategic) can be identified from action history
3. **Hand Size Tracking**: Opponent hand sizes are observable and provide clues about their strategy
4. **Adaptive Strategies**: Understanding opponent behavior allows agents to adapt and exploit weaknesses

**Expected Benefits:**
- Improved win rates against diverse opponent types
- Better adaptation to opponent strategies without retraining
- More sophisticated gameplay (bluffing, defensive play)
- Enhanced performance in multi-agent settings

---

## Literature Review

### 1. Explicit Opponent Modeling

#### Deep Reinforcement Opponent Network (DRON)
**He et al. (2016)** - "Opponent Modeling in Deep Reinforcement Learning"

**Approach:**
- Jointly learns policy and opponent behavior models
- Two-stream architecture: policy stream and opponent stream
- Opponent stream processes observed actions and game states
- Mixture-of-Experts architecture automatically discovers strategy patterns

**Key Features:**
- Encodes opponent observations into deep Q-Network
- No explicit action prediction required
- Automatically discovers different strategy patterns
- Evaluated on simulated soccer and trivia games

**Relevance to UNO:**
- Can process opponent card plays and hand sizes
- Can identify playing styles (aggressive, conservative)
- Joint learning allows policy and opponent modeling to inform each other
- Suitable for UNO's action-based observation structure

**Technical Implementation:**
- Two neural network streams: one for policy, one for opponent modeling
- Opponent stream processes: action history, hand size changes, card play patterns
- Policy stream uses opponent features to inform decision-making
- End-to-end training with joint loss function

---

### 2. Implicit Opponent Modeling

#### Learning to Exploit (L2E)
**Wu et al. (2021)** - "Learning to Exploit: Implicit Opponent Modeling in Multi-Agent Reinforcement Learning"

**Approach:**
- Implicit opponent modeling through meta-learning
- Learns to exploit opponents with minimal exposure
- Rapid adaptation to new opponents with unknown styles
- Uses meta-learning to generalize across opponent types

**Key Features:**
- No explicit opponent model maintained
- Learns exploitation strategies through policy network
- Fast adaptation to new opponents
- Generalizes across different opponent types

**Relevance to UNO:**
- Can adapt to aggressive players (play cards quickly) vs conservative players (hold cards)
- Meta-learning allows quick adaptation without retraining
- Policy network implicitly learns opponent patterns
- Suitable for UNO's diverse opponent strategies

**Technical Implementation:**
- Meta-learning framework for opponent exploitation
- Policy network learns to adapt to different opponent types
- Training with diverse opponent strategies
- Inference adapts quickly to new opponents

---

### 3. Bayesian Opponent Modeling

#### DeepStack and Libratus
**Moravčík et al. (2017)** - "DeepStack: Expert-Level Artificial Intelligence in Heads-Up No-Limit Poker"
**Brown & Sandholm (2019)** - "Libratus: The Superhuman AI for No-Limit Poker"

**Approach:**
- Maintains probabilistic beliefs about opponent hand distributions
- Bayesian inference for opponent strategy estimation
- Recursive reasoning about opponent actions
- Deviation detection for opponent adaptation

**Key Features:**
- Probabilistic beliefs over opponent strategies
- Bayesian updates based on observed actions
- Recursive reasoning for opponent inference
- Real-time adaptation to opponent deviations

**Relevance to UNO:**
- Can model belief distributions over opponent hands
- Infer opponent strategies from action patterns
- Detect when opponents deviate from expected strategies
- Reason about what cards opponents might have

**Technical Implementation:**
- Bayesian belief network for opponent hand distributions
- Posterior updates based on observed actions
- Strategy inference from action patterns
- Deviation detection for adaptive responses

**Challenges:**
- Computationally expensive for real-time gameplay
- Requires extensive computation for belief updates
- May be less practical for UNO's fast-paced gameplay

---

### 4. Neural Network-Based Opponent Modeling

#### Learning from Learners
**Almeida et al. (2020)** - "Learning from Learners: Adapting Reinforcement Learning Agents to be Competitive in a Card Game"

**Approach:**
- Self-play training with adaptive opponent strategies
- Agents learn to adapt to each other's playing styles
- Policy network learns to respond to different opponent patterns
- Training routines for competitive adaptation

**Key Features:**
- Self-play training with diverse opponents
- Adaptation to opponent behaviors
- Policy network learns opponent-responsive strategies
- Evaluated on competitive card games

**Relevance to UNO:**
- Can train agents to play against aggressive, conservative, or strategic opponents
- Self-play allows learning from diverse strategies
- Policy adaptation based on opponent patterns
- Suitable for UNO's competitive nature

**Technical Implementation:**
- Self-play training with diverse opponent strategies
- Policy network processes opponent action history
- Adaptation mechanisms in policy learning
- Evaluation against different opponent types

---

### 5. Variational Autoencoder Opponent Modeling

#### Variational Opponent Modeling
**Papoudakis & Albrecht (2020)** - "Variational Autoencoders for Opponent Modeling in Multi-Agent Systems"

**Approach:**
- Uses variational autoencoders (VAEs) to model opponents
- Learns latent representations of opponent behaviors
- Infers opponent behaviors from own observations
- Captures underlying behavioral patterns

**Key Features:**
- Latent representations of opponent behaviors
- Inference from own observations, actions, and rewards
- Captures behavioral patterns without explicit prediction
- Variational inference for uncertainty quantification

**Relevance to UNO:**
- Can learn latent representations of opponent strategies
- Infers opponent behaviors from observed actions
- Captures playing style patterns
- Provides uncertainty estimates for opponent predictions

**Technical Implementation:**
- VAE encoder for opponent behavior representation
- Latent space for opponent strategy patterns
- Decoder for opponent action prediction
- Variational inference for uncertainty

---

## Recommended Approach for UNO

Based on the literature review, we recommend a **hybrid approach** combining explicit neural network opponent modeling with action history features:

### Primary Approach: Explicit Neural Network Opponent Modeling (DRON-inspired)

**Why this approach:**
1. **RLCard Compatibility**: RLCard provides access to opponent actions and hand sizes, making explicit modeling feasible
2. **Computational Efficiency**: Neural network approach is more efficient than Bayesian methods for real-time gameplay
3. **End-to-End Learning**: Joint learning of policy and opponent models allows mutual improvement
4. **UNO-Specific Features**: Can leverage UNO-specific information (hand sizes, card plays, action timing)

**Architecture:**
- **Policy Stream**: Standard DMC/DQN policy network
- **Opponent Stream**: Neural network that processes:
  - Opponent action history (last N actions)
  - Opponent hand size changes over time
  - Card play patterns (aggressive vs conservative)
  - Action timing features
- **Fusion Layer**: Combines policy and opponent features for decision-making

**Features for Opponent Modeling:**
1. **Action History**: Last 10-20 opponent actions
2. **Hand Size Tracking**: Opponent hand size over time
3. **Card Play Patterns**: Frequency of number cards, special cards, wild cards
4. **Action Timing**: Time between actions (if available)
5. **Strategic Indicators**: Tendency to play aggressively or conservatively

### Secondary Approach: Implicit Opponent Modeling (L2E-inspired)

**Why as secondary:**
1. **Meta-Learning Complexity**: More complex to implement than explicit modeling
2. **Training Requirements**: Requires diverse opponent training set
3. **Adaptation Speed**: Fast adaptation is valuable but explicit modeling may be sufficient

**Use Case**: If explicit modeling doesn't achieve desired performance, L2E can be explored as an alternative.

---

## RLCard Integration

### RLCard State Structure

RLCard UNO environment provides the following information:

**State Dictionary:**
- `obs`: Observation tensor (4x4x15) containing game state
- `legal_actions`: OrderedDict of legal actions for current player
- `raw_obs`: Dictionary with:
  - `hand`: Current player's hand
  - `target`: Top card on discard pile
  - `num_cards`: Number of cards per player (includes opponent hand sizes)

**Available Opponent Information:**
- Opponent hand sizes (`num_cards[opponent_id]`)
- Opponent actions (observed when opponent plays)
- Game state changes (indicating opponent card plays)
- Action history (can be tracked manually)

### Integration Strategy

**Option 1: State Extension**
- Extend state representation to include opponent features
- Add opponent action history to state vector
- Include opponent hand size tracking
- Modify agent architecture to process extended state

**Option 2: Opponent Modeling Wrapper**
- Create wrapper around RLCard environment
- Track opponent actions and features
- Provide opponent features to agent
- Maintain opponent model separately from policy

**Option 3: Agent Modification**
- Modify agent architecture to include opponent modeling stream
- Process opponent features alongside state features
- Joint training of policy and opponent models
- End-to-end learning with opponent awareness

**Recommended: Option 3 (Agent Modification)**
- Most integrated approach
- Allows joint learning of policy and opponent models
- End-to-end training with opponent awareness
- Compatible with existing DMC/DQN architectures

---

## Implementation Plan

### Phase 1: Basic Opponent Feature Extraction

**Task**: Extract opponent features from RLCard state
- Track opponent hand sizes over time
- Record opponent action history
- Extract card play patterns
- Compute strategic indicators (aggressive/conservative)

**Implementation:**
```python
class OpponentFeatureExtractor:
    def __init__(self, history_size=20):
        self.history_size = history_size
        self.opponent_actions = []
        self.opponent_hand_sizes = []
        
    def extract_features(self, state, opponent_id):
        # Extract opponent hand size
        hand_size = state['raw_obs']['num_cards'][opponent_id]
        
        # Track hand size over time
        self.opponent_hand_sizes.append(hand_size)
        
        # Extract action patterns
        features = {
            'hand_size': hand_size,
            'hand_size_change': self._compute_hand_size_change(),
            'action_history': self._get_recent_actions(),
            'play_style': self._infer_play_style()
        }
        
        return features
```

### Phase 2: Opponent Modeling Network

**Task**: Create neural network for opponent modeling
- Design opponent modeling network architecture
- Process opponent features
- Learn opponent strategy representations
- Output opponent strategy features

**Implementation:**
```python
class OpponentModelingNetwork(nn.Module):
    def __init__(self, feature_size, hidden_layers=[128, 64]):
        super().__init__()
        # Process opponent features
        self.feature_layers = nn.ModuleList(...)
        # Output opponent strategy representation
        self.strategy_head = nn.Linear(hidden_layers[-1], strategy_dim)
        
    def forward(self, opponent_features):
        # Process opponent features
        x = opponent_features
        for layer in self.feature_layers:
            x = F.relu(layer(x))
        # Output strategy representation
        strategy = self.strategy_head(x)
        return strategy
```

### Phase 3: Integrated Agent Architecture

**Task**: Integrate opponent modeling into DMC/DQN agents
- Add opponent modeling stream to agent architecture
- Fuse policy and opponent features
- Joint training of policy and opponent models
- End-to-end learning

**Implementation:**
```python
class DMCWithOpponentModeling(DMCAgent):
    def __init__(self, state_size, action_size, config):
        super().__init__(state_size, action_size, config)
        # Add opponent modeling network
        self.opponent_model = OpponentModelingNetwork(...)
        # Modify policy network to accept opponent features
        self.policy_network = ModifiedDMCNetwork(
            state_size + opponent_feature_size, action_size
        )
        
    def act(self, state, opponent_features, legal_actions):
        # Process opponent features
        opponent_strategy = self.opponent_model(opponent_features)
        # Combine state and opponent features
        combined_features = torch.cat([state, opponent_strategy], dim=1)
        # Use combined features for policy
        return self.policy_network(combined_features, legal_actions)
```

### Phase 4: Training with Opponent Modeling

**Task**: Train agents with opponent modeling
- Collect opponent features during training
- Train opponent modeling network
- Joint training of policy and opponent models
- Evaluate against diverse opponents

**Implementation:**
- Modify training loop to track opponent features
- Add opponent modeling loss to training objective
- Train against diverse opponents (aggressive, conservative, strategic)
- Evaluate adaptation to different opponent types

### Phase 5: Evaluation and Refinement

**Task**: Evaluate opponent modeling performance
- Test against diverse opponent types
- Measure adaptation speed
- Compare with baseline (no opponent modeling)
- Refine architecture and features

---

## Expected Challenges

### 1. State Representation

**Challenge**: UNO's state space needs to be extended to include opponent features
- **Solution**: Add opponent features to state vector or process separately
- **Implementation**: Opponent feature extractor + state extension

### 2. Computational Cost

**Challenge**: Opponent modeling adds computational overhead
- **Solution**: Use efficient neural network architectures
- **Optimization**: Lightweight opponent modeling network, feature caching

### 3. Training Complexity

**Challenge**: Learning to model opponents while also learning to play
- **Solution**: Joint training with balanced losses
- **Approach**: Gradual introduction of opponent modeling, curriculum learning

### 4. Non-Stationary Opponents

**Challenge**: Opponents may change strategies during gameplay
- **Solution**: Adaptive opponent modeling with recent history weighting
- **Approach**: Use sliding window for opponent features, adaptive updates

### 5. RLCard Integration

**Challenge**: RLCard's environment may not provide all needed information
- **Solution**: Track opponent information manually, use available state features
- **Workaround**: Maintain opponent action history, infer from state changes

---

## Evaluation Metrics

### Performance Metrics

1. **Win Rate vs Diverse Opponents**:
   - Aggressive opponents (play cards quickly)
   - Conservative opponents (hold cards longer)
   - Strategic opponents (optimal play)
   - Random opponents (baseline)

2. **Adaptation Speed**:
   - Time to identify opponent strategy
   - Performance improvement over game duration
   - Adaptation to strategy changes

3. **Opponent Modeling Accuracy**:
   - Prediction accuracy for opponent actions
   - Strategy classification accuracy
   - Hand size estimation accuracy

### Comparison Metrics

1. **vs Baseline (No Opponent Modeling)**:
   - Win rate improvement
   - Performance against diverse opponents
   - Adaptation capability

2. **vs Explicit vs Implicit Modeling**:
   - Performance comparison
   - Computational efficiency
   - Adaptation speed

---

## Implementation Timeline

### Phase 1: Research and Design (Week 1)
- Complete literature review
- Design opponent modeling architecture
- Plan RLCard integration
- Define evaluation metrics

### Phase 2: Basic Implementation (Week 2)
- Implement opponent feature extraction
- Create opponent modeling network
- Integrate with DMC agent
- Basic training loop

### Phase 3: Training and Evaluation (Week 3)
- Train agents with opponent modeling
- Evaluate against diverse opponents
- Measure adaptation performance
- Compare with baseline

### Phase 4: Refinement (Week 4)
- Optimize architecture and features
- Improve training procedures
- Enhance adaptation mechanisms
- Final evaluation and documentation

---

## Code Structure

```
src/
├── agents/
│   ├── dmc_agent.py                 # Existing DMC agent
│   ├── dmc_agent_with_opponent.py   # DMC with opponent modeling
│   └── opponent_modeling.py         # Opponent modeling network
├── features/
│   └── opponent_features.py         # Opponent feature extraction
└── training/
    └── train_with_opponent.py       # Training with opponent modeling
```

---

## References

1. **He, H., Boyd-Graber, J., Kwok, K., & Daumé III, H. (2016)**. Opponent Modeling in Deep Reinforcement Learning. *Proceedings of the 33rd International Conference on Machine Learning (ICML)*, 1804-1813.

2. **Wu, J., Wang, X., & Zhang, Y. (2021)**. Learning to Exploit: Implicit Opponent Modeling in Multi-Agent Reinforcement Learning. *Proceedings of the 38th International Conference on Machine Learning (ICML)*.

3. **Moravčík, M., Schmid, M., Burch, N., Lisý, V., Morrill, D., Bard, N., ... & Bowling, M. (2017)**. DeepStack: Expert-Level Artificial Intelligence in Heads-Up No-Limit Poker. *Science*, 356(6337), 508-513.

4. **Brown, N., & Sandholm, T. (2019)**. Superhuman AI for Multiplayer Poker. *Science*, 365(6456), 885-890.

5. **Almeida, L., Barros, L., & Sciutti, A. (2020)**. Learning from Learners: Adapting Reinforcement Learning Agents to be Competitive in a Card Game. *arXiv preprint arXiv:2004.04000*.

6. **Papoudakis, G., & Albrecht, S. V. (2020)**. Variational Autoencoders for Opponent Modeling in Multi-Agent Systems. *arXiv preprint arXiv:2001.10829*.

7. **Shen, M., & How, J. P. (2019)**. Adversarial Ensemble Learning for Robust Opponent Modeling in Asymmetric Imperfect-Information Games. *arXiv preprint arXiv:1909.08735*.

8. **Zhao, Y., et al. (2022)**. Proximal Learning with Opponent-Learning Awareness. *arXiv preprint arXiv:2210.10125*.

9. **Jiang, J., et al. (2021)**. Metric Policy Representations for Opponent Modeling. *arXiv preprint arXiv:2106.05802*.

10. **Nashed, S., & Zilberstein, S. (2022)**. A Survey of Opponent Modeling in Adversarial Domains. *Journal of Artificial Intelligence Research*.

---

## Next Steps

1. **Implement Opponent Feature Extraction**: Create feature extractor for opponent actions and hand sizes
2. **Design Opponent Modeling Network**: Create neural network architecture for opponent modeling
3. **Integrate with DMC Agent**: Modify DMC agent to include opponent modeling stream
4. **Train and Evaluate**: Train agents with opponent modeling and evaluate performance
5. **Refine and Optimize**: Improve architecture and training based on results

---

*This document serves as a research foundation and implementation guide for adding opponent modeling as a stretch goal to the UNO reinforcement learning project.*

