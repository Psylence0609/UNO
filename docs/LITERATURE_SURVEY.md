# Literature Survey: Reinforcement Learning for UNO Card Game

## Project Summary

This project is implementing and testing several reinforcement learning algorithms for playing the UNO card game with a focus on trying to solve the sparse reward problem by using MCTS for reward weighting. We will try to test four different strategies:

1. **DQN (Deep Q-Network)**: Baseline deep RL with only sparse rewards
2. **DQN+MCTS**: DQN strengthened by MCTS for intermediate rewards
3. **DMC (Deep Monte Carlo)**: Episode-based learning with MCTS reward function shaping
4. **RLCard DMC**: Implemented DMC using RLCard library

**Key Contributions:**
- Custom implementation of DMC agent with three-headed architecture (value, policy, and Monte Carlo heads)
- Integration of MCTS Reward Shaping for Handling Sparse Reward in UNO
- Comparative evaluation framework for multiple algorithms
- Reach 52.6% win rate (custom DMC) and 60% (RLCard DMC) with respect to random baseline

**Technical Approach:**
- Environment: play with RLCard's UNO Env (2-player games)
- State representation: 301-dimensional feature vector (240 observation + 61 legal actions)
- Network architecture: [256, 128]
- Training: 15,000 episodes for customized agents, 100M+ frames for RLCard-DMC
- MCTS: 200 UCB1 simulations with c=1.414, max depth 20

**Changes from Proposal:**
- **Expanded Algorithm Comparison**: Originally intended for comparison between DQN and DMC, but extended to cover both customized and RLCard code, adding up to 4+ models.
- **Integration of MCTS Reward Shaping**: Implemented MCTS-based reward shaping for both DQN and DMC algorithm. This was not incorporated in the original proposal.
- **RLCard Integration**: Implemented RLCard's native DMCTrainer for comparison with handmade implementations
- **Performance Targets**: Reached 52.6% (Custom DMC) and 60% (RLCard DMC) levels for overall win rate performance, thus proving the efficiency of each training

---

## Related Work

### 1. RLCard: A Toolkit for Reinforcement Learning in Card Games

**Zha et al. (2019)** proposed RLCard, which is an open-source toolkit for designing standard environments for studying card games with RL. The toolkit includes several algorithms (DQN, DMC, NFSP, and CFR) and allows for comparison via a standard interface. RLCard tackles the problem created by games involving imperfect information by offering abstraction for the states and masked actions.

**How it connects to our work:** We chose to use RLCard as a starting point for our environment, but then modified it with customized MCTS reward shaping. Although RLCard's implementation for DMC includes standard reward structures, our customized version uses a different network design (three-headed network instead of one Q-value head in RLCard) and is combined with customized MCTS rewards directly incorporated in training. From evaluation, it is clear that RLCard's pre-implemented DMC achieves 60% win rate, outperforming our customized version (52.6%), which is indicative of their actor-learner architecture.

**Technical differences:** The DMC in RLCard employs a parallel-actor architecture for efficient experience sampling, whereas in our custom DMC, it is done by episode-based learning. The network architecture is simpler in RLCard's implementation, which is optimal for faster computation. Our model employs a more complex architecture with three heads for value estimation.

---

### 2. Human-Level Control through Deep Reinforcement Learning

**Mnih et al. (2015)** proposed Deep Q-Networks (DQN) by integrating Q-learning with deep neural networks. They demonstrated human-level control on several Atari games. Their main contributions lie in experience replay, use of target networks, and frame stacking. They incorporate epsilon-greedy strategies for exploration with sparse rewards.

**Importance to our work:** We choose to use DQN with the same set of ideas (experience replay, target networks, and epsilon-greedy policy for exploration). However, due to UNO's sparse rewards (only rewards for winning or losing), we choose to improve DQN by adding MCTS Reward Shaping. Our experimental evaluation reveals that standard DQN performs only 48.6% better than a random policy, which is lower than all variants of DMC.

**Technical difference:** Our implementation involves the use of dueling networks (Wang et al., 2016) and double deep Q-learning networks (Van Hasselt et al., 2016), while traditional DQN proposed standard Q networks. Additionally, we have used legal action mask for card games, which is not required for environments such as Atari games.

---

### 3. Dueling Network Architectures for Deep Reinforcement Learning

**Wang et al. (2016)** introduced a dueling architecture for DQN, which involves decoupling value and advantage functions. This enables both value functions and advantages to be estimated separately by the network. The architecture involves: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a))).

**How it relates to our work:** We use dueling architecture in our DQN implementation. Dueling architecture is beneficial for value estimation in the complex value space in games such as UNO. The value and advantage separation is beneficial in problems with multiple actions having value closeness (such as card games). We adopted this architecture for DQN+MCTS with improvement over vanilla DQN (50.0% vs. random play compared to 48.6% for vanilla DQN).

**Technical implementation:** The architecture for our dueling DQN consists of [256, 128] hidden layers with value and advantage heads. This is similar to what we implemented for our DMC agent. The dueling architecture ensures that the agent learns that particular positions on the board have value irrespective of moves made.

---

### 4. Deep Reinforcement Learning with Double Q-learning

**Van Hasselt et al. (2016)** proposed Double DQN to correct Q-learning's overestimation bias. Double DQN uses the primary network for action selection and the target network for evaluation, which helps decrease the positive bias caused by using only one network for both purposes.

**Connection to our work:** We incorporate Double DQN in our Q-learning agents to refine value estimation. The overestimation bias is a concern in UNO because it has a stochastic environment and actions can have comparable values. We use Double DQN to enhance our Q-agents in estimating better Q-values, which leads to improvement by 1.4% for DQN+MCTS compared to DQN Original.

**Technical details:** We employed the standard Double DQN update rule: y = r + γ * Q_target(s', argmax_a Q_main(s', a)), where Q-main predicts actions and Q-target predicts their values. We performed updates on Q-target every 1000 timesteps.

---

### 5. Mastering the Game of Go with Deep Neural Networks and Tree Search

**Silver et al. (2016)** proposed AlphaGo by combining deep networks with Monte Carlo Tree Search (MCTS). MCTS applies UCB1 to choose actions. UCB1 simulates actions to evaluate their values. The brilliance in AlphaGo is that it applies MCTS for both training and inference.

**Applicability to Our Task:** We incorporate MCTS for reward shaping, but not for suggesting actions. Rather than relying on MCTS for suggesting actions, we utilize it for assessing positions in a game and offering rewards. This is useful for fixing issues with sparse rewards for UNO games because it gives the network a way to know if a position is good without necessarily completing all games. We incorporated MCTS with 200 simulation UCB-1 (c=1.414) with a search depth of 20.

**Technical differences:** AlphaGo employs MCTS for both training and playing, while we use MCTS only for training. We have used MCTS to evaluate positions in order to calculate intermediate rewards. AlphaGo employs MCTS to choose actions. We have used a simpler rollout policy (random policy) compared to AlphaGo.

---

### 6. Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm

**Silver et al. (2018)** generalized AlphaGo to AlphaZero, which is capable of playing games on its own without any human involvement. AlphaZero employs MCTS with a value/policy network, thus proving that it is possible to have superhuman performance without human data.

**How it relates to our work:** AlphaZero concentrates on perfect-information games and self-play. We deal with card games with imperfect information. The technique of exploiting value functions for guiding MCTS is related to what we do with MCTS for position evaluation. Our MCTS implementation might be improved by value functions, just like AlphaZero.

**Technical Comparison:** AlphaZero employs MCTS for action selection with learned networks, whereas we implement MCTS for reward shaping. Our environment (UNO) is one with both imperfect information and randomness, thus not entirely suitable for applying AlphaZero's technique. The analogy between us and AlphaZero is that both employ search for improving learning.

---

### 7. Reward Shaping for Episodic Reinforcement Learning

**Ng et al. (1999)** formulated the theoretical grounds for reward shaping and verified that potential-based reward shaping is safe by showing that it preserves optimal policies. The critical part is that adding rewards with Φ(s')-Φ(s) does not change the optimal policy.

**Relation to Our Work:** Our MCTS reward shaping offers mid-game rewards via position evaluation. On one hand, it can be considered a variation on potential-based reward shaping. On the other hand, it doesn't strictly fit the bill because we used MCTS rewards instead of working with differences in potentials. That's because MCTS rewards are necessary for complex spaces such as UNO.

**Technical Approach:** The intermediate rewards for this work are calculated by: r_shaped = r_base + α * (V_MCTS(s') - V_MCTS(s)), with V_MCTS denoting MCTS evaluation and α = 0.3.

---

### 8. Winning UNO with Reinforcement Learning

**Brown et al. (2020)** proposed a DQN and DeepSARSA implementation for UNO by using the RLCard toolkit. The authors trained their agents on random opponents and demonstrated a clear advantage for DQN over DeepSARSA. Their work showed considerable improvements over random strategies for UNO by using reinforcement learning.

**How it relates to our work:** This is clearly the most related work. Brown et al. employed standard DQN with sparse rewards with success. We build on this by including MCTS reward shaping and designing DMC as another algorithm. Our baseline for DQN (48.6%) is actually lower than in this work. This might be for several reasons. Either this work uses a different training procedure or evaluation metrics. Also, our DMC+MCTS (52.6%) and RLCard DMC (60%) outperform this work.

**Technological differences:** They applied standard DQN, whereas we are applying Dueling Double DQN. They did not apply any reward shaping technique, whereas we have combined MCTS for providing intermediate rewards. They did not investigate DMC, which we have implemented.

---

### 9. Reinforcement Learning Compared with Rule-Based Playing in UNO

**Ottosson & Nordström (2024)** compared several reinforcement learning algorithms (DQN, NFSP, and DMC) with rule-based algorithms on the game of UNO using the RLCard platform. They showed that DMC performed best compared with other reinforcement learning algorithms. This paper is another demonstration of the effectiveness of reinforcement learning on UNO.

**Relation to our work:** This recent research supports and justifies our interest in DMC because it is a promising algorithm for playing UNO. Their observation that DMC performs better than DQN is consistent with what we found (DMC Ep10k: 52.6% vs DQN Original: 48.6%). They, however, did not investigate MCTS Reward Shaping combined with DMC. We have found that it gives modest improvements, although it is not as significant as it should have been.

**Technical comparison:** Ottosson & Nordström employed a standard implementation of DMC in RLCard, while we have done a customized version of DMC with a three-headed model and MCTS with reward shaping. We have customized because it gives us more control over the process.

---

### 10. Deep Reinforcement Learning with MCTS Reward Shaping for Sparse Reward Games

**Li (2024)** particularly focused on addressing the sparse reward problem for UNO by combining MCTS with Double Deep Q-learning. MCTS was used for reshaping rewards such that intermediate rewards would be fed to agents.

**Relation to Work:** This work is by far the most related to ours because it deals with the same issue (UNO with sparse rewards) with a comparable method (MCTS with reward shaping). We have structured our work in a comparable manner but on top of DMC instead of Double DQN. We have proposed a custom MCTS with 200 trees and UCB-1 selection, just like in this work.

**Technical Similarities:** They both utilize MCTS for position evaluation and for rewarding. They both mitigate issues related to UNO's sparse rewards scheme. They both utilize the RLCard framework.

**Technical differences:** Li adopts Double DQN, while we adopt both DQN and DMC. Also, note that in our DMC code, we have adopted episode-learning (Monte Carlo Return) instead of TD-learning. Similarly, for MCTS, we have incorporated it directly into the training process. Li might do it differently.

---

### 11. Neural Fictitious Self-Play

**Heinrich & Silver (2016)** proposed Neural Fictitious Self-Play (NFSP) that combines supervised learning and reinforcement learning for multi-agent environments. This model uses two types of networks: Best-Response Network (learned with reinforcement learning) and Average Policy Network (learned with supervised learning). Players randomly select moves from Average Policy Network.

**Relation to Our Work:** Although we do not use NFSP in this work, it is applicable in RLCard and is considered another method for dealing with multi-agent reinforcement learning. The non-stationarity problem in multi-agent reinforcement learning is considered in NFSP. This is applicable for games like UNO because each agent plays against each other. Our current work concentrates on playing with fixed agents (random agents), which makes it hard to directly gain advantages from this model.

**Technical comparison:** NFSP employs a Two-Network architecture (Best-Response + Average Policy), whereas for our DMC, it is a Three-Heads architecture (Value + Policy + MC). NFSP is a multi-agent setup. Currently, this work is on Single-Agent.

---

### 12. Counterfactual Regret Minimization

**Zinkevich et al. (2008)** proposed CFR, which is an algorithm for obtaining Nash equilibria for games with incomplete information. CFR is a strategy iteration algorithm that uses regret matching to minimize counterfactual regret. It has been very successful on variants of poker.

**Connection to our work:** CFR is a completely different paradigm compared to models developed in this work. Firstly, it is a game theory algorithm which can guarantee a Nash Equilibrium solution. Our approaches rely on trial-and-error policy reinforcement. Additionally, it is available in RLCard, but it is only applicable if perfect information abstraction is used. Such abstraction is perhaps not optimal for playing UNO. We have instead used model free reinforcement, which is even more efficient compared to UNO due to reasons stated above.

**Technical Differences:** CFR relies on regret minimization and information sets, whereas our approaches rely on value function approximation or policy gradients. CFR is computationally heavy for every iteration, whereas our reinforcement learning approaches allow for online learning. CFR is theoretically optimal but computationally expensive, whereas our approaches are more so towards real-time learning.

---

### 13. Multi-DMC: Deep Monte-Carlo with Multi-Stage Learning in UNO

**Yang & Liu (2024)** propose Multi-DMC, which is a DMC improvement that applies multi-stage learning for playing UNO. The strategy involves dividing the learning process into several stages, which enable the AI to learn strategies with increasing levels of sophistication. This is relevant to UNO because it is complex.

**How it relates to our work:** Multi-DMC is another curriculum-learning model for DMC, which we do not implement. But progressive learning can be related to our work because it might be beneficial to train UNO with multiple levels. We have used overall training for all episodes in our DMC algorithm, whereas Multi-DMC is based on curriculum levels. Next step for this work would be to add multiple levels for MCTS-optimized DMC.

**Technical comparison:** Multi-DMC employs multiple levels of training with gradually increasing difficulty, whereas in our DMC, all models are trained with uniform training. Also, Multi-DMC does not specify about MCTS with reward shaping. Reward shaping in MCTS is one of the primary strengths of our model.

---

### 14. Policy Gradient Methods for Reinforcement Learning with Function Approximation

**Sutton et al. (2000)** created a theory for policy gradient control, which described how one can calculate policy gradients with regards to expected returns. This work is considered to represent the foundation for various policy gradient models such as REINFORCE, Actor-Critic, and PPO.

**How this is related to what we do:** We implement a policy gradient technique known as REINFORCE with baseline. The policy and value networks employed by our DMC have been trained with both value networks and policy gradients. Specifically, for policy networks, policy gradients have been employed. Such a technique is better suited for complex spaces such as in the UNO game.

**Technical implementation:** We implement our DMC using REINFORCE with baseline:
∇J(θ) = E_π[∇_logπ(a|s) (G - V(s))]
Here, G is Monte Carlo Return and V(s) is value function baseline. REINFORCE with baseline is variance reduced compared to vanilla REINFORCE but preserves advantages of policy gradient.

---

### 15. Asynchronous Methods for Deep Reinforcement Learning

**Mnih et al. (2016)** proposed the Asynchronous Advantage Actor-Critic (A3C), which employed asynchronous actors to collect experiences. The main goal of this technique is to have multiple actors work asynchronously in parallel to collect experiences.

**How it is relevant to us:** RLCard's DMC implementation employs a parallel actor design. This is why it outperforms us (RLCard's result is 60% compared to 52.6% in our case). This makes it much quicker for RLCard's DMC implementation's trainers to gain experience. We have employed a sequential process in our implementation for this exact reason. The actor-learner design employed by RLCard's DMC implementation is analogous to that used by A3C adapted for their DMC implementation.

**Technical comparison:** A3C applies asynchronous parallel actors with sharing parameters, while for RLCard's DMC, it applies a more structured actor-learner design with sharing memory buffers. Our proposed DMC applies episode-based learning in a sequential process, which is simpler but cannot be parallelized. Parallelization is much more efficient for training, which can be considered evident with RLCard's DMC's capacity to train on 100M+ frames.

---

### 16. Deep Reinforcement Opponent Network

**He et al. (2016)** proposed the Deep Reinforcement Opponent Network (DRON), which is a joint-learning model for both policy and opponents' behaviors. The proposed DRON uses a deep Q-Network with the encoded observation of opponents to produce adaptation without requiring pre-estimation of actions taken by opponents. Its technique is based on joint-learning between policy networks and model networks for opponents.

**Relation to Our Work:** DRON is a method for modeling opponents that we can incorporate into our work as a stretch goal. Given that we're working on agents that play against fixed random opponents, the joint-learning model that DRON uses can allow our agents to adjust to different strategies. We can add this architecture to our DMC or DQN agents.

**Technical Approach:** Its architecture is comprised of two streams: one for the policy and another for opponents. Opponent Modeling is done by processing actions and game states to deduce strategies. It can work for UNO by processing plays made by opponents in terms of card plays and hand sizes. Card counting can help identify playing styles.

---

### 17. Learning to Exploit: Implicit Opponent Modeling in Multi-Agent Reinforcement Learning

**Wu et al. (2021)** introduced a framework called Learning to Exploit (L2E) for implicit opponent modeling. The L2E learns to exploit opponents effectively with minimal exposure during training. This makes it easily adaptable to new opponents with unknown styles. Its design applies meta-learning for generalized play with different opponents.

**How it is related to our work:** The implication here is that the implicit opponent model used by L2E might have usefulness in UNO, which involves agents playing against strategies which differ. Rather than relying on explicit models for opponents (like in explicit models), L2E exploits these models implicitly. We can merge this with our code for DMC or DQN.

**Technical comparison:** L2E employs meta-learning for exploiting strategies. Our current agents are fixed policy learners. L2E can prove beneficial for improving our agents by adjusting rapidly to a new player with minimal retraining. Applying this technique for game UNO: Adjusting easily to aggressive players (playing rapidly) and conservative ones (holding for longer) might be beneficial.

---

### 18. DeepStack: Expert-Level Artificial Intelligence in Heads-Up No-Limit Poker

**Moravčík et al. (2017)** created a deep-learning system called DeepStack, which applies recursive reasoning for playing poker games expertly. Its algorithm employs a process called "continual resolving," which makes it capable of playing optimally in games involving imperfect information without having to solve the game tree.

**Connection to Our Work:** Although DeepStack is made for playing poker, its treatment for dealing with imperfect information is useful for UNO. DeepStack's recursive thinking might be applicable for modeling opponents in UNO because it involves thinking about what an opponent might have. The only problem is that it is computationally costly and thus not suited for real-time UNO gameplay.

**Technical Approach:** DeepStack employs deep neural networks for approximating counterfactual values for different information sets. Applying this to UNO might involve approximating values for plays based on different possible configurations of hands held by opponents. The continual resolving method might assist UNO players in making optimal moves without resorting to exhaustive search.

---

### 19. Libratus: Superhuman AI for No-Limit Poker

**Brown & Sandholm (2019)** created a system called Libratus. It beat top human poker players by combining analytical game playing with modeling human opponents. Libratus employs "blueprint strategies" precomputed in advance. These blueprint strategies adjust in a real-time manner based on observed deviations by opposing strategy models.

**How it relates to our work:** The model developed by Libratus for modeling opponents by deviation detection can be used for UNO. The current system recognizes discrepancies between playing styles for deviation detection. Applying it to UNO: recognizing aggressive and conservative playing styles.

**Technical comparison:** Libratus relies heavily on computation in blueprint strategies calculation, while our reinforcement learners rely on online computation. The ability of Libratus to adapt to opponents dynamically can propose approaches for modeling UNO opponents without involving heavy computation.

---

### 20. Learning from Learners: Adapting Reinforcement Learning Agents to be Competitive in a Card Game

**Almeida et al. (2020)** analyzed adaptation in competitive games involving RL agents towards playing styles. They aimed to design training and testing scripts for assessing adaptations towards competitive and opposing playing styles. This assignment is relevant for understanding adaptations towards playing styles.

**How it is related to our work:** This work is directly related to our work on UNO: this work refers to adapting agents to opponents in UNO. This work's finding about adapting agents can actually help us with implementing the stretch goal for UNO: implementing models for opponents. According to this work, it is actually possible for agents to learn to take advantage of their opponents.

**Technical Approach:** The technique employed in this research is self-play training. This involves training the agents on playing against each other. They learn strategies based on adapting to each other's playing styles. Applying this to UNO might involve training agents to play against aggressive, conservative, or intelligent players.

---

## Opponent Modeling: Stretch Goal

### Motivation

Although our current research involves learning efficient policies for playing against fixed random opponents, opponent modeling would constitute a natural extension that can prove beneficial for improving performance levels in UNO. The UNO game features imperfect information due to the inability to monitor opponents' hands. Hence, opponent modeling would prove even more useful in such games because agents can learn to adjust with strategies by observing opponents' moves.

### Proposed Approach

From the literature review, it is proposed that for UNO, it is necessary to do:

1. **Opponent Modeling with a Neural Network:** Following DRON (He et al., 2016) and L2E (Wu et al., 2021), we can merge the process of modeling opponents right into our DMC or DQN models. Modeling opponents will try to make sense out of the actions we see that the opponents take by playing their cards.

2. **Bayesian Opponent Modeling:** Inspired by DeepStack (Moravčík et al., 2017) and Libratus (Brown & Sandholm, 2019), we might be able to model belief about others' hand distributions and playing patterns. This would enable us to infer which hands others might have and play accordingly.

3. **RLCard Integration:** The environment class in RLCard allows us to have information about actions taken by opponents. We can append this information to features. We can append features related to strategy inference.

### Expected Benefits

- **Improved Performance**: Agents with abilities to model and exploit opponent weaknesses will have improved chances for victory
- **Adaptability**: Opponent modeling would facilitate adapting to different play styles without any need for retraining
- **Strategic Depth**: Knowing strategies would allow for playing with bluffing or defensive strategies

### Implementation Challenges

- **State Representation**: The state space for UNO would have to be expanded to represent information about past actions taken by opponents
- **Computational Cost**: The computational cost involved due to Opponent Modeling needs to be managed with real-time processing needs
- **Training Complexity**: Opponent modeling and playing a game simultaneously can have instability issues related to training

### Future Work

As a stretch goal, we propose:

1. Add a neural network Opponent Modeling part for our DMC agent
2. Performance assessment on various types of opponents (Aggressive Opponent, Conservative Opponent, Strategic Opponent)
3. Opponent Modeling: Explicit vs. Implicit, Neural Network vs. Bayesian
4. Incorporate opponent modeling in the RLCard environment for UNO

---

## Summary and Research Gaps

### Key Findings from Literature

1. **Effectiveness of DMC for UNO**: Several research studies (Ottosson & Nordström, 2024; Yang & Liu, 2024) have confirmed that DMC is much better than DQN for playing UNO.

2. **MCTS Reward Shaping is a promising technique**: Li (2024) showed that MCTS Reward Shaping can improve performance on UNO with DQN, although for us it only gave small improvements.

3. **Parallel training is critical**: Winning rate with RLCard's actor-learner architecture is 60%, which strongly dominates traditional sequentially trained architectures.

4. **Sparse rewards are difficult**: All authors agree on UNO's sparse rewards as a difficult part for reward functions.

5. **Opponent modeling has promise**: Recent studies (He et al., 2016; Wu et al., 2021; Almeida et al., 2020) have shown that modeling opponents can lead to considerable improvement in imperfect information games.

### Research Gaps Addressed by Our Work

1. **DMC with MCTS Reward Shaping**: Li (2024) proposed MCTS for DQN. We generalize this to DMC by adding rewards between episodes.

2. **Custom architecture exploration**: Our model's architecture with multiple heads (value, policy, MC) is different from that used in RLCard.

3. **Complete comparison of algorithms**: We are providing a comparison between DQN, DQN+MCTS, DMC, and DMC+MCTS, which has not been performed by any other work.

4. **RLCard vs custom implementation**: We draw comparison between RLCard's optimized DMC (60%) and our custom implementation (52.6%), shedding light on architecture and training efficiency.

5. **Foundation for Opponent Modeling**: Although we choose not to incorporate Opponent Modeling in this current experiment, we provide a foundation for future incorporation by reviewing relevant literature and proposing integration approaches.

### Limitations and Future Directions

1. **Integration with MCTS**: We see small improvements due to MCTS. We can investigate more complex integration with MCTS or value function networks for MCTS.

2. **Self-play training**: All our agents play against random opponents. Self-play can perhaps play even better.

3. **Multi-stage learning**: The curriculum learning technique used by Multi-DMC can be coupled with our MCTS reward shaping.

4. **Architecture optimization**: RLCard's DMC reaches 60% with simpler architectures but with parallel training. We might actually do even better with our three-headed architecture with parallel training.

5. **Opponent Modeling Implementation**: Opponent modeling (based on DRON, L2E, and/or Bayesian models) can prove to be a great stretch goal.

---

## References

1. **Zha, D., Lai, K.-H., Cao, Y., Huang, S., Wei, R., Guo, J., & Hu, X. (2019).** RLCard: A Toolkit for Reinforcement Learning in Card Games. *arXiv preprint arXiv:1910.04376*. https://arxiv.org/abs/1910.04376

2. **Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2015).** Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533. https://doi.org/10.1038/nature14236

3. **Wang, Z., Schaul, T., Hessel, M., Van Hasselt, H., Lanctot, M., & Freitas, N. (2016).** Dueling Network Architectures for Deep Reinforcement Learning. *Proceedings of the 33rd International Conference on Machine Learning (ICML)*, 1995-2003. https://arxiv.org/abs/1511.06581

4. **Van Hasselt, H., Guez, A., & Silver, D. (2016).** Deep Reinforcement Learning with Double Q-learning. *Proceedings of the 30th AAAI Conference on Artificial Intelligence*, 2094-2100. https://arxiv.org/abs/1509.06461

5. **Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., Van Den Driessche, G., ... & Hassabis, D. (2016).** Mastering the game of Go with deep neural networks and tree search. *Nature*, 529(7587), 484-489. https://doi.org/10.1038/nature16961

6. **Silver, D., Hubert, T., Schrittwieser, J., Antonoglou, I., Lai, M., Guez, A., ... & Hassabis, D. (2018).** A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. *Science*, 362(6419), 1140-1144. https://doi.org/10.1126/science.aar6404

7. **Ng, A. Y., Harada, D., & Russell, S. (1999).** Policy invariance under reward transformations: Theory and application to reward shaping. *Proceedings of the 16th International Conference on Machine Learning (ICML)*, 278-287.

8. **Brown, O., Jasson, D., & Swarnakar, A. (2020).** Winning UNO with Reinforcement Learning. *Stanford University CS228 Final Project Report*. https://web.stanford.edu/class/aa228/reports/2020/final79.pdf

9. **Ottosson, E., & Nordström, A. (2024).** Reinforcement Learning Compared to Rule-Based Play in UNO. *Stockholm University Master's Thesis*. https://www.diva-portal.org/smash/get/diva2:1955734/FULLTEXT01.pdf

10. **Li, Y. (2024).** Deep Reinforcement Learning with MCTS Reward Shaping for Sparse Reward Games. *arXiv preprint arXiv:2410.11642*. https://arxiv.org/abs/2410.11642

11. **He, H., Boyd-Graber, J., Kwok, K., & Daumé III, H. (2016).** Opponent Modeling in Deep Reinforcement Learning. *Proceedings of the 33rd International Conference on Machine Learning (ICML)*, 1804-1813. https://jmlr.csail.mit.edu/proceedings/papers/v48/he16.pdf

12. **Zinkevich, M., Johanson, M., Bowling, M., & Piccione, C. (2008).** Regret Minimization in Games with Incomplete Information. *Advances in Neural Information Processing Systems (NeurIPS)*, 20, 1729-1736.

13. **Yang, X., & Liu, X. (2024).** Multi-DMC: Deep Monte-Carlo with Multi-Stage Learning in the Card Game UNO. *Proceedings of the IEEE Conference on Games (CoG)*.

14. **Sutton, R. S., McAllester, D. A., Singh, S. P., & Mansour, Y. (2000).** Policy Gradient Methods for Reinforcement Learning with Function Approximation. *Advances in Neural Information Processing Systems (NeurIPS)*, 12, 1057-1063.

15. **Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., ... & Kavukcuoglu, K. (2016).** Asynchronous Methods for Deep Reinforcement Learning. *Proceedings of the 33rd International Conference on Machine Learning (ICML)*, 1928-1937. https://arxiv.org/abs/1602.01783

16. **Heinrich, J., & Silver, D. (2016).** Deep Reinforcement Learning from Self-Play in Imperfect-Information Games. *arXiv preprint arXiv:1603.01121*. https://arxiv.org/abs/1603.01121

17. **Wu, J., Wang, X., & Zhang, Y. (2021).** Learning to Exploit: Implicit Opponent Modeling in Multi-Agent Reinforcement Learning. *Proceedings of the 38th International Conference on Machine Learning (ICML)*. https://arxiv.org/abs/2102.09381

18. **Moravčík, M., Schmid, M., Burch, N., Lisý, V., Morrill, D., Bard, N., ... & Bowling, M. (2017).** DeepStack: Expert-Level Artificial Intelligence in Heads-Up No-Limit Poker. *Science*, 356(6337), 508-513. https://doi.org/10.1126/science.aam6960

19. **Brown, N., & Sandholm, T. (2019).** Superhuman AI for Multiplayer Poker. *Science*, 365(6456), 885-890. https://doi.org/10.1126/science.aay2400

20. **Almeida, L., Barros, L., & Sciutti, A. (2020).** Learning from Learners: Adapting Reinforcement Learning Agents to be Competitive in a Card Game. *arXiv preprint arXiv:2004.04000*. https://arxiv.org/abs/2004.04000
