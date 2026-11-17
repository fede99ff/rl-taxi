# Tabular and Deep Q-Learning on Taxi-v3

Five reinforcement-learning agents solving Gymnasium's [Taxi-v3](https://gymnasium.farama.org/environments/toy_text/taxi/) environment, from tabular methods to deep Q-networks, implemented from scratch in Python and PyTorch.

| Agent | Type | Test success rate (50,000 episodes) | Avg. reward per episode |
|---|---|---|---|
| Q-learning | tabular, off-policy | _TBD_ | _TBD_ |
| SARSA | tabular, on-policy | _TBD_ | _TBD_ |
| DQN | deep, single network | **100%** | 7.90 |
| DQN + target network | deep, frozen target network | **100%** | 7.92 |
| Double DQN | deep, decoupled action selection and evaluation | **100%** | 7.94 |

_Project for the Artificial Intelligence and Machine Learning course, MSc in Engineering in Computer Science, Sapienza University of Rome._

## The problem

A taxi moves on a 5×5 grid and must pick up a passenger and drop them at the right location. The environment has 500 discrete states and 6 actions (4 moves, pick-up, drop-off). Each step costs −1, a successful drop-off gives +20 and an illegal pick-up or drop-off costs −10. Episodes are truncated after 200 steps; an episode counts as a success when the passenger is delivered.

## Agents

**Q-learning (off-policy).** Q-table updated with the Bellman optimality target `r + γ · max_a' Q(s', a')`.

**SARSA (on-policy).** Same table, but the target uses the action actually chosen by the ε-greedy policy: `r + γ · Q(s', a')`.

**DQN.** The Q-table is replaced by an MLP (one-hot state → 256 → 256 → 6 Q-values) trained on mini-batches from an experience-replay buffer. The same network predicts Q(s, a) and computes the target `r + γ · max_a Q(s', a)`.

**DQN + target network.** As DQN, but the target is computed by a frozen copy of the network, `r + γ · max_a Q_target(s', a)`, which is synced with the online network every 7 episodes. Keeping the target fixed between syncs stabilises training.

**Double DQN.** The max operator in the DQN target tends to pick actions whose value is overestimated by chance. Double DQN (van Hasselt et al., 2016) decouples the two roles: the online network selects the best next action and the target network evaluates it, `r + γ · Q_target(s', argmax_a Q(s', a))`. Everything else is identical to the target-network agent.

All agents bootstrap from the next state only when the episode is not terminated: transitions are stored with the `terminated` flag, not `terminated or truncated`, so time-limit truncations still bootstrap correctly.

## Results

All three deep agents solve the environment: their greedy policies deliver the passenger in 100% of 50,000 test episodes. Their average reward per episode (7.90–7.94) is close to the optimum for Taxi-v3, so the learned routes are close to the shortest ones; Double DQN obtains the highest total reward, followed by the target-network DQN.

_TBD: tabular agents' results and training curves (average reward, episode length and success rate, 100-episode moving average)._

Results come from a single run with a fixed seed; with three agents at 100%, differences in average reward of a few hundredths are within run-to-run variation.

## Possible extensions

- **Per-step updates.** Updating every few environment steps, with a budget fixed in environment steps rather than episodes, would make the comparison between agents independent of episode length.
- **Multiple seeds.** Averaging over several seeds would give confidence intervals on the success rates and convergence times. On a small, deterministic environment like Taxi-v3, the advantage of Double DQN over the target-network DQN is expected to be small, and only a multi-seed comparison could measure it reliably.

## Project structure

```
src/
├── main.py              # trains and tests all agents, plots the metrics
├── tabular_agents.py    # Q-learning and SARSA agents + training/testing loop
├── DQN_agents.py        # DQN, target-network DQN and Double DQN agents + training/testing loop
├── DQN_network.py       # MLP used as Q-function approximator
├── replay_buffer.py     # experience replay buffer
├── metrics.py           # moving averages, success rate, convergence time, plots
├── q_table.npy, SARSA_q_table.npy                  # trained tabular agents
└── single_dqn_model.pth, target_dqn_model.pth, ddqn_model.pth   # trained deep agents
```

## How to run

```bash
pip install -r requirements.txt
cd src
python main.py
```

`main.py` trains every agent, saves the trained models in `src/`, tests them and plots the metrics. Full training takes about 30 minutes on a laptop CPU. To only evaluate the provided models, run the testing calls in `main.py` with `is_training=False`, which is the default.
Notice that the update for the DQN part has been performed at the end of the episode as it is way less time demanding as the training was performed on a CPU.
