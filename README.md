# FRAC-MARL

This repository provides an implementation of **Fully Resilient Actor-Critic Multi-Agent Reinforcement Learning (FRAC-MARL)**, a Byzantine-resilient multi-agent actor-critic reinforcement learning algorithm. We evaluate FRAC-MARL on the cooperative formation control task from [MPE2](https://mpe2.farama.org/environments/simple_formation/), in which a team of decentralized agents learns to achieve an evenly spaced formation around a shared landmark.

During training, agents exchange critic and team-reward weights with their neighbors over a communication graph that may contain corrupted (Byzantine) edges. The repository includes one undefended baseline and three Byzantine-resilient defense mechanisms, enabling direct comparison of their performance under simulated Byzantine attacks.

## Contents

- [Defenses](#defenses)
- [Task](#task)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)

## Defenses

Every agent trains an actor, a critic, and a team-reward network via the same local
gradient steps; they differ only in how they exchange and aggregate neighbors' weights
each consensus round (`build_agents`/`consensus_update` in `train.py`).

| `mode` | Agent class | Defense |
|---|---|---|
| `normal` | `Agent` | No defense. Fixed-weight consensus average over every accepted peer (own weight = 1 − #peers/n). Run with `F_BUDGET=0` as the attack-free baseline. |
| `naive` | `Agent` | The same undefended agent, but run with `F_BUDGET=F_BUDGET_G` so it's exposed to attacks with nothing in place - shows what an undefended team looks like under attack. |
| `resilient` | `Redundancy_Agent` | Two-round message relay + a `tau`-threshold majority vote (`redundancy_filter`) across the graph's redundant paths, then the same fixed-weight averaging over whatever passed the filter. |
| `projection` | `Projection_Agent` | PyTorch port of Algorithm 2 from Ye et al. (2024, see [References](#references)): projects each candidate through the local network to recover its implied error, trims the outliers, replaces the hidden layers with the trimmed mean, then fits the network to the trimmed team error by gradient descent. |
| `trimmed_mean` | `Trim_Mean_Agent` | Byrd-TD(λ) baseline from Wu et al. (2021): accepts every message unconditionally, and instead trims each parameter coordinate-wise (drops the F largest/smallest peer values per coordinate) before averaging. |

A shared `simulate_attack()` on the base `Agent` optionally corrupts a message on the
wire with a directional bias, up to a per-round attack budget - this is the Byzantine
threat model every defense above is evaluated against.

## Task

Agents act in `mpe2`'s [`simple_formation_v1` environment](https://mpe2.farama.org/environments/simple_formation/) (wrapped by
`environment/mpe_formation_env.py`): N agents choose one of 5 discrete actions each step
(no movement, or move in one of 4 directions) to arrange themselves evenly around a shared
landmark. Reward is each agent's own (negated) distance to its currently-assigned slot
on the formation circle; an episode succeeds once the whole team is within the target
radius of its formation. 

## Repository layout

```
frac-marl/
├── train.py                      # main training script -- trains all 5 modes
├── test_and_visualize.py         # load a checkpoint, render one rollout to GIF/SVG
├── plot_reward.py                # plot averaged reward curves across seeds
├── construct_redundant_graphs.py # build the r-redundant communication graphs
├── requirements.txt
├── agents/
│   ├── agents.py                 # Agent            -- base actor-critic agent
│   ├── redundancy_agents.py      # Redundancy_Agent -- message-redundancy defense
│   ├── projection_agents.py      # Projection_Agent -- gradient-recovered trimmed-mean defense
│   └── trim_mean_agent.py        # Trim_Mean_Agent  -- coordinate-wise trimmed-mean baseline
└── environment/
    └── mpe_formation_env.py      # thin wrapper around mpe2's simple_formation_v1
```

## Installation

```bash
git clone https://github.com/joonlee16/frac-marl.git
cd frac-marl
pip install -r requirements.txt
```

## Usage

### Train

```bash
python train.py
```

By default, `train.py`'s `__main__` trains all five modes back-to-back for seed 0. Each
`run()` call ends with a blocking `plt.show()` of the raw reward curve - close the
window to move on to the next mode. Checkpoints are saved to
`weights/F=<F_BUDGET_G>/{mode}_final{seed}.pt` (and `_best{seed}.pt` for the
best-reward checkpoint); the module-level `F_BUDGET_G` in `train.py` sets both the
communication graph's redundancy and this checkpoint directory name.

To train a single mode and also save its reward curve (needed by `plot_reward.py`):

```python
from train import run
run(mode="projection", F_BUDGET=1, seed=0, save_rewards=True)
```

### Visualize a trained policy

```bash
python test_and_visualize.py weights/F=1/resilient_best0.pt --out rollout.gif --episode-seed 0 --fps 15
```

Runs one on-policy rollout with the checkpoint's actors and renders it as an animated
GIF of the team converging on its formation.

### Plot reward curves

```bash
python plot_reward.py --results-dir rewards/F=1 --out reward_summary.svg
```

Averages every `{mode}{seed}.npy` file found in `--results-dir` across seeds and plots
one smoothed curve per mode with a shaded uncertainty band (`--shade std|sem|minmax`).

## References

1. Zhang, K., Yang, Z., Liu, H., Zhang, T., & Başar, T. (2018). [Fully Decentralized
   Multi-Agent Reinforcement Learning with Networked Agents](https://proceedings.mlr.press/v80/zhang18n.html).
   ICML 2018. - the base decentralized actor-critic algorithm `Agent` implements.
2. Ye, L., Figura, M., Lin, Y., Pal, M., Das, P., Liu, J., & Gupta, V. (2024). [Resilient
   Multi-Agent Reinforcement Learning With Function Approximation](https://doi.org/10.1109/TAC.2024.3409676).
   IEEE Transactions on Automatic Control. - `Projection_Agent` is a PyTorch port of this
   paper's Algorithm 2, adapted from the authors'
   [reference implementation](https://github.com/mainakpal08/Resilient-consensus-based-MARL/tree/v1).
3. Wu, Z., Shen, H., Chen, T., & Ling, Q. (2021). [Byzantine-Resilient Decentralized
   Policy Evaluation With Linear Function Approximation](https://ieeexplore.ieee.org/document/9462519).
   IEEE Transactions on Signal Processing, 69, 3839–3853. - `Trim_Mean_Agent` reproduces
   this paper's Byrd-TD(λ) aggregation rule with λ=0.
