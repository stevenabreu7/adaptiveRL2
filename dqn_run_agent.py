import random
import numpy as np
import os
import pickle
import warnings
from dqn_agent import run_experiment, ExpCfg, eval_agent

warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")


cfg = ExpCfg(
    env_name = 'CartPole-v1',
    n_episodes = 5,
    n_steps = 100,
    architecture = [128, 128],
    replay_size = 1000,
    batch_size = 32,
    learning_rate = 1e-3,
    target_update_frequency = 10,
    gamma = 0.9,
    eps_schedule = (1.0, 0.05, 0.999)
)

trial_name = 'trial-{}-{}e-{}s-{}lr-{}g-{}fr-{}bs-{}rs'.format(
    'x'.join(map(str, cfg.architecture)), cfg.n_episodes, cfg.n_steps, cfg.learning_rate, 
    cfg.gamma, cfg.target_update_frequency, cfg.batch_size, cfg.replay_size
)
train_dir = f'dqn_results/{trial_name}'
os.makedirs(train_dir, exist_ok=True)

# if __name__ == "__main__":
seed = 0
random.seed(seed)
print("Trial #{} with {} episodes (max {} steps)".format(seed, cfg.n_episodes, cfg.n_steps))
### agent = DQNAgent(cfg, seed=seed)

# returns reward and mean loss per episode
params, reward, loss = run_experiment(cfg, train_dir=f'dqn_results/{trial_name}', seed=seed)
### trainstate, reward, loss = agent.train()
loss_nonan = loss[np.isnan(loss) == False]
print('reward min/max/mean:', reward.min(), reward.max(), reward.mean())
print('loss min/max/mean:', loss_nonan.min(), loss_nonan.max(), loss_nonan.mean())
np.savetxt(f'{train_dir}/reward.csv', reward, delimiter=',')
np.savetxt(f'{train_dir}/loss.csv', loss, delimiter=',')
### np.save(f'{train_dir}/params.npy', agent.trainstate.params)

# eval RL agent
reward = eval_agent(cfg, params, epsilon=0.05)
### reward = agent.eval(trainstate, n_episodes=200, epsilon=0.05)
print('eval reward min/max/mean:', reward.min(), reward.max(), reward.mean())
np.savetxt(f'{train_dir}/reward-eval.csv', reward, delimiter=',')
