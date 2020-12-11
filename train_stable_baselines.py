import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import gym_waf, gym
import utils
import os
import numpy as np
import argparse

import matplotlib.pyplot as plt
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.common import set_global_seeds
from stable_baselines import PPO2
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines import results_plotter
from stable_baselines.results_plotter import load_results, ts2xy


def make_env(env_id, rank, log_dir, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    os.makedirs(log_dir, exist_ok=True)

    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        log_file = os.path.join(log_dir, str(rank))
        return Monitor(env, log_file)
    set_global_seeds(seed)
    return _init


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                            self.best_mean_reward, mean_reward))

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)

        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO2 trainer for gym-waf.')
    parser.add_argument('env_id', metavar='ENV', type=str,
                        help='env id (WafBrain-diff-v0, WafLibinjection-v0')
    parser.add_argument('-n', dest='num_cpu', type=int, default=4,
                        help='number of parallel processes (default 4)')
    parser.add_argument('-t', dest='time_steps', type=float, default=1e4,
                        help='max training time steps (default 1e4)')
    parser.add_argument('-d', dest='debug', action='store_true',
                        help='turn on debug log')
    args = parser.parse_args()

    env_id = args.env_id
    num_cpu = args.num_cpu  # Number of processes to use
    time_steps = args.time_steps
    debug = args.debug

    if debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    log_dir = utils.prepare_logdir()

    env = SubprocVecEnv([make_env(env_id, i, log_dir) for i in range(num_cpu)])

    model = PPO2(MlpPolicy, env, verbose=1,
                 nminibatches=64, ent_coef=1e-4, learning_rate=1e-4, n_steps=512, cliprange=0.3)
    # Create the callback: check every 1000 steps
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, verbose=1)
    # Train the agent
    model.learn(total_timesteps=int(time_steps), callback=callback)

    results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "PPO2 {}".format(env_id))
    plt.savefig(os.path.join(log_dir, 'progress.png'))
