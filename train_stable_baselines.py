import gym_waf, gym
import utils
import os

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.common import make_vec_env, set_global_seeds
from stable_baselines import PPO2


def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    log_dir = "tmp_log/"
    os.makedirs(log_dir, exist_ok=True)

    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return Monitor(env, log_dir)
    set_global_seeds(seed)
    return _init


if __name__ == '__main__':
    env_id = "WafBrain-v0"
    num_cpu = 4  # Number of processes to use

    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    # Vectorized, but use DummyVec, which is not multiprocessing
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0)

    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=5000)

    utils.save_model('ppo2_wafbrain', model)
