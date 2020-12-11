import gym_waf
import utils

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2


if __name__ == '__main__':
    env_id = "WafBrain-v0"
    num_cpu = 4  # Number of processes to use

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you:
    env = make_vec_env(env_id, n_envs=num_cpu, seed=0)

    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=5000)

    utils.save_model('ppo2_wafbrain', model)
