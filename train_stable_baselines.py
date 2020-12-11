from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
import gym_waf, gym
import os, time

# Create unique log dir
log_dir = "/tmp/gym/{}".format(int(time.time()))
os.makedirs(log_dir, exist_ok=True)

env = gym.make('WafBrain-v0')
env = Monitor(env, log_dir, allow_early_resets=True)
env = DummyVecEnv([lambda: env])
model = PPO2('MlpPolicy', env, verbose=1).learn(10000)
