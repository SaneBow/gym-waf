from stable_baselines import PPO2
import gym_waf

model = PPO2('MlpPolicy', 'WafBrain-v0', verbose=1).learn(10000)
