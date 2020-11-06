from gym.envs.registration import register

MAXTURNS = 10

register(
    id='waf-brain-v0',
    entry_point='gym_waf.envs:WafBrainEnv',
    kwargs={}
)