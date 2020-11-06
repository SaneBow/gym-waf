from gym.envs.registration import register

MAXTURNS = 10

register(
    id='waf-brain-v0',
    entry_point='gym_waf.envs:WafBrainEnv',
    kwargs={'payload': '1 or 1=1 -- a', 'score_threshold': 0.2, 'maxturns': 20}
)