from gym.envs.registration import register
import os

MAXTURNS = 10
DATASET = os.path.join(os.path.dirname(__file__), 'data', 'sqli-1k.csv')

register(
    id='WafBrain-v0',
    entry_point='gym_waf.envs:WafBrainEnv',
    kwargs={'payloads_file': DATASET, 'maxturns': MAXTURNS, 'score_threshold': 0.2}
)

register(
    id='WafLibinjection-v0',
    entry_point='gym_waf.envs:LibinjectionEnv',
    kwargs={'payloads_file': DATASET, 'maxturns': MAXTURNS}
)