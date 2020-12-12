from gym.envs.registration import register
import os

MAXTURNS = 10
DATASET = os.path.join(os.path.dirname(__file__), 'data', 'sqli-1k.csv')
DATASET_single = os.path.join(os.path.dirname(__file__), 'data', 'sqli-1.csv')

register(
    id='WafBrain-v0',
    entry_point='gym_waf.envs:WafBrainEnv',
    kwargs={'payloads_file': DATASET, 'maxturns': MAXTURNS, 'score_threshold': 0.1}
)

register(
    id='WafBrain-diff-v0',
    entry_point='gym_waf.envs:WafBrainEnv',
    kwargs={'payloads_file': DATASET, 'maxturns': MAXTURNS, 'score_threshold': 0.1, 'use_diff_reward': True}
)

register(
    id='WafLibinj-v0',
    entry_point='gym_waf.envs:LibinjectionEnv',
    kwargs={'payloads_file': DATASET, 'maxturns': MAXTURNS}
)

register(
    id='WafBrain-single-v0',
    entry_point='gym_waf.envs:WafBrainEnv',
    kwargs={'payloads_file': DATASET_single, 'maxturns': MAXTURNS, 'score_threshold': 0.1}
)

register(
    id='WafLibinj-single-v0',
    entry_point='gym_waf.envs:LibinjectionEnv',
    kwargs={'payloads_file': DATASET_single, 'maxturns': MAXTURNS}
)