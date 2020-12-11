from .waf_score_env import WafScoreEnv
from .interfaces import WafBrainInterface


class WafBrainEnv(WafScoreEnv):
    def __init__(self, payloads_file, maxturns=20, score_threshold=0.1):
        super().__init__(payloads_file, maxturns=maxturns)
        self.interface = WafBrainInterface(score_threshold=score_threshold)

    def _get_score(self, payload):
        return self.interface.get_score(payload)
