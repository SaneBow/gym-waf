from .waf_score_env import WafScoreEnv
from .interfaces import WafBrainInterface


class WafBrainEnv(WafScoreEnv):
    def __init__(self, payloads_file, score_threshold, maxturns=20) -> None:
        super().__init__(payloads_file, score_threshold, maxturns=maxturns)
        self.interface = WafBrainInterface()

    def _get_score(self, payload):
        return self.interface.get_score(payload)
