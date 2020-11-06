from .waf_score_env import WafScoreEnv
from .interfaces import WafBrainInterface

class WafBrainEnv(WafScoreEnv):
    def __init__(self, payload, score_threshold, maxturns=20) -> None:
        super(WafBrainEnv, self).__init__(payload, score_threshold, maxturns=maxturns)
        interface = WafBrainInterface()
        self.score_function = interface.get_score