from .waf_score_env import WafScoreEnv
from .interfaces import WafBrainInterface

class WafBrainEnv(WafScoreEnv):
    def __init__(self) -> None:
        super(WafBrainEnv, self).__init__()
        interface = WafBrainInterface()
        self.score_function = interface.get_score