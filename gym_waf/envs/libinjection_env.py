from .waf_label_env import WafLabelEnv
from .interfaces import LibinjectionInterface


class LibinjectionEnv(WafLabelEnv):
    def __init__(self, payloads_file, maxturns=20):
        super().__init__(payloads_file, maxturns=maxturns)
        self.interface = LibinjectionInterface()

    def _get_label(self, payload):
        return self.interface.get_label(payload)
