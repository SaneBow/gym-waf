from .waf_label_env import WafLabelEnv
from .interfaces import LibinjectionInterface

class LibinjectionEnv(WafLabelEnv):
    def __init__(self, payload, maxturns=20) -> None:
        super(LibinjectionEnv, self).__init__(payload, maxturns=maxturns)
        interface = LibinjectionInterface()
        self.label_function = interface.get_label