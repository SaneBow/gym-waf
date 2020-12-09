from .base_interface import LocalInterface, ClassificationFailure
from libinjection import is_sql_injection


class LibinjectionInterface(LocalInterface):
    def __init__(self) -> None:
        super().__init__()

    def get_label(self, payload):
        result = is_sql_injection(payload)
        if result is None:
            raise ClassificationFailure('WAF-Brain classifier returns None')
        return result['is_sqli']