from .base_interface import LocalInterface, ClassificationFailure
from waf_brain.inferring import process_payload
from keras.models import load_model
import os

class WafBrainInterface(LocalInterface):
    def __init__(self) -> None:
        super().__init__()
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '../models/waf-brain.h5')
        self.model = load_model(model_path)

    def get_score(self, payload):
        result = process_payload(self.model, 'q', [payload], True)
        if result is None:
            raise ClassificationFailure('WAF-Brain classifier returns None')
        return result['score']