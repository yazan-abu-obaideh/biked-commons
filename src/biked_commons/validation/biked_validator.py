from typing import List

import pandas as pd

from biked_commons.validation.validation_result import ValidationResult
from biked_commons.validation.validations_lists import CLIPS_VALIDATION_FUNCTIONS


class BikedValidator:
    def validate_clip(self, clip_data: pd.DataFrame) -> List[ValidationResult]:
        return [f(clip_data) for f in CLIPS_VALIDATION_FUNCTIONS]
