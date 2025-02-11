from typing import List

import pandas as pd

from biked_commons.validation.base_validation_function import validate_designs
from biked_commons.validation.clip_validation_functions import CLIPS_VALIDATIONS
from biked_commons.validation.raw_validation_functions import RAW_VALIDATION_FUNCTIONS
from biked_commons.validation.validation_result import ValidationResult


# noinspection PyPep8Naming
def validate_raw_BikeCad(designs: pd.DataFrame) -> List[ValidationResult]:
    return validate_designs(RAW_VALIDATION_FUNCTIONS, designs)


def validate_clip(designs: pd.DataFrame) -> List[ValidationResult]:
    return validate_designs(CLIPS_VALIDATIONS, designs)
