import pandas as pd

from biked_commons.transformation.representation import Representation
from biked_commons.validation.base_validation_function import construct_dataframe_validator
from biked_commons.validation.clip_validation_functions import CLIPS_VALIDATIONS
from biked_commons.validation.raw_validation_functions import RAW_VALIDATION_FUNCTIONS


def infer_representation(dataframe: pd.DataFrame) -> Representation:
    pass


def is_raw(dataframe: pd.DataFrame) -> bool:
    pass


def is_biked(dataframe: pd.DataFrame) -> bool:
    pass


def is_clip(dataframe: pd.DataFrame) -> bool:
    pass


# noinspection PyPep8Naming
def validate_raw_BikeCad(designs: pd.DataFrame) -> pd.DataFrame:
    validator = construct_dataframe_validator(RAW_VALIDATION_FUNCTIONS)
    return validator(designs)


def validate_clip(designs: pd.DataFrame) -> pd.DataFrame:
    validator = construct_dataframe_validator(CLIPS_VALIDATIONS)
    return validator(designs)
