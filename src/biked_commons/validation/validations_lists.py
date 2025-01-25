from typing import Callable, List

import numpy as np
import pandas as pd

from biked_commons.validation.validation_result import ValidationResult

__MULTIPLIER = 1000

# TODO: grab all validations from biked/functions

__COMBINED_VALIDATIONS_RAW = [
    lambda df: df["Saddle height"] < (df["ST Length"] * __MULTIPLIER) + 40,
    lambda df: df["Saddle height"] > ((df["ST Length"] * __MULTIPLIER) + df["Seatpost LENGTH"] + 30),
    lambda df: df["BSD rear"] < df["ERD rear"],
    lambda df: df["BSD front"] < df["ERD front"],
    lambda df: df["HT LX"] >= df["HT Length"],
    lambda df: ((df["HT UX"] + df["HT LX"]) >= df['HT Length']),
]

__CLIPS_VALIDATIONS_RAW = [
    lambda df: df["Saddle height"] < df["Seat tube length"] + 40,
    lambda df: df["Saddle height"] > (df["Seat tube length"] + df["Seatpost LENGTH"] + 30),
    lambda df: df["BSD rear"] < df["ERD rear"],
    lambda df: df["BSD front"] < df["ERD front"],
    lambda df: df["Head tube lower extension2"] >= df["Head tube length textfield"],
    lambda df: ((df["Head tube upper extension2"] + df["Head tube lower extension2"]) >= df[
        'Head tube length textfield']),
    lambda df: df["CS textfield"] <= 0,
]


def _wrap_function(validation_function: Callable):
    def wrapped_function(designs: pd.DataFrame) -> ValidationResult:
        try:
            validation_result = validation_function(designs).astype("int32")
            print(f"Validation result: fraction invalid [{np.sum(validation_result) / len(designs)}]")
            return ValidationResult(per_design_result=pd.DataFrame(validation_result, columns=["invalid"]),
                                    encountered_exception=False)
        except KeyError as e:
            print(f"Validation function failed {e}...")
            zeros = np.zeros(shape=(len(designs), 1))
            # TODO: should we be assuming that when a function fails, the issue is with the function?
            return ValidationResult(per_design_result=pd.DataFrame(zeros, columns=["invalid"]),
                                    encountered_exception=True)

    return wrapped_function


def _build_validations(raw_validations_list: List[Callable]):
    return [_wrap_function(validation_function=validation_function)
            for validation_function in raw_validations_list]


CLIPS_VALIDATION_FUNCTIONS = _build_validations(__CLIPS_VALIDATIONS_RAW)
COMBINED_VALIDATION_FUNCTIONS = _build_validations(__COMBINED_VALIDATIONS_RAW)
