from abc import abstractmethod, ABC
from typing import List

import numpy as np
import pandas as pd

from biked_commons.validation.validation_result import ValidationResult


# TODO: write validation functions to [optionally] be able to grab values from the default bike when not found?

class ValidationFunction(ABC):
    @abstractmethod
    def friendly_name(self) -> str:
        """
        Should return a user-friendly and easily comprehensible name for the validation in question.
        """
        pass

    @abstractmethod
    def validate(self, designs: pd.DataFrame) -> pd.DataFrame:
        """
        Should return a pandas dataframe with shape (len(designs), 1) or (len(designs),)
        The values in the dataframe represent validity. 1 is invalid, 0 is valid.
        """
        pass


def validate_designs(validation_functions: List[ValidationFunction],
                     designs: pd.DataFrame) -> List[ValidationResult]:
    results = []
    for validation_function in validation_functions:
        try:
            res = pd.DataFrame(data=(validation_function.validate(designs)), columns=["invalid"], index=designs.index)
            validation_result = ValidationResult(validation_name=validation_function.friendly_name(),
                                                 per_design_result=res,
                                                 encountered_exception=False)
        except Exception as e:
            print(f"Validation function [{validation_function.friendly_name()}] encountered exception [{e}]")
            res = pd.DataFrame(np.ones(shape=(len(designs), 1)), columns=["invalid"], index=designs.index)
            validation_result = ValidationResult(validation_name=validation_function.friendly_name(),
                                                 per_design_result=res,
                                                 encountered_exception=True)
        results.append(validation_result)
    return results
