import inspect
from typing import List

import numpy as np
import pandas as pd

from biked_commons.validation import simple_validation_functions
from biked_commons.validation.base_validation_function import ValidationFunction
from biked_commons.validation.validation_result import ValidationResult


def is_subclass(o):
    is_abstract_base_class = o is ValidationFunction
    if is_abstract_base_class:
        return False
    return inspect.isclass(o) and issubclass(o, ValidationFunction)


__RAW_VALIDATION_FUNCTIONS: List[ValidationFunction] = [member[1]() for member in
                                                        inspect.getmembers(simple_validation_functions,
                                                                           predicate=is_subclass)]


# noinspection PyPep8Naming
def validate_raw_BikeCad(designs: pd.DataFrame, verbose=False):
    results = []
    for validation_function in __RAW_VALIDATION_FUNCTIONS:
        try:
            res = validation_function.validate(designs)
            results.append(ValidationResult(per_design_result=res, encountered_exception=False))
            if verbose:
                print(validation_function.friendly_name() + " ran successfully")
        except Exception as e:
            print(f"Validation function {validation_function.friendly_name()} encountered exception {e}")
            res = pd.DataFrame(np.ones(shape=(len(designs), 1)))
            results.append(ValidationResult(per_design_result=res, encountered_exception=True))
    return results
