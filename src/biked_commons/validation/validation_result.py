import attrs
import pandas as pd


@attrs.define(frozen=True)
class ValidationResult:
    validation_name: str
    per_design_result: pd.DataFrame # TODO: make it so this doesn't return booleans
    # TODO: return the difference from the "good value" (remember the conventions)
    encountered_exception: bool
