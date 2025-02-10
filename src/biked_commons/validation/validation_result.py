import attrs
import pandas as pd


@attrs.define(frozen=True)
class ValidationResult:
    validation_name: str
    per_design_result: pd.DataFrame
    encountered_exception: bool
