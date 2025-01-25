import attrs
import pandas as pd


@attrs.define(frozen=True)
class ValidationResult:
    per_design_result: pd.DataFrame
    encountered_exception: bool
