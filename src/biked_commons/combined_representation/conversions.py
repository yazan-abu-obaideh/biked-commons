from typing import Callable

import attrs
import pandas as pd


@attrs.define(frozen=True)
class Conversion:
    name: str
    master_name: str
    conversion_function: Callable[[pd.Series], pd.Series]
    inverse_conversion_function: Callable[[pd.Series], pd.Series]


class MeterToMm(Conversion):
    @classmethod
    def from_names(cls, name: str, master_name: str):
        return Conversion(
            name=name,
            master_name=master_name,
            conversion_function=lambda x: x * 1000,
            inverse_conversion_function=lambda x: x / 1000
        )
