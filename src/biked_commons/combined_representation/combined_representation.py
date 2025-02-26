from typing import List, Callable

import attrs
import pandas as pd


@attrs.define(frozen=True)
class Conversion:
    name: str
    master_name: str
    conversion_function: Callable[[pd.Series], pd.Series]


class DatasetDescription:
    def __init__(self, data: pd.DataFrame, conversions: List[Conversion]):
        self.data = data
        self.conversions = conversions

    @classmethod
    def from_data(cls, data: pd.DataFrame):
        return DatasetDescription(data=data, conversions=[])


class CombinedRepresentation:
    def __init__(self, descriptions: List[DatasetDescription]):
        self.data = pd.concat([self.convert(d) for d in descriptions], axis=1)
        self.remove_extra_columns()

    def convert(self, d: DatasetDescription):
        start_data = d.data
        for conversion in d.conversions:
            start_data[conversion.master_name] = conversion.conversion_function(start_data[conversion.name])
            start_data.drop(labels=[conversion.name], axis=1, inplace=True)
        return start_data

    def get_data(self):
        return self.data

    def remove_extra_columns(self):
        columns = self.data.columns
        seen = set()
        for col_idx, column in enumerate(columns):
            if column in seen:
                self.data.drop(columns=[col_idx], axis=1, inplace=True)
            else:
                seen.add(column)
