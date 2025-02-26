from typing import List, Callable, Dict

import attrs
import pandas as pd


@attrs.define(frozen=True)
class Conversion:
    name: str
    master_name: str
    conversion_function: Callable[[pd.Series], pd.Series]
    inverse_conversion_function: Callable[[pd.Series], pd.Series]


class DatasetDescription:
    def __init__(self, data: pd.DataFrame, conversions: List[Conversion]):
        self.data = data
        self.conversions = conversions

    @classmethod
    def from_data(cls, data: pd.DataFrame):
        return DatasetDescription(data=data, conversions=[])


class CombinedRepresentation:
    def __init__(self, descriptions: Dict[str, DatasetDescription]):
        self.original: Dict[str, DatasetDescription] = descriptions
        self.data: pd.DataFrame = pd.concat([self.convert(d) for title, d in descriptions.items()], axis=1)
        self.remove_extra_columns()

    def get_data(self) -> pd.DataFrame:
        return self.data

    def get_as_representation(self, representation_id) -> pd.DataFrame:
        conversions = self.original[representation_id].conversions
        result: pd.DataFrame = self.data.copy(deep=True)
        for conversion in conversions:
            result[conversion.name] = conversion.inverse_conversion_function(result[conversion.master_name])
        original_columns = set(self.original[representation_id].data.columns)
        for c in result.columns:
            if c not in original_columns:
                result.drop(labels=[c], axis=1, inplace=True)
        return result

    def convert(self, d: DatasetDescription) -> pd.DataFrame:
        start_data = d.data.copy(deep=True)
        for conversion in d.conversions:
            start_data[conversion.master_name] = conversion.conversion_function(start_data[conversion.name])
            start_data.drop(labels=[conversion.name], axis=1, inplace=True)
        return start_data

    def remove_extra_columns(self):
        columns = self.data.columns
        seen = set()
        desired = []
        for col_idx, column in enumerate(columns):
            if column not in seen:
                seen.add(column)
                desired.append(col_idx)
        self.data = self.data.iloc[:, desired]
