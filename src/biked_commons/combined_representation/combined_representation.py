from typing import List, Dict

import pandas as pd

from biked_commons.combined_representation.conversions import Conversion
from biked_commons.combined_representation.merging import DuplicateColumnRemovalStrategy, \
    DuplicateColumnRemovalStrategies, RowMergeStrategy, RowMergeStrategies
from build.lib.biked_commons.exceptions import UserInputException


class DatasetDescription:
    def __init__(self, data: pd.DataFrame, conversions: List[Conversion]):
        self.conversions = conversions
        self.original_columns = list(data.columns)
        self._cleaned = False
        self.__data = data

    @classmethod
    def from_data(cls, data: pd.DataFrame):
        return DatasetDescription(data=data, conversions=[])

    def get_data(self, copy=True) -> pd.DataFrame:
        if self._cleaned:
            raise Exception("Get data should not be called after calling 'clean'")
        if copy:
            return self.__data.copy(deep=True)
        return self.__data

    def has_been_cleaned(self) -> bool:
        return self._cleaned

    def clean(self) -> None:
        self.__data = pd.DataFrame()  # empty
        self._cleaned = True


class CombinedRepresentation:
    def __init__(self,
                 id_to_description: Dict[str, DatasetDescription],
                 column_removal_strategy: DuplicateColumnRemovalStrategy = DuplicateColumnRemovalStrategies.KEEP_FIRST,
                 row_merge_strategy: RowMergeStrategy = RowMergeStrategies.IGNORE
                 ):
        self._id_to_description: Dict[str, DatasetDescription] = id_to_description
        self._merged_data: pd.DataFrame = self._merge_representations(column_removal_strategy, row_merge_strategy)
        self._clean_descriptions(id_to_description)

    def get_data(self, copy=True) -> pd.DataFrame:
        if copy:
            return self._merged_data.copy(deep=True)
        return self._merged_data

    def get_as_representation(self, representation_id: str) -> pd.DataFrame:
        description = self._id_to_description.get(representation_id, None)
        if description is None:
            raise UserInputException(f"Representation '{representation_id}' does not exist")

        result = self.get_data()
        for conversion in description.conversions:
            result[conversion.name] = conversion.inverse_conversion_function(result[conversion.master_name])

        return pd.DataFrame(result,
                            columns=description.original_columns,
                            index=result.index)

    def _merge_representations(self, column_removal_strategy, row_merge_strategy):
        converted_and_concatenated = pd.concat(
            [self._convert(description) for description in self._id_to_description.values()],
            axis=1)
        handled_columns = column_removal_strategy.remove_duplicate_columns(converted_and_concatenated)
        return row_merge_strategy.merge_rows(handled_columns, [d.get_data() for d in self._id_to_description.values()])

    def _convert(self, dataset_description: DatasetDescription) -> pd.DataFrame:
        start_data = dataset_description.get_data()
        for conversion in dataset_description.conversions:
            start_data[conversion.master_name] = conversion.conversion_function(start_data[conversion.name])
            start_data.drop(labels=[conversion.name], axis=1, inplace=True)
        return start_data

    def _clean_descriptions(self, id_to_description: Dict[str, DatasetDescription]):
        for desc in id_to_description.values():
            desc.clean()
