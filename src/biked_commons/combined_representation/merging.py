import abc
from functools import reduce
from typing import List

import pandas as pd


class DuplicateColumnRemovalStrategy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def remove_duplicate_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        pass


class RowMergeStrategy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def merge_rows(self, merged_data: pd.DataFrame, original_data: List[pd.DataFrame]) -> pd.DataFrame:
        pass


class _IgnoreColumns(DuplicateColumnRemovalStrategy):

    def remove_duplicate_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        return data


class _KeepFirst(DuplicateColumnRemovalStrategy):

    def remove_duplicate_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        starting_columns = data.columns
        seen = set()
        desired = []
        for index, column in enumerate(starting_columns):
            if column not in seen:
                seen.add(column)
                desired.append(index)
        return data.iloc[:, desired]


class _IgnoreRows(RowMergeStrategy):

    def merge_rows(self, merged_data: pd.DataFrame, original_data: List[pd.DataFrame]) -> pd.DataFrame:
        return merged_data


class _StrictIntersection(RowMergeStrategy):

    def merge_rows(self, merged_data: pd.DataFrame, original_data: List[pd.DataFrame]) -> pd.DataFrame:
        sets = [set(data.index) for data in original_data]
        intersection = reduce(set.intersection, sets)
        return pd.DataFrame(merged_data, index=list(intersection))


class DuplicateColumnRemovalStrategies:
    KEEP_FIRST = _KeepFirst()
    IGNORE = _IgnoreColumns()


class RowMergeStrategies:
    IGNORE = _IgnoreRows()
    STRICT_INTERSECTION = _StrictIntersection()
