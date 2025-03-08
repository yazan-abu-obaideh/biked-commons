import abc
from typing import Callable

import attrs
import pandas as pd


class ReversibleConversion(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        pass

    @abc.abstractmethod
    def reverse(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        pass


class RenameColumn(ReversibleConversion):
    def __init__(self, from_name: str, to_name: str):
        self._from = from_name
        self._to = to_name

    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        return dataframe.rename(columns={
            self._from: self._to
        })

    def reverse(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        return dataframe.rename(columns={
            self._to: self._from
        })


class ScaleColumn(ReversibleConversion):
    def __init__(self, column: str, multiplier: float):
        self._multiplier = multiplier
        self._column = column

    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        return dataframe.assign(**{
            self._column: dataframe[self._column] * self._multiplier
        })

    def reverse(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        return dataframe.assign(
            **{
                self._column: dataframe[self._column] / self._multiplier
            })


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
