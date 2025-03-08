from abc import ABCMeta, abstractmethod

import attrs
import pandas as pd


class ReversibleConversion(metaclass=ABCMeta):
    @abstractmethod
    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def reverse(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        pass


@attrs.define(frozen=True)
class RenameColumn(ReversibleConversion):
    from_name: str
    to_name: str

    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        return dataframe.rename(columns={
            self.from_name: self.to_name
        })

    def reverse(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        return dataframe.rename(columns={
            self.to_name: self.from_name
        })


@attrs.define(frozen=True)
class ScaleColumn(ReversibleConversion):
    column: str
    multiplier: float

    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        return dataframe.assign(**{
            self.column: dataframe[self.column] * self.multiplier
        })

    def reverse(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        return dataframe.assign(
            **{
                self.column: dataframe[self.column] / self.multiplier
            })
