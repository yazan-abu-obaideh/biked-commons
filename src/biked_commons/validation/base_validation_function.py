from abc import abstractmethod, ABC

import pandas as pd


class ValidationFunction(ABC):
    @abstractmethod
    def friendly_name(self) -> str:
        pass

    @abstractmethod
    def validate(self, designs: pd.DataFrame) -> pd.DataFrame:
        pass
