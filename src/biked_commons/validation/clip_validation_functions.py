from typing import List

import pandas as pd

from biked_commons.validation.base_validation_function import ValidationFunction


class SaddleHeightTooSmall(ValidationFunction):
    def friendly_name(self) -> str:
        return "Saddle height too small"

    def validate(self, designs: pd.DataFrame) -> pd.DataFrame:
        return designs["Saddle height"] - (designs["Seat tube length"] + 40)


class SaddleHeightTooGreat(ValidationFunction):
    def friendly_name(self) -> str:
        return "Saddle height too great"

    def validate(self, designs: pd.DataFrame) -> pd.DataFrame:
        return (designs["Seat tube length"] + designs["Seatpost LENGTH"] + 30) - designs["Saddle height"]


class BsdRearTooSmall(ValidationFunction):
    def friendly_name(self) -> str:
        return "Bsd rear too small"

    def validate(self, designs: pd.DataFrame) -> pd.DataFrame:
        # noinspection PyTypeChecker
        return designs["BSD rear"] - designs["ERD rear"]


class BsdFrontTooSmall(ValidationFunction):
    def friendly_name(self) -> str:
        return "Bsd front too small"

    def validate(self, designs: pd.DataFrame) -> pd.DataFrame:
        # noinspection PyTypeChecker
        return designs["BSD front"] - designs["ERD front"]


class HeadTubeLowerExtensionTooGreat(ValidationFunction):
    def friendly_name(self) -> str:
        return "Head tube lower extension too great"

    def validate(self, designs: pd.DataFrame) -> pd.DataFrame:
        # noinspection PyTypeChecker
        return designs["Head tube length textfield"] - designs["Head tube lower extension2"]


class HeadTubeLengthTooGreat(ValidationFunction):
    def friendly_name(self) -> str:
        return "Head tube length too great"

    def validate(self, designs: pd.DataFrame) -> pd.DataFrame:
        # noinspection PyTypeChecker
        return designs['Head tube length textfield'] - (
                designs["Head tube upper extension2"] + designs["Head tube lower extension2"])


class ChainStayLessThanZero(ValidationFunction):
    def friendly_name(self) -> str:
        return "Chain stay less than zero"

    def validate(self, designs: pd.DataFrame) -> pd.DataFrame:
        return 0 - designs["CS textfield"]


CLIPS_VALIDATIONS: List[ValidationFunction] = [
    SaddleHeightTooSmall(),
    SaddleHeightTooGreat(),
    BsdRearTooSmall(),
    BsdFrontTooSmall(),
    HeadTubeLowerExtensionTooGreat(),
    HeadTubeLengthTooGreat(),
    ChainStayLessThanZero()
]
