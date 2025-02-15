from typing import List

import numpy as np
import pandas as pd

from biked_commons.validation.base_validation_function import ValidationFunction


class SeatPostTooShort(ValidationFunction):
    def friendly_name(self) -> str:
        return "Seat post too short"

    def validate(self, designs: pd.DataFrame) -> pd.DataFrame:
        return (designs["Seat tube length"] + designs["Seatpost LENGTH"] + 30) - designs["Saddle height"]


class FrontWheelOuterDiameter(ValidationFunction):
    def friendly_name(self) -> str:
        return "Front wheel outer diameter smaller than rim outer diameter"

    def validate(self, designs: pd.DataFrame) -> pd.DataFrame:
        # noinspection PyTypeChecker
        return designs["Wheel diameter front"] - designs["BSD front"]


class RearWheelOuterDiameter(ValidationFunction):
    def friendly_name(self) -> str:
        return "Rear wheel outer diameter smaller than rim outer diamter"

    def validate(self, designs: pd.DataFrame) -> pd.DataFrame:
        # noinspection PyTypeChecker
        return designs["Wheel diameter rear"] - designs["BSD rear"]


class RearSpokes(ValidationFunction):
    def friendly_name(self) -> str:
        return "Rear spokes too short"

    def validate(self, designs: pd.DataFrame) -> pd.DataFrame:
        return designs["ERD rear"] - (designs["BSD rear"] - designs["Rim depth rear"] * 2)


class FrontSpokes(ValidationFunction):
    def friendly_name(self) -> str:
        return "Front spokes too short"

    def validate(self, designs: pd.DataFrame) -> pd.DataFrame:
        return designs["ERD front"] - (designs["BSD front"] - designs["Rim depth front"] * 2)


class RearSpokesTooLong(ValidationFunction):
    def friendly_name(self) -> str:
        return "Rear spokes too long"

    def validate(self, designs: pd.DataFrame) -> pd.DataFrame:
        # noinspection PyTypeChecker
        return designs["Wheel diameter rear"] - designs["ERD rear"]


class BsdSmaller(ValidationFunction):
    def friendly_name(self) -> str:
        return "BSD < ERD rear"

    def validate(self, designs: pd.DataFrame) -> pd.DataFrame:
        # noinspection PyTypeChecker
        return designs["BSD rear"] - designs["ERD rear"]


class BsdSmallerFront(ValidationFunction):
    def friendly_name(self) -> str:
        return "BSD < ERD front"

    def validate(self, designs: pd.DataFrame) -> pd.DataFrame:
        # noinspection PyTypeChecker
        return designs["BSD front"] - designs["ERD front"]


class FrontSpokesTooLong(ValidationFunction):
    def friendly_name(self) -> str:
        return "Front spokes too long"

    def validate(self, designs: pd.DataFrame) -> pd.DataFrame:
        # noinspection PyTypeChecker
        return designs["Wheel diameter front"] - designs["ERD front"]


class CheckHtlx(ValidationFunction):
    def friendly_name(self) -> str:
        return "HTLX > HTL"

    def validate(self, designs: pd.DataFrame) -> pd.DataFrame:
        # noinspection PyTypeChecker
        return designs["Head tube length textfield"] - designs["Head tube lower extension2"]


class CheckHtlxHtux(ValidationFunction):
    def friendly_name(self) -> str:
        return "HTLX+HTUX>HTL"

    def validate(self, designs: pd.DataFrame) -> pd.DataFrame:
        extension_ = designs["Head tube upper extension2"] + designs["Head tube lower extension2"]
        # noinspection PyTypeChecker
        return designs["Head tube length textfield"] - extension_


class CheckDownTubeReachesHeadTubeJunction(ValidationFunction):
    def friendly_name(self) -> str:
        return "Down tube too short to reach head tube junction"

    def validate(self, designs: pd.DataFrame) -> pd.DataFrame:
        Stack = designs["Stack"]
        HTL = designs["Head tube length textfield"]
        HTLX = designs["Head tube lower extension2"]
        HTA = designs["Head angle"] * np.pi / 180
        DTL = designs["DT Length"]
        # TODO: check that this is correct
        DTJY = (Stack - (HTL - HTLX) * np.sin(HTA))
        return np.logical_and(HTA < np.pi / 2, (DTJY ** 2 >= DTL ** 2))


class CheckDownTubeIntersectsFrontWheel(ValidationFunction):
    def friendly_name(self) -> str:
        return "Down tube intersects Front Wheel"

    def validate(self, designs: pd.DataFrame) -> pd.DataFrame:
        Stack = designs["Stack"]
        HTL = designs["Head tube length textfield"]
        HTLX = designs["Head tube lower extension2"]
        HTA = designs["Head angle"] * np.pi / 180
        DTL = designs["DT Length"]
        BBD = designs["BB textfield"]
        DTJY = Stack - (HTL - HTLX) * np.sin(HTA)
        DTJX = np.sqrt(DTL ** 2 - DTJY ** 2)
        FWX = DTJX + (DTJY - BBD) / np.tan(HTA)
        FCD = np.sqrt(FWX ** 2 + BBD ** 2)
        FBSD = designs["BSD front"]
        DTOD = designs["Down tube diameter"]

        ang = np.arctan2(DTJY, DTJX) - np.arctan2(BBD, FWX)
        return np.logical_and(ang < np.pi / 2,
                              np.sin(ang) * FCD < FBSD / 2 - DTOD)


RAW_VALIDATION_FUNCTIONS: List[ValidationFunction] = [
    SeatPostTooShort(),
    FrontWheelOuterDiameter(),
    RearWheelOuterDiameter(),
    RearSpokes(),
    FrontSpokes(),
    RearSpokesTooLong(),
    BsdSmaller(),
    BsdSmallerFront(),
    FrontSpokesTooLong(),
    CheckHtlx(),
    CheckHtlxHtux(),
    CheckDownTubeReachesHeadTubeJunction(),
    CheckDownTubeIntersectsFrontWheel(),
]
