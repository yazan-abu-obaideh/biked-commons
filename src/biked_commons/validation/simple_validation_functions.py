import pandas as pd

from biked_commons.validation.base_validation_function import ValidationFunction


class SeatPostTooShort(ValidationFunction):
    def friendly_name(self) -> str:
        return "Seat post too short"

    def validate(self, designs: pd.DataFrame) -> pd.DataFrame:
        return designs["Saddle height"] > (designs["Seat tube length"] + designs["Seatpost LENGTH"] + 30)


class FrontWheelOuterDiameter(ValidationFunction):
    def friendly_name(self) -> str:
        return "Front wheel outer diameter smaller than rim outer diameter"

    def validate(self, designs: pd.DataFrame) -> pd.DataFrame:
        # noinspection PyTypeChecker
        return designs["Wheel diameter front"] < designs["BSD front"]


class RearWheelOuterDiameter(ValidationFunction):
    def friendly_name(self) -> str:
        return "Rear wheel outer diameter smaller than rim outer diamter"

    def validate(self, designs: pd.DataFrame) -> pd.DataFrame:
        # noinspection PyTypeChecker
        return designs["Wheel diameter rear"] < designs["BSD rear"]


class RearSpokes(ValidationFunction):
    def friendly_name(self) -> str:
        return "Rear spokes too short"

    def validate(self, designs: pd.DataFrame) -> pd.DataFrame:
        return (designs["BSD rear"] - designs["Rim depth rear"] * 2) > designs["ERD rear"]


class FrontSpokes(ValidationFunction):
    def friendly_name(self) -> str:
        return "Front spokes too short"

    def validate(self, designs: pd.DataFrame) -> pd.DataFrame:
        return (designs["BSD front"] - designs["Rim depth front"] * 2) > designs["ERD front"]


class RearSpokesTooLong(ValidationFunction):
    def friendly_name(self) -> str:
        return "Rear spokes too long"

    def validate(self, designs: pd.DataFrame) -> pd.DataFrame:
        # noinspection PyTypeChecker
        return designs["Wheel diameter rear"] < designs["ERD rear"]


class BsdSmaller(ValidationFunction):
    def friendly_name(self) -> str:
        return "BSD < ERD rear"

    def validate(self, designs: pd.DataFrame) -> pd.DataFrame:
        # noinspection PyTypeChecker
        return designs["BSD rear"] < designs["ERD rear"]


class BsdSmallerFront(ValidationFunction):
    def friendly_name(self) -> str:
        return "BSD < ERD front"

    def validate(self, designs: pd.DataFrame) -> pd.DataFrame:
        # noinspection PyTypeChecker
        return designs["BSD front"] < designs["ERD front"]


class FrontSpokesTooLong(ValidationFunction):
    def friendly_name(self) -> str:
        return "Front spokes too long"

    def validate(self, designs: pd.DataFrame) -> pd.DataFrame:
        # noinspection PyTypeChecker
        return designs["Wheel diameter front"] < designs["ERD front"]


class CheckHtlx(ValidationFunction):
    def friendly_name(self) -> str:
        return "HTLX > HTL"

    def validate(self, designs: pd.DataFrame) -> pd.DataFrame:
        # noinspection PyTypeChecker
        return designs["Head tube lower extension2"] >= designs["Head tube length textfield"]


class CheckHtlxHtux(ValidationFunction):
    def friendly_name(self) -> str:
        return "HTLX+HTUX>HTL"

    def validate(self, designs: pd.DataFrame) -> pd.DataFrame:
        extension_ = designs["Head tube upper extension2"] + designs["Head tube lower extension2"]
        # noinspection PyTypeChecker
        return extension_ >= designs["Head tube length textfield"]
