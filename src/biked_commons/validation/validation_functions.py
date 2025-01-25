from abc import ABC, abstractmethod

import pandas as pd


class ValidationFunction(ABC):
    @abstractmethod
    def friendly_name(self) -> str:
        pass

    @abstractmethod
    def validate(self, designs: pd.DataFrame) -> pd.DataFrame:
        pass


class DisallowedNegativeValues(ValidationFunction):
    def friendly_name(self) -> str:
        return "Disallowed negative values"

    def validate(self, designs: pd.DataFrame) -> pd.DataFrame:
        raise Exception("Unimplemented function!")


class SeatPostTooShort(ValidationFunction):
    def friendly_name(self) -> str:
        return "Seat post too short"

    def validate(self, designs: pd.DataFrame) -> pd.DataFrame:
        length_ = (designs["Seat tube length"] + designs["Seatpost LENGTH"] + 30)
        return designs["Saddle height"] > length_


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
        rear_ = (designs["BSD rear"] - designs["Rim depth rear"] * 2)
        return rear_ > designs["ERD rear"]


class FrontSpokes(ValidationFunction):
    def friendly_name(self) -> str:
        return "Front spokes too short"

    def validate(self, designs: pd.DataFrame) -> pd.DataFrame:
        front_ = (designs["BSD front"] - designs["Rim depth front"] * 2)
        return front_ > designs["ERD front"]


class RearSpokesTooLong(ValidationFunction):
    def friendly_name(self) -> str:
        return "Rear spokes too long"

    def validate(self, designs: pd.DataFrame) -> pd.DataFrame:
        # noinspection PyTypeChecker
        return designs["Wheel diameter rear"] < designs["ERD rear"]


_MANDATORY_POSITIVE = ['CS textfield', 'Stack', 'Head angle',
                       'Head tube length textfield', 'Seat tube length',
                       'Seat angle', 'DT Length', 'BB diameter', 'ttd', 'dtd', 'csd', 'ssd',
                       'Chain stay position on BB', 'MATERIAL',
                       'Head tube upper extension2', 'Seat tube extension2',
                       'Head tube lower extension2', 'SEATSTAYbrdgshift', 'CHAINSTAYbrdgshift',
                       'SEATSTAYbrdgdia1', 'CHAINSTAYbrdgdia1', 'SEATSTAYbrdgCheck',
                       'CHAINSTAYbrdgCheck', 'Dropout spacing',
                       'Wall thickness Bottom Bracket', 'Wall thickness Top tube',
                       'Wall thickness Head tube', 'Wall thickness Down tube',
                       'Wall thickness Chain stay', 'Wall thickness Seat stay',
                       'Wall thickness Seat tube', 'ERD rear', 'Wheel width rear',
                       'Dropout spacing style', 'BSD front', 'Wheel width front', 'ERD front',
                       'BSD rear', 'Fork type', 'Stem kind', 'Display AEROBARS',
                       'Handlebar style', 'Head tube type', 'BB length', 'Head tube diameter',
                       'Wheel cut', 'Seat tube diameter', 'Top tube type',
                       'bottle SEATTUBE0 show', 'bottle DOWNTUBE0 show',
                       'Front Fender include', 'Rear Fender include', 'BELTorCHAIN',
                       'Number of cogs', 'Number of chainrings', 'Display RACK',
                       'FIRST color R_RGB', 'FIRST color G_RGB', 'FIRST color B_RGB',
                       'RIM_STYLE front', 'RIM_STYLE rear', 'SPOKES composite front',
                       'SBLADEW front', 'SBLADEW rear', 'Saddle length', 'Saddle height',
                       'Down tube diameter', 'Seatpost LENGTH']
