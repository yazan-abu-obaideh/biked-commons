from typing import List
import torch
from biked_commons.validation.base_validation_function import ValidationFunction


class SaddleHeightTooSmall(ValidationFunction):
    def friendly_name(self) -> str:
        return "Saddle height too small"

    def variable_names(self) -> List[str]:
        return ["Saddle height", "Seat tube length"]

    def validate(self, designs: torch.tensor) -> torch.tensor:
        saddle_height, seat_tube_length = designs[:, :len(self.variable_names())].T
        return saddle_height - (seat_tube_length + 40)


class SaddleHeightTooGreat(ValidationFunction):
    def friendly_name(self) -> str:
        return "Saddle height too great"

    def variable_names(self) -> List[str]:
        return ["Seat tube length", "Seatpost LENGTH", "Saddle height"]

    def validate(self, designs: torch.tensor) -> torch.tensor:
        seat_tube_length, seatpost_length, saddle_height = designs[:, :len(self.variable_names())].T
        return (seat_tube_length + seatpost_length + 30) - saddle_height


class BsdRearTooSmall(ValidationFunction):
    def friendly_name(self) -> str:
        return "Bsd rear too small"

    def variable_names(self) -> List[str]:
        return ["BSD rear", "ERD rear"]

    def validate(self, designs: torch.tensor) -> torch.tensor:
        bsd_rear, erd_rear = designs[:, :len(self.variable_names())].T
        return bsd_rear - erd_rear


class BsdFrontTooSmall(ValidationFunction):
    def friendly_name(self) -> str:
        return "Bsd front too small"

    def variable_names(self) -> List[str]:
        return ["BSD front", "ERD front"]

    def validate(self, designs: torch.tensor) -> torch.tensor:
        bsd_front, erd_front = designs[:, :len(self.variable_names())].T
        return bsd_front - erd_front


class HeadTubeLowerExtensionTooGreat(ValidationFunction):
    def friendly_name(self) -> str:
        return "Head tube lower extension too great"

    def variable_names(self) -> List[str]:
        return ["Head tube length textfield", "Head tube lower extension2"]

    def validate(self, designs: torch.tensor) -> torch.tensor:
        head_tube_length, head_tube_lower_extension = designs[:, :len(self.variable_names())].T
        return head_tube_length - head_tube_lower_extension


class HeadTubeLengthTooGreat(ValidationFunction):
    def friendly_name(self) -> str:
        return "Head tube length too great"

    def variable_names(self) -> List[str]:
        return ["Head tube length textfield", "Head tube upper extension2", "Head tube lower extension2"]

    def validate(self, designs: torch.tensor) -> torch.tensor:
        head_tube_length, head_tube_upper_extension, head_tube_lower_extension = designs[:,
                                                                                 :len(self.variable_names())].T
        return head_tube_length - (head_tube_upper_extension + head_tube_lower_extension)


class ChainStayLessThanZero(ValidationFunction):
    def friendly_name(self) -> str:
        return "Chain stay less than zero"

    def variable_names(self) -> List[str]:
        return ["CS textfield"]

    def validate(self, designs: torch.tensor) -> torch.tensor:
        cs_textfield, = designs[:, :len(self.variable_names())].T
        return 0 - cs_textfield


CLIPS_VALIDATIONS: List[ValidationFunction] = [
    SaddleHeightTooSmall(),
    SaddleHeightTooGreat(),
    BsdRearTooSmall(),
    BsdFrontTooSmall(),
    HeadTubeLowerExtensionTooGreat(),
    HeadTubeLengthTooGreat(),
    ChainStayLessThanZero()
]
