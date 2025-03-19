import math
from typing import List

import torch

from biked_commons.validation.base_validation_function import ValidationFunction


class SeatPostTooShort(ValidationFunction):
    def friendly_name(self) -> str:
        return "Seat post too short"

    def variable_names(self) -> List[str]:
        return ["Seat tube length", "Seatpost LENGTH", "Saddle height"]

    def validate(self, designs: torch.Tensor) -> torch.Tensor:
        seat_tube_length, seatpost_length, saddle_height = designs[:, :len(self.variable_names())].T
        return (seat_tube_length + seatpost_length + 30) - saddle_height


class FrontWheelOuterDiameter(ValidationFunction):
    def friendly_name(self) -> str:
        return "Front wheel outer diameter smaller than rim outer diameter"

    def variable_names(self) -> List[str]:
        return ["Wheel diameter front", "BSD front"]

    def validate(self, designs: torch.Tensor) -> torch.Tensor:
        wheel_diameter_front, bsd_front = designs[:, :len(self.variable_names())].T
        return wheel_diameter_front - bsd_front


class RearWheelOuterDiameter(ValidationFunction):
    def friendly_name(self) -> str:
        return "Rear wheel outer diameter smaller than rim outer diameter"

    def variable_names(self) -> List[str]:
        return ["Wheel diameter rear", "BSD rear"]

    def validate(self, designs: torch.Tensor) -> torch.Tensor:
        wheel_diameter_rear, bsd_rear = designs[:, :len(self.variable_names())].T
        return wheel_diameter_rear - bsd_rear


class RearSpokes(ValidationFunction):
    def friendly_name(self) -> str:
        return "Rear spokes too short"

    def variable_names(self) -> List[str]:
        return ["ERD rear", "BSD rear", "Rim depth rear"]

    def validate(self, designs: torch.Tensor) -> torch.Tensor:
        erd_rear, bsd_rear, rim_depth_rear = designs[:, :len(self.variable_names())].T
        return erd_rear - (bsd_rear - 2 * rim_depth_rear)


class FrontSpokes(ValidationFunction):
    def friendly_name(self) -> str:
        return "Front spokes too short"

    def variable_names(self) -> List[str]:
        return ["ERD front", "BSD front", "Rim depth front"]

    def validate(self, designs: torch.Tensor) -> torch.Tensor:
        erd_front, bsd_front, rim_depth_front = designs[:, :len(self.variable_names())].T
        return erd_front - (bsd_front - 2 * rim_depth_front)


class RearSpokesTooLong(ValidationFunction):
    def friendly_name(self) -> str:
        return "Rear spokes too long"

    def variable_names(self) -> List[str]:
        return ["Wheel diameter rear", "ERD rear"]

    def validate(self, designs: torch.Tensor) -> torch.Tensor:
        wheel_diameter_rear, erd_rear = designs[:, :len(self.variable_names())].T
        return wheel_diameter_rear - erd_rear


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


class FrontSpokesTooLong(ValidationFunction):
    def friendly_name(self) -> str:
        return "Front spokes too long"

    def variable_names(self) -> List[str]:
        return ["Wheel diameter front", "ERD front"]

    def validate(self, designs: torch.Tensor) -> torch.Tensor:
        wheel_diameter_front, erd_front = designs[:, :len(self.variable_names())].T
        return wheel_diameter_front - erd_front


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


class CheckDownTubeReachesHeadTubeJunction(ValidationFunction):
    def friendly_name(self) -> str:
        return "Down tube too short to reach head tube junction"

    def variable_names(self) -> List[str]:
        return ["Stack", "Head tube length textfield", "Head tube lower extension2", "Head angle", "DT Length"]

    def validate(self, designs: torch.Tensor) -> torch.Tensor:
        stack, head_tube_length, head_tube_lower_extension, head_angle, dt_length = designs[:,
                                                                                    :len(self.variable_names())].T

        # Convert to radians and ensure head_angle_rad is within valid range
        head_angle_rad = head_angle * math.pi / 180
        head_angle_rad = torch.clamp(head_angle_rad, min=0, max=math.pi / 2)

        # Compute junction y-coordinate
        dtjy = stack - (head_tube_length - head_tube_lower_extension) * torch.sin(head_angle_rad)

        # Compute validity score (negative = more valid, positive = more invalid)
        angle_validity = head_angle_rad - (math.pi / 2)
        length_validity = dt_length ** 2 - dtjy ** 2

        # Combine the two conditions smoothly
        result = torch.where(head_angle_rad < math.pi / 2, length_validity, angle_validity)

        return result


class CheckDownTubeIntersectsFrontWheel(ValidationFunction):
    def friendly_name(self) -> str:
        return "Down tube intersects Front Wheel"

    def variable_names(self) -> List[str]:
        return ["Stack", "Head tube length textfield", "Head tube lower extension2", "Head angle", "DT Length",
                "BB textfield", "BSD front", "Down tube diameter"]

    def validate(self, designs: torch.Tensor) -> torch.Tensor:
        (stack, head_tube_length, head_tube_lower_extension,
         head_angle, dt_length, bb_textfield,
         bsd_front, dt_diameter) = designs[:, :len(self.variable_names())].T

        # Convert to radians and ensure head_angle_rad is within a valid range
        head_angle_rad = head_angle * math.pi / 180
        head_angle_rad = torch.clamp(head_angle_rad, min=0, max=math.pi / 2)

        # Compute junction coordinates
        dtjy = stack - (head_tube_length - head_tube_lower_extension) * torch.sin(head_angle_rad)
        dtjx = torch.sqrt(dt_length ** 2 - dtjy ** 2)
        fwx = dtjx + (dtjy - bb_textfield) / torch.tan(head_angle_rad)
        fcd = torch.sqrt(fwx ** 2 + bb_textfield ** 2)

        # Compute intersection condition
        ang = torch.atan2(dtjy, dtjx) - torch.atan2(bb_textfield, fwx)
        intersection_validity = torch.sin(ang) * fcd - (bsd_front / 2 - dt_diameter)  # Positive means invalid

        # If head angle is too large, push it towards the valid region
        angle_validity = head_angle_rad - (math.pi / 2)  # Positive means invalid

        # Smoothly combine both conditions
        result = torch.where(head_angle_rad < math.pi / 2, intersection_validity, angle_validity)

        return result


RAW_VALIDATION_FUNCTIONS: List[ValidationFunction] = [
    SeatPostTooShort(),
    FrontWheelOuterDiameter(),
    RearWheelOuterDiameter(),
    RearSpokes(),
    FrontSpokes(),
    RearSpokesTooLong(),
    BsdFrontTooSmall(),
    BsdRearTooSmall(),
    FrontSpokesTooLong(),
    HeadTubeLengthTooGreat(),
    CheckDownTubeReachesHeadTubeJunction(),
    CheckDownTubeReachesHeadTubeJunction(),
    CheckDownTubeIntersectsFrontWheel(),
]
