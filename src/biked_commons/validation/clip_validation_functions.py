from typing import List
import torch
import math
from biked_commons.validation.base_validation_function import ValidationFunction



class SaddleHeightTooSmall(ValidationFunction):
    def friendly_name(self) -> str:
        return "Saddle height too small"

    def variable_names(self) -> List[str]:
        return ["Saddle height", "Seat tube length"]

    def validate(self, designs: torch.tensor) -> torch.tensor:
        saddle_height, seat_tube_length = designs[:, :len(self.variable_names())].T
        return (seat_tube_length + 40) - saddle_height


class SeatPostTooShort(ValidationFunction):
    def friendly_name(self) -> str:
        return "Seat post too short"

    def variable_names(self) -> List[str]:
        return ["Seat tube length", "Seatpost LENGTH", "Saddle height"]

    def validate(self, designs: torch.Tensor) -> torch.Tensor:
        seat_tube_length, seatpost_length, saddle_height = designs[:, :len(self.variable_names())].T
        return saddle_height - (seat_tube_length + seatpost_length + 30) 


class BsdRearTooSmall(ValidationFunction):
    def friendly_name(self) -> str:
        return "Bsd rear too small"

    def variable_names(self) -> List[str]:
        return ["BSD rear", "ERD rear"]

    def validate(self, designs: torch.tensor) -> torch.tensor:
        bsd_rear, erd_rear = designs[:, :len(self.variable_names())].T
        return erd_rear - bsd_rear


class BsdFrontTooSmall(ValidationFunction):
    def friendly_name(self) -> str:
        return "Bsd front too small"

    def variable_names(self) -> List[str]:
        return ["BSD front", "ERD front"]

    def validate(self, designs: torch.tensor) -> torch.tensor:
        bsd_front, erd_front = designs[:, :len(self.variable_names())].T
        return erd_front - bsd_front


class HeadTubeLowerExtensionTooGreat(ValidationFunction):
    def friendly_name(self) -> str:
        return "Head tube lower extension too great"

    def variable_names(self) -> List[str]:
        return ["Head tube length textfield", "Head tube lower extension2"]

    def validate(self, designs: torch.tensor) -> torch.tensor:
        head_tube_length, head_tube_lower_extension = designs[:, :len(self.variable_names())].T
        return head_tube_lower_extension - head_tube_length


class HeadTubeLengthTooGreat(ValidationFunction):
    def friendly_name(self) -> str:
        return "Head tube length too great"

    def variable_names(self) -> List[str]:
        return ["Head tube length textfield", "Head tube upper extension2", "Head tube lower extension2"]

    def validate(self, designs: torch.tensor) -> torch.tensor:
        head_tube_length, head_tube_upper_extension, head_tube_lower_extension = designs[:,
                                                                                 :len(self.variable_names())].T
        return (head_tube_upper_extension + head_tube_lower_extension) - head_tube_length


class ChainStayLessThanZero(ValidationFunction):
    def friendly_name(self) -> str:
        return "Chain stay less than zero"

    def variable_names(self) -> List[str]:
        return ["CS textfield"]

    def validate(self, designs: torch.tensor) -> torch.tensor:
        cs_textfield, = designs[:, :len(self.variable_names())].T
        return 0 - cs_textfield


class ChainStayShouldBeGreaterThanWheelRadius(ValidationFunction):
    def friendly_name(self) -> str:
        return "Chain stay should be greater than wheel radius"

    def variable_names(self) -> List[str]:
        return ["CS textfield", "BSD rear"]

    def validate(self, designs: torch.tensor) -> torch.tensor:
        CS_textfield, BSD_rear = designs[:, :len(self.variable_names())].T
        return (BSD_rear/2) - CS_textfield
    

# class SeatStayShouldBeGreaterThanWheelRadius(ValidationFunction):
#     def friendly_name(self) -> str:
#         return "Seat stay should be greater than wheel radius"

#     def variable_names(self) -> List[str]:
#         return ["CS textfield", "BB textfield","Seat tube length", "Seat stay junction0", "Seat angle"]

#     def validate(self, designs: torch.tensor) -> torch.tensor:

#         CS_textfield, BB_textfield, Seat_tube_length, Seat_stay_junction0, Seat_angle = designs[:, :len(self.variable_names())].T
#         x = Seat_tube_length-(BB_textfield/numpy.sin(numpy.radians(Seat_angle)))-Seat_stay_junction0
#         y = BB_textfield/numpy.tan(numpy.radians(Seat_angle))
#         z = torch.sqrt((CS_textfield ** 2)-(BB_textfield ** 2))
#         h = z-y
#         g = torch.sqrt(h**2 + x**2 - 2*h*x*numpy.cos(numpy.radians(Seat_angle)))
#         return (674 / 2) - g
    

class SeatStayShouldBeGreaterThanWheelRadius(ValidationFunction):
    def friendly_name(self) -> str:
        return "Seat stay should be greater than wheel radius"

    def variable_names(self) -> List[str]:
        return ["CS textfield", "BB textfield","Seat tube length", "Seat stay junction0", "Seat angle"]

    def validate(self, designs: torch.tensor) -> torch.tensor:
        CS_textfield, BB_textfield, Seat_tube_length, Seat_stay_junction0, Seat_angle = designs[:, :len(self.variable_names())].T
        Seat_angle_rad = (Seat_angle * math.pi) / 180
        x = Seat_tube_length-(BB_textfield/torch.sin(Seat_angle_rad))-Seat_stay_junction0
        y = BB_textfield/torch.tan(Seat_angle_rad)
        z = torch.sqrt((CS_textfield ** 2)-(BB_textfield ** 2))
        h = z-y
        g = torch.sqrt(h**2 + x**2 - 2*h*x*torch.cos(Seat_angle_rad))
        return (674 / 2) - g
    
# class ThePedalShouldntIntersectTheFrontWheel(ValidationFunction):
#     def friendly_name(self) -> str:
#         return "The pedal shouldn't intersect the front wheel"

#     def variable_names(self) -> List[str]:
#         return ["Stack", "Head tube length textfield", "Head tube lower extension2", "Head angle", "BB textfield", "DT Length"]

#     def validate(self, designs: torch.tensor) -> torch.tensor:
#         Stack, Head_tube_length_textfield, Head_tube_lower_extension2, Head_angle, BB_textfield, DT_length = designs[:, :len(self.variable_names())].T
#         # Extract variables from the DataFrame
#         HTL = Head_tube_length_textfield
#         HTLX = Head_tube_lower_extension2
#         HTA = Head_angle * numpy.pi / 180  # Convert degrees to radians
#         BBD = BB_textfield
#         DTL = DT_length

#         # Calculate DTJY and DTJX
#         DTJY = Stack - (HTL - HTLX) * numpy.sin(HTA)
#         DTJX = numpy.sqrt(DTL ** 2 - DTJY ** 2)

#         # Calculate FWX and FCD
#         FWX = DTJX + (DTJY - BBD) / numpy.tan(HTA)
#         FCD = numpy.sqrt(FWX ** 2 + BBD ** 2)
#         wheel_radius = 674/2
#         crank_length = 172.5
#         return  (crank_length + wheel_radius)-FCD
    
class ThePedalShouldntIntersectTheFrontWheel(ValidationFunction):
    def friendly_name(self) -> str:
        return "The pedal shouldn't intersect the front wheel"

    def variable_names(self) -> List[str]:
        return ["Stack", "Head tube length textfield", "Head tube lower extension2", "Head angle", "BB textfield", "DT Length"]

    def validate(self, designs: torch.tensor) -> torch.tensor:
        Stack, Head_tube_length_textfield, Head_tube_lower_extension2, Head_angle, BB_textfield, DT_length = designs[:, :len(self.variable_names())].T
        # Extract variables from the DataFrame
        HTL = Head_tube_length_textfield
        HTLX = Head_tube_lower_extension2
        HTA = (Head_angle * math.pi) / 180  # Convert degrees to radians
        BBD = BB_textfield
        DTL = DT_length

        # Calculate DTJY and DTJX
        DTJY = Stack - (HTL - HTLX) * torch.sin(HTA)
        DTJX = torch.sqrt(DTL ** 2 - DTJY ** 2)

        # Calculate FWX and FCD
        FWX = DTJX + (DTJY - BBD) / torch.tan(HTA)
        FCD = torch.sqrt(FWX ** 2 + BBD ** 2)
        wheel_radius = 674/2
        crank_length = 172.5
        return  (crank_length + wheel_radius)-FCD
    
class TheCrankShouldntHitTheGroundWhenItIsInItsLowerPosition(ValidationFunction):
    def friendly_name(self) -> str:
        return "The crank shouldn't hit the ground when it is in its lower position"

    def variable_names(self) -> List[str]:
        return ["BB textfield"]

    def validate(self, designs: torch.tensor) -> torch.tensor:
        BB_textfield = designs[:, :len(self.variable_names())].T
        # Extract variables from the DataFrame
        wheel_radius = 674/2
        crank_length = 172.5
        return  (crank_length + BB_textfield) - wheel_radius


CLIPS_VALIDATIONS: List[ValidationFunction] = [
    SaddleHeightTooSmall(),
    SeatPostTooShort(),
    BsdRearTooSmall(),
    BsdFrontTooSmall(),
    HeadTubeLowerExtensionTooGreat(),
    HeadTubeLengthTooGreat(),
    ChainStayLessThanZero(),
    ChainStayShouldBeGreaterThanWheelRadius(),
    SeatStayShouldBeGreaterThanWheelRadius(),
    ThePedalShouldntIntersectTheFrontWheel(),
    TheCrankShouldntHitTheGroundWhenItIsInItsLowerPosition()
]