from typing import List

import pandas as pd
import torch
import math

# from biked_commons.validation.base_validation_function import ValidationFunction
from base_validation_function import ValidationFunction

import torch
from typing import List

__EPS__ = 1e-6

def law_of_cos(a, b, c):
    return (a**2 + b**2 - c**2) / (2 * a * b)

class SaddleTooFarFromPedals(ValidationFunction):
    def friendly_name(self) -> str:
        return "Saddle too far from pedals"

    def variable_names(self) -> List[str]:
        return ["upper_leg", "lower_leg", "hip_x", "hip_y", "crank_length"]

    def validate(self, designs: torch.Tensor) -> torch.Tensor:
        ul, ll, hip_x, hip_y, crank_length = designs[:, :len(self.variable_names())].T
        Lspl = torch.sqrt(hip_x**2 + hip_y**2) + crank_length
        return Lspl - (ul + ll)
    
class SaddleTooFarFromHandles(ValidationFunction):
    def friendly_name(self) -> str:
        return "Saddle too far from handles"

    def variable_names(self) -> List[str]:
        return ["arm_length", "torso_length", "hip_x", "hip_y", "hand_x", "hand_y"]

    def validate(self, designs: torch.Tensor) -> torch.Tensor:
        arm, back, hip_x, hip_y, hand_x, hand_y = designs[:, :len(self.variable_names())].T
        Lsh = torch.sqrt((hip_x + hand_x) ** 2 + (hip_y - hand_y) ** 2)
        return Lsh - (arm + back)  

class SaddleTooCloseToPedals(ValidationFunction):
    def friendly_name(self) -> str:
        return "Saddle too close to pedals"

    def variable_names(self) -> List[str]:
        return ["lower_leg", "upper_leg", "hip_x", "hip_y", "crank_length"]

    def validate(self, designs: torch.Tensor) -> torch.Tensor:
        lower_leg, upper_leg, hip_x, hip_y, crank_length = designs[:, :len(self.variable_names())].T
        min_saddle_to_pedal = torch.sqrt(hip_x**2 + hip_y**2) - crank_length
        min_hip_to_foot = torch.abs(lower_leg - upper_leg)
        return min_hip_to_foot - min_saddle_to_pedal

class HipAngleUnderNDegrees(ValidationFunction):
    def __init__(self, n: float = 10):
        self.n = n  # Store the user-defined threshold

    def friendly_name(self) -> str:
        return f"Hip angle under {self.n} degrees"

    def variable_names(self) -> List[str]:
        return ["hip_x", "hip_y", "hand_x", "hand_y", "torso_length", "arm_length", "crank_length", "lower_leg", "upper_leg"]

    def validate(self, designs: torch.Tensor) -> torch.Tensor:
        hip_x, hip_y, hand_x, hand_y, torso_length, arm_length, crank_length, lower_leg, upper_leg = designs[:, :len(self.variable_names())].T
        rad2deg = 180 / math.pi

        L_saddle_to_handle = torch.sqrt((hip_x + hand_x) ** 2 + (hip_y - hand_y) ** 2)

        # Calculate angle of handle -> saddle -> horizontal 
        denom = hip_x + hand_x
        Tsh = torch.atan2((hand_y - hip_y), (hip_x + hand_x)) * rad2deg

        # Calculate angle of pedal -> saddle -> horizontal (should be negative)
        Tsp = torch.atan2(-hip_y, hip_x) * rad2deg

        # Calculate angle from head -> saddle -> head
        cos_Thsh = law_of_cos(torso_length, L_saddle_to_handle, arm_length)
        cos_Thsh = torch.clamp(cos_Thsh, -1, 1)  # Ensure valid input for acos
        Tssh = torch.acos(cos_Thsh) * rad2deg

        # Calculate angle from knee -> saddle -> pedal
        min_saddle_to_pedal = torch.sqrt(hip_x**2 + hip_y**2) - crank_length
        cos_Tksp = law_of_cos(lower_leg, min_saddle_to_pedal, upper_leg)
        cos_Tksp = torch.clamp(cos_Tksp, -1, 1)  # Ensure valid input for acos
        Tksp = torch.acos(cos_Tksp) * rad2deg

        result = self.n - (Tsh + Tssh) - (Tsp + Tksp)

        # If the cosine value would be greater than 1 before taking arccos, set a fallback option 
        # that pushes the cosine to be less than 1 (so gradients still give guidance)
        cos_Thsh_invalid = cos_Thsh > 1
        cos_Tksp_invalid = cos_Tksp > 1

        # Use torch.where() to smoothly replace invalid values
        result = torch.where(cos_Tksp_invalid, cos_Tksp - 1, result)
        result = torch.where(cos_Thsh_invalid, cos_Thsh - 1, result)

        return result

class KneeAngleUnderNDegrees(ValidationFunction):
    def __init__(self, n: float = 20):
        self.n = n

    def friendly_name(self) -> str:
        return f"Knee angle under {self.n} degrees"

    def variable_names(self) -> List[str]:
        return ["upper_leg", "lower_leg", "hip_x", "hip_y", "crank_length"]

    def validate(self, designs: torch.Tensor) -> torch.Tensor:
        ul, ll, hip_x, hip_y, crank_length = designs[:, :len(self.variable_names())].T
        rad2deg = 180 / math.pi

        min_saddle_to_pedal = torch.sqrt(hip_x**2 + hip_y**2) - crank_length

        # Calculate angle from saddle -> knee -> pedal
        cos_Tskp = law_of_cos(ul, ll, min_saddle_to_pedal)
        
        # Ensure cos_Tskp is within valid range for acos
        cos_Tskp = torch.clamp(cos_Tskp, -1, 1)
        Tskp = torch.acos(cos_Tskp) * rad2deg

        # If the cosine value would be greater than 1 before taking arccos, set a fallback option 
        # that pushes the cosine to be less than 1 (so gradients still give guidance)
        cos_Tskp_invalid = cos_Tskp > 1
        result = torch.where(cos_Tskp_invalid, cos_Tskp - 1, Tskp - self.n)

        min_saddle_to_pedal_invalid = min_saddle_to_pedal <=0
        result = torch.where(min_saddle_to_pedal_invalid, __EPS__-min_saddle_to_pedal, result)

        return result



AERO_VALIDATIONS: List[ValidationFunction] = [
    SaddleTooFarFromPedals(),
    SaddleTooFarFromHandles(),
    SaddleTooCloseToPedals(),
    HipAngleUnderNDegrees(10),
    KneeAngleUnderNDegrees(20),
]
