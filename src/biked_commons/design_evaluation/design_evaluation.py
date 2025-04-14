from abc import abstractmethod, ABC
from typing import List, Callable

from biked_commons import resource_utils
from biked_commons.bike_embedding import ordered_columns
from biked_commons.prediction.usability_predictors import UsabilityPredictorBinary, UsabilityPredictorContinuous
from biked_commons.usability import usability_ordered_columns

import pandas as pd
import torch


class EvaluationFunction(ABC):
    @abstractmethod
    def friendly_name(self) -> str:
        """
        Should return a user-friendly and easily comprehensible name for the validation in question.
        """
        pass

    @abstractmethod
    def variable_names(self) -> List[str]:
        """
        Should return a list of variable names used in the validation.
        """
        pass

    @abstractmethod
    def return_names(self) -> List[str]:
        """
        Should return the names of the various quantities calcualted in the evaluation.
        """
        pass

    @abstractmethod
    def evaluate(self, designs: torch.Tensor) -> torch.Tensor:
        """
        Should return a PyTorch tensor with shape (len(designs), 1) or (len(designs),).
        The values in the tensor represent validity. 1 is invalid, 0 is valid.
        """
        pass

class AeroEvaluator(EvaluationFunction):
    def __init__(self, device="cpu"):
        model_path = resource_utils.resource_path("models") + '/aero.pth'
        self.model = torch.load(model_path).to(device)
        self.device = device

    def friendly_name(self) -> str:
        return "Aerodynamics Evaluator"

    def variable_names(self) -> List[str]:
        return ['hand_x', 'hand_y', 'hip_x', 'hip_y', 'crank_length']
    
    def return_names(self) -> List[str]:
        return ['drag_force']

    def evaluate(self, designs: torch.Tensor, conditioning: dict) -> torch.Tensor:
        #rider dims must be of the form 'upper_leg', 'lower_leg', 'arm_length', 'torso_length', 'neck_and_head_length',

        rider_dims = conditioning["Rider"]
        rider_dims = rider_dims.expand(designs.shape[0], -1)
        predictions = self.model(designs)
        return predictions
    
class AestheticsEvaluator(EvaluationFunction):
    def __init__(self, mode = "Image", device="cpu"):
        model_path = resource_utils.resource_path("models") + '/clip.pth'
        self.model = torch.load(model_path).to(device)
        self.mode = mode #Image, Text, or Image Path
        self.device = device

    def friendly_name(self) -> str:
        return "Compare To Image Evaluator"

    def variable_names(self) -> List[str]:
        return ordered_columns.ORDERED_COLUMNS
    
    def return_names(self) -> List[str]:
        if self.mode in ["Image", "Image Path"]:
            return ['Cosine Similarity to Image']
        if self.mode == "Text":
            return ['Cosine Similarity to Text']

    def evaluate(self, designs: torch.Tensor, conditioning: dict) -> torch.Tensor:
        condition = conditioning[self.mode]
        if self.mode == "Image":
            ...
        elif self.mode == "Image Path":
            ...
        elif self.mode == "Text":
            ...
        predictions = self.model(designs)
        return predictions
    

class UsabilityEvaluator(EvaluationFunction):
    def __init__(self, target_type = 'cont'):
        self.target_type = target_type
        if target_type == 'cont':
            self.model = UsabilityPredictorContinuous()
        elif target_type == 'binary':
            self.model = UsabilityPredictorBinary()
        else:
            raise ValueError("Invalid target_type. Choose either 'cont' or 'binary'.")
        

    def friendly_name(self) -> str: 
        if self.target_type == 'cont':
            return "Usability Evaluator (Continuous)"
        elif self.target_type == 'binary':
            return "Usability Evaluator (Binary)"

    def variable_names(self) -> List[str]:
        return usability_ordered_columns.ORDERED_COLUMNS
    
    def return_names(self) -> List[str]:
        if self.target_type == 'cont':
            return ['Usability Score - 0 to 1']
        elif self.target_type == 'binary':
            return ['Usability Class - 0 or 1']

    def evaluate(self, designs: torch.Tensor, conditioning: dict = {}) -> torch.Tensor:
        if self.target_type == 'cont':
            return self.model.predict(designs)
        elif self.target_type == 'binary':
            x_input = designs.detach().cpu().numpy()
            predictions = self.model.predict(x_input)
            return torch.tensor(predictions, dtype=torch.float32, device=designs.device)
    

def construct_tensor_evaluator(evaluation_functions: List[EvaluationFunction], column_names: List[str]):

    column_names = list(column_names)

    def evaluate_tensor(designs: torch.Tensor, conditioning={}) -> torch.Tensor:

        n = designs.shape[0]
        v = len(evaluation_functions)

        # Initialize results tensor with zeros (default: valid)
        results_tensor = torch.zeros((n, v), dtype=torch.float32, device=designs.device)

        for i, evaluation_function in enumerate(evaluation_functions):
            # try:


            # Get the indices of the required variables for this function
            var_indices = [column_names.index(var) for var in evaluation_function.variable_names()]

            # Extract the relevant slices from the tensor
            sliced_designs = designs[:, var_indices]

            # Apply validation
            res = evaluation_function.validate(sliced_designs, conditioning)  # Expected to return a torch.Tensor

            # Store results
            results_tensor[:, i] = res.flatten()

        return results_tensor

    return evaluate_tensor



def construct_dataframe_evaluator(evaluation_functions: List[EvaluationFunction]):

    # First, construct the tensor-based validator (this one doesn't need column mapping)
    def evaluate_dataframe(designs: pd.DataFrame, conditioning={}) -> pd.DataFrame:

        # Convert DataFrame to a PyTorch tensor (float32)
        designs_tensor = torch.tensor(designs.to_numpy(), dtype=torch.float32)

        # Use the tensor validator (construct it dynamically based on DataFrame columns)
        tensor_validator = construct_tensor_evaluator(evaluation_functions, list(designs.columns))
        results_tensor = tensor_validator(designs_tensor, conditioning)

        # Convert results back to a DataFrame
        results_df = pd.DataFrame(
            results_tensor.numpy(),  # Convert tensor to NumPy
            columns=[vf.friendly_name() for vf in evaluation_functions],
            index=designs.index  # Preserve original index
        )

        return results_df

    return evaluate_dataframe
                                  

