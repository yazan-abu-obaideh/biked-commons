from abc import abstractmethod, ABC
from typing import List

from biked_commons import resource_utils
from biked_commons.bike_embedding import ordered_columns
from biked_commons.prediction.usability_predictors import UsabilityPredictorBinary, UsabilityPredictorContinuous
from biked_commons.usability import usability_ordered_columns
from biked_commons.transformation import interface_points

import pandas as pd
import torch


class EvaluationFunction(ABC):
    def __init__(self, device="cpu", dtype=torch.float32):
        self.device = device
        self.dtype = dtype

    @abstractmethod
    def variable_names(self) -> List[str]:
        pass

    @abstractmethod
    def return_names(self) -> List[str]:
        pass

    @abstractmethod
    def evaluate(self, designs: torch.Tensor, conditioning: dict = {}) -> torch.Tensor:
        pass


class AeroEvaluator(EvaluationFunction):
    def __init__(self, device="cpu", dtype=torch.float32):
        super().__init__(device, dtype)
        model_path = resource_utils.resource_path("models") + '/aero.pth'
        self.model = torch.load(model_path).to(self.device)

    def variable_names(self) -> List[str]:
        return [
            "Stack",
            "Handlebar style OHCLASS: 0", "Handlebar style OHCLASS: 1", "Handlebar style OHCLASS: 2",
            "Seat angle", "Saddle height", "Head tube length textfield", "Head tube lower extension2",
            "Head angle", "DT Length"
        ]

    def return_names(self) -> List[str]:
        return ['Drag Force']

    def evaluate(self, designs: torch.Tensor, conditioning: dict = {}) -> torch.Tensor:
        int_pts = interface_points.calculate_interface_points(designs)
        assert "Rider" in conditioning, "Rider dimensions must be provided in conditioning to calculate aerodynamics."
        rider_dims = conditioning["Rider"]
        if rider_dims.shape[0] == 1:
            rider_dims = rider_dims.expand(designs.shape[0], -1)
        combinations = torch.cat((int_pts, rider_dims), dim=1)
        combinations = combinations.to(self.device, dtype=self.dtype)
        predictions = self.model(combinations)
        return predictions


class AestheticsEvaluator(EvaluationFunction):
    def __init__(self, mode="Image", device="cpu", dtype=torch.float32):
        super().__init__(device, dtype)
        model_path = resource_utils.resource_path("models") + '/clip.pth'
        self.model = torch.load(model_path).to(self.device)
        self.mode = mode  # Image, Text, or Image Path

    def variable_names(self) -> List[str]:
        return ordered_columns.ORDERED_COLUMNS

    def return_names(self) -> List[str]:
        if self.mode in ["Image", "Image Path"]:
            return ['Cosine Similarity to Image']
        elif self.mode == "Text":
            return ['Cosine Similarity to Text']

    def evaluate(self, designs: torch.Tensor, conditioning: dict = {}) -> torch.Tensor:
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
    def __init__(self, target_type='cont', device="cpu", dtype=torch.float32):
        super().__init__(device, dtype)
        self.target_type = target_type
        if target_type == 'cont':
            self.model = UsabilityPredictorContinuous()
        elif target_type == 'binary':
            self.model = UsabilityPredictorBinary()
        else:
            raise ValueError("Invalid target_type. Choose either 'cont' or 'binary'.")

    def variable_names(self) -> List[str]:
        return usability_ordered_columns.ORDERED_COLUMNS

    def return_names(self) -> List[str]:
        if self.target_type == 'cont':
            return ['Usability Score - 0 to 1']
        elif self.target_type == 'binary':
            return ['Usability Class - 0 or 1']

    def evaluate(self, designs: torch.Tensor, conditioning: dict = {}) -> torch.Tensor:
        if self.target_type == 'cont':
            return self.model.predict(designs.to(self.device, dtype=self.dtype))
        elif self.target_type == 'binary':
            x_input = designs.detach().cpu().numpy()
            predictions = self.model.predict(x_input)
            return torch.tensor(predictions, dtype=self.dtype, device=self.device)

    
def construct_tensor_evaluator(evaluation_functions: List[EvaluationFunction], column_names: List[str]):

    column_names = list(column_names)

    # Flatten all return names across evaluators
    all_return_names = []
    for vf in evaluation_functions:
        all_return_names.extend(vf.return_names())

    def evaluate_tensor(designs: torch.Tensor, conditioning={}) -> torch.Tensor:
        n = designs.shape[0]
        total_outputs = sum(len(vf.return_names()) for vf in evaluation_functions)
        results_tensor = torch.zeros((n, total_outputs), dtype=torch.float32, device=designs.device)

        current_col = 0
        for vf in evaluation_functions:
            var_indices = [column_names.index(var) for var in vf.variable_names()]
            sliced_designs = designs[:, var_indices]

            res = vf.evaluate(sliced_designs, conditioning)  # Expect shape (n,) or (n, k)

            if res.dim() == 1:
                res = res.unsqueeze(1)

            num_outputs = res.shape[1]
            results_tensor[:, current_col:current_col + num_outputs] = res
            current_col += num_outputs

        return results_tensor

    return evaluate_tensor, all_return_names

def construct_dataframe_evaluator(evaluation_functions: List[EvaluationFunction]):

    def evaluate_dataframe(designs: pd.DataFrame, conditioning={}) -> pd.DataFrame:
        designs_tensor = torch.tensor(designs.values, dtype=torch.float32)
        tensor_evaluator, return_names = construct_tensor_evaluator(evaluation_functions, list(designs.columns))
        results_tensor = tensor_evaluator(designs_tensor, conditioning)

        results_df = pd.DataFrame(
            results_tensor.detach().cpu().numpy(),
            columns=return_names,
            index=designs.index
        )

        return results_df

    return evaluate_dataframe


