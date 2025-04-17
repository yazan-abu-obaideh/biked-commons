from abc import abstractmethod, ABC
from typing import List
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import dill

from biked_commons import resource_utils
from biked_commons.bike_embedding import ordered_columns, clip_embedding_calculator, embedding_predictor
from biked_commons.prediction.usability_predictors import UsabilityPredictorBinary, UsabilityPredictorContinuous
from biked_commons.usability import usability_ordered_columns
from biked_commons.transformation import interface_points
from biked_commons.ergonomics import joint_angles




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
        model_path = resource_utils.resource_path("models") + '/clip.pt'
        self.scaler = embedding_predictor._get_pickled_scaler()
        self.model = torch.load(model_path).to(self.device)
        self.mode = mode  # Image, Text, or Image Path
        self.embedding_model = clip_embedding_calculator.ClipEmbeddingCalculatorImpl()

    def variable_names(self) -> List[str]:
        return ordered_columns.ORDERED_COLUMNS

    def return_names(self) -> List[str]:
        if self.mode in ["Image", "Image Path"]:
            return ['Cosine Similarity to Image']
        elif self.mode == "Text":
            return ['Cosine Similarity to Text']

    def evaluate(self, designs: torch.Tensor, conditioning: dict = {}) -> torch.Tensor:
        cond = conditioning.get(self.mode)
        if cond is None:
            raise ValueError(f"No conditioning provided for mode '{self.mode}'")

        def is_singleton_list(x):
            return isinstance(x, (list, tuple)) and len(x) == 1

        if self.mode == "Image":
            if isinstance(cond, torch.Tensor):
                cond_list = [cond]
            elif isinstance(cond, list):
                cond_list = cond
            else:
                raise TypeError("For Image mode, conditioning must be a Tensor or list of Tensors")
        elif self.mode == "Image Path":
            if isinstance(cond, str):
                cond_list = [cond]
            elif isinstance(cond, (list, tuple)):
                cond_list = list(cond)
            else:
                raise TypeError("For Image Path mode, conditioning must be a path or list of paths")
        elif self.mode == "Text":
            if isinstance(cond, str):
                cond_list = [cond]
            elif isinstance(cond, (list, tuple)):
                cond_list = list(cond)
            else:
                raise TypeError("For Text mode, conditioning must be text or list of texts")
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        if is_singleton_list(cond_list):
            single = cond_list[0]
            if self.mode == "Image":
                img = single.unsqueeze(0) if single.dim() == 3 else single
                embed = self.embedding_model.from_image_tensor(img)
            elif self.mode == "Image Path":
                embed = self.embedding_model.from_image_path([single])
            else:
                embed = self.embedding_model.from_text([single])
        else:
            embeds = []
            for item in cond_list:
                if self.mode == "Image":
                    img = item.unsqueeze(0) if item.dim() == 3 else item
                    em = self.embedding_model.from_image_tensor(img)
                elif self.mode == "Image Path":
                    em = self.embedding_model.from_image_path([item])
                else:
                    em = self.embedding_model.from_text([item])
                embeds.append(em.squeeze(0))
            embed = torch.stack(embeds, dim=0)

        designs = self.scaler(designs)
        preds = self.model(designs)
        N = preds.size(0)

        if embed.dim() == 1:
            embed = embed.unsqueeze(0)

        B_cond = embed.size(0)
        if B_cond == 1 and N > 1:
            embed = embed.expand(N, -1)
        elif B_cond != N:
            raise ValueError(f"Number of condition embeddings ({B_cond}) does not match number of designs ({N})")

        cos_sim = F.cosine_similarity(preds, embed, dim=1)
        return (1 - cos_sim) / 2



class ErgonomicsEvaluator(EvaluationFunction):
    def __init__(self, device="cpu", dtype=torch.float32):
        super().__init__(device, dtype)
    def variable_names(self) -> List[str]:
        return [
            "Stack",
            "Handlebar style OHCLASS: 0", "Handlebar style OHCLASS: 1", "Handlebar style OHCLASS: 2",
            "Seat angle", "Saddle height", "Head tube length textfield", "Head tube lower extension2",
            "Head angle", "DT Length"
        ]

    def return_names(self) -> List[str]:
        return ['Knee Angle Error', 'Hip Angle Error', "Arm Angle Error"]

    def evaluate(self, designs: torch.Tensor, conditioning: dict = {}) -> torch.Tensor:
        assert "Rider" in conditioning, "Rider dimensions must be provided in conditioning to calculate ergonomics."
        rider_dims = conditioning["Rider"]
        if rider_dims.shape[0] == 1:
            rider_dims = rider_dims.expand(designs.shape[0], -1)

        assert "Use Case" in conditioning, "Use Case must be provided in conditioning to calculate ergonomics."
        use_case = conditioning["Use Case"]
        
        allowed_use_cases = {"road", "mtb", "commute"}
        if isinstance(use_case, str):
            if use_case not in allowed_use_cases:
                raise ValueError("Invalid use case. Choose either 'road', 'mtb', or 'commute'.")
            use_case = [use_case] * designs.shape[0]
        elif isinstance(use_case, (list, np.ndarray)):
            if len(use_case) != designs.shape[0]:
                raise ValueError("Length of use case list must match number of designs.")
            if not all(u in allowed_use_cases for u in use_case):
                raise ValueError("Invalid use case in list. All entries must be 'road', 'mtb', or 'commute'.")
        else:
            raise TypeError("Use Case must be a string or a list/array of strings.")

        int_pts = interface_points.calculate_interface_points(designs)
        predictions = joint_angles.adjusted_nll(int_pts, rider_dims, use_case)
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


