from abc import abstractmethod, ABC
from typing import List
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import dill

from biked_commons.bike_embedding import ordered_columns, clip_embedding_calculator
from biked_commons.prediction.usability_predictors import UsabilityPredictorBinary, UsabilityPredictorContinuous
from biked_commons.usability import usability_ordered_columns
from biked_commons.transformation import interface_points, framed
from biked_commons.ergonomics import joint_angles
from biked_commons.prediction import aero_predictor, clip_predictor
from biked_commons.prediction.prediction_utils import Preprocessor
from biked_commons.resource_utils import models_and_scalers_path, split_datasets_path
from biked_commons.validation.base_validation_function import construct_tensor_validator
from biked_commons.validation.clip_validation_functions import CLIPS_VALIDATIONS





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

    @abstractmethod # 1 = objective, 0 = constraint
    def return_types(self) -> List[str]:
        pass

    @abstractmethod
    def evaluate(self, designs: torch.Tensor, conditioning: dict = {}) -> torch.Tensor:
        pass


class AeroEvaluator(EvaluationFunction):
    def __init__(self, device="cpu", dtype=torch.float32):
        super().__init__(device, dtype)
        model_path = models_and_scalers_path("aero_model.pt")
        scaler_path = models_and_scalers_path("aero_scaler.pt")
        self.model = torch.load(model_path).to(self.device)
        self.preprocessor = Preprocessor(scaler_path=scaler_path, preprocess_fn=aero_predictor.calculate_features, device=device)

    def variable_names(self) -> List[str]:
        return [
            "Stack",
            "Handlebar style OHCLASS: 0", "Handlebar style OHCLASS: 1", "Handlebar style OHCLASS: 2",
            "Seat angle", "Saddle height", "Head tube length textfield", "Head tube lower extension2",
            "Head angle", "DT Length"
        ]

    def return_names(self) -> List[str]:
        return ['Drag Force']
    
    def return_types(self) -> List[str]:
        return [1]

    def evaluate(self, designs: torch.Tensor, conditioning: dict = {}) -> torch.Tensor:
        int_pts = interface_points.calculate_interface_points(designs)
        assert "Rider" in conditioning, "Rider dimensions must be provided in conditioning to calculate aerodynamics."
        rider_dims = conditioning["Rider"]
        if rider_dims.shape[0] == 1:
            rider_dims = rider_dims.expand(designs.shape[0], -1)
        combinations = torch.cat((int_pts, rider_dims), dim=1)
        combinations = combinations.to(self.device, dtype=self.dtype)
        combinations = self.preprocessor(combinations)
        predictions = self.model(combinations)
        return predictions

class FrameValidityEvaluator(EvaluationFunction):
    def __init__(self, device="cpu", dtype=torch.float32):
        super().__init__(device, dtype)
        model_path = models_and_scalers_path("validity_model.pt")
        scaler_path = models_and_scalers_path("validity_scaler.pt")
        self.model = torch.load(model_path).to(self.device)
        self.preprocessor = Preprocessor(scaler_path=scaler_path, preprocess_fn=None, device=device)
        
        self.converter = framed.clip_to_framed_tensor_builder(ordered_columns.ORDERED_COLUMNS, framed.FRAMED_ORDERED_COLUMNS)

    def variable_names(self) -> List[str]:
        return ordered_columns.ORDERED_COLUMNS

    def return_names(self) -> List[str]:
        return ['Predicted Frame Validity']
    
    def return_types(self) -> List[str]:
        return [0]

    def evaluate(self, designs: torch.Tensor, conditioning: dict = {}) -> torch.Tensor:

        framed_tensor = self.converter(designs)
        framed_tensor = framed_tensor.to(self.device, dtype=self.dtype)
        framed_tensor = self.preprocessor(framed_tensor)
        predictions = self.model(framed_tensor)
        validity = predictions-0.5
        return validity
    
class StructuralEvaluator(EvaluationFunction):
    def __init__(self, device="cpu", dtype=torch.float32):
        super().__init__(device, dtype)
        model_path = models_and_scalers_path("structural_model.pt")
        scaler_path = models_and_scalers_path("structural_scaler.pt")
        self.model = torch.load(model_path).to(self.device)
        self.preprocessor = Preprocessor(scaler_path=scaler_path, preprocess_fn=None, device=device)
        
        self.converter = framed.clip_to_framed_tensor_builder(ordered_columns.ORDERED_COLUMNS, framed.FRAMED_ORDERED_COLUMNS)

    def variable_names(self) -> List[str]:
        return ordered_columns.ORDERED_COLUMNS

    def return_names(self) -> List[str]:
        return ['Mass', 'Planar Compliance', 'Transverse Compliance', 'Eccentric Compliance', 'Planar Safety Factor', 'Eccentric Safety Factor']

    def return_types(self) -> List[str]:
        return [1,1,1,1,0,0]

    def evaluate(self, designs: torch.Tensor, conditioning: dict = {}) -> torch.Tensor:
        framed_tensor = self.converter(designs)
        framed_tensor = framed_tensor.to(self.device, dtype=self.dtype)
        framed_tensor = self.preprocessor(framed_tensor)
        predictions = self.model(framed_tensor)
        predictions[:, 4:6] = 1.5 - predictions[:, 4:6]
        return predictions

class AestheticsEvaluator(EvaluationFunction):
    def __init__(self, mode="Image", device="cpu", dtype=torch.float32):
        super().__init__(device, dtype)
        model_path = models_and_scalers_path("clip_model.pt")
        scaler_path = models_and_scalers_path("clip_scaler.pt")
        self.preprocessor = Preprocessor(scaler_path=scaler_path, preprocess_fn=clip_predictor.remove_wall_thickness, device=device)
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
        
    def return_types(self) -> List[str]:
        return [1]

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
        designs = self.preprocessor(designs)
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

class ValidationEvaluator(EvaluationFunction):
    def __init__(self, device="cpu", dtype=torch.float32):
        super().__init__(device, dtype)
        self.clip_parameters = pd.read_csv(split_datasets_path("CLIP_X_test.csv"), index_col=0).columns.tolist() #TODO maybe include a list somewhere to avoid loading a dataset?
        validator, validation_names = construct_tensor_validator(CLIPS_VALIDATIONS, self.clip_parameters)
        self.validator = validator
        self.validation_names = validation_names

    def variable_names(self) -> List[str]:
        return self.clip_parameters

    def return_names(self) -> List[str]:
        return self.validation_names
    
    def return_types(self) -> List[str]:
        return [0] * len(self.validation_names)

    def evaluate(self, designs: torch.Tensor, conditioning: dict = {}) -> torch.Tensor:
        # designs = designs.to(self.device, dtype=self.dtype)
        predictions = self.validator(designs)
        return predictions

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
    
    def return_types(self) -> List[str]:
        return [1, 1, 1]

    def evaluate(self, designs: torch.Tensor, conditioning: dict = {}) -> torch.Tensor:
        assert "Rider" in conditioning, "Rider dimensions must be provided in conditioning to calculate ergonomics."
        rider_dims = conditioning["Rider"]
        if rider_dims.shape[0] == 1:
            rider_dims = rider_dims.expand(designs.shape[0], -1)

        assert "Use Case" in conditioning, "Use Case must be provided in conditioning to calculate ergonomics."
        use_case = conditioning["Use Case"]
        if use_case.ndim == 1:
            if use_case.shape != (3,):
                raise ValueError("If 1D, Use Case array must have shape (3,), got {}".format(use_case.shape))
            if not np.array_equal(use_case, use_case.astype(bool)):
                raise ValueError("Use Case 1D array must contain only 0s and 1s")
            if use_case.sum() != 1:
                raise ValueError("Use Case 1D array must be a valid one-hot vector (sum == 1)")
            # Broadcast to (n,3)
            use_case = np.tile(use_case, (designs.shape[0], 1))

        elif use_case.ndim == 2:
            n, k = use_case.shape
            if k != 3:
                raise ValueError("If 2D, Use Case array must have shape (n,3), got {}".format(use_case.shape))
            if n != designs.shape[0]:
                raise ValueError("Number of rows in Use Case (got {}) must match number of designs ({})"
                                .format(n, designs.shape[0]))
            # Check binary values and one-hot per row
            if not np.array_equal(use_case, use_case.astype(bool)):
                raise ValueError("Use Case 2D array must contain only 0s and 1s")
            row_sums = use_case.sum(axis=1)
            if not np.all(row_sums == 1):
                bad = np.where(row_sums != 1)[0]
                raise ValueError(f"Rows at indices {bad.tolist()} are not valid one-hot vectors")

        else:
            raise ValueError("Use Case array must be 1D or 2D, got {}-D".format(use_case.ndim))
        
        index_to_label = ["road", "mtb", "commute"]
        use_case_list = [index_to_label[idx] for idx in use_case.argmax(axis=1)]

        int_pts = interface_points.calculate_interface_points(designs)
        predictions = joint_angles.adjusted_nll(int_pts, rider_dims, use_case_list)
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
        
    def return_types(self) -> List[str]:
        if self.target_type == 'cont':
            return [1]
        elif self.target_type == 'binary':
            return [0]

    def evaluate(self, designs: torch.Tensor, conditioning: dict = {}) -> torch.Tensor:
        if self.target_type == 'cont':
            return self.model.predict(designs.to(self.device, dtype=self.dtype))
        elif self.target_type == 'binary':
            x_input = designs.detach().cpu().numpy()
            predictions = self.model.predict(x_input)
            predictions = predictions - 0.5 #TODO confirm that 0=valid
            return torch.tensor(predictions, dtype=self.dtype, device=self.device)

    
def construct_tensor_evaluator(evaluation_functions: List[EvaluationFunction], column_names: List[str]):

    column_names = list(column_names)

    # Flatten all return names across evaluators
    all_return_names = []
    all_return_types = []
    for vf in evaluation_functions:
        all_return_names.extend(vf.return_names())
        all_return_types.extend(vf.return_types())

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

    return evaluate_tensor, all_return_names, all_return_types

def construct_dataframe_evaluator(evaluation_functions: List[EvaluationFunction]):

    def evaluate_dataframe(designs: pd.DataFrame, conditioning={}) -> pd.DataFrame:
        designs_tensor = torch.tensor(designs.values, dtype=torch.float32)
        tensor_evaluator, return_names, return_types = construct_tensor_evaluator(evaluation_functions, list(designs.columns))
        results_tensor = tensor_evaluator(designs_tensor, conditioning)

        results_df = pd.DataFrame(
            results_tensor.detach().cpu().numpy(),
            columns=return_names,
            index=designs.index
        )

        return results_df, return_types

    return evaluate_dataframe


