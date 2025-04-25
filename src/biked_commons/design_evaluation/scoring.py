from abc import abstractmethod, ABC
from typing import List
import torch
import pandas as pd
import numpy as np
import pygmo as pg
from sklearn.preprocessing import StandardScaler
import os
from biked_commons.conditioning import conditioning
from biked_commons.resource_utils import split_datasets_path
from biked_commons.design_evaluation.design_evaluation import construct_tensor_evaluator, StandardEvaluations, EvaluationFunction


class ScoringFunction(ABC):
    def __init__(self, device="cpu", dtype=torch.float32):
        self.device = device
        self.dtype = dtype

    @abstractmethod
    def return_names(self) -> List[str]:
        pass

    @abstractmethod
    def evaluate(self, designs: torch.Tensor, conditioning: dict = {}) -> torch.Tensor:
        pass

def compute_ref_point(ref_scores):
    ref_scores[np.isnan(ref_scores)] = -float("inf")
    ref_point = np.max(ref_scores, axis=0)
    return ref_point

def recompute_ref_point(evaluator, objective_names, isobjective):
    print("Calculating reference point for scoring functions...")
    data = pd.read_csv(split_datasets_path("bike_bench.csv"), index_col=0)
    num_data = data.shape[0]
    rider_condition = conditioning.sample_riders(num_data, split="test")
    use_case_condition = conditioning.sample_use_case(num_data, split="test")
    text_condition = conditioning.sample_text(num_data, split="test")

    condition = {"Rider": rider_condition, "Use Case": use_case_condition, "Text": text_condition}

    scores = evaluator(torch.tensor(data.values, dtype=torch.float32), condition)
    objective_scores = scores[:, isobjective]
    objective_scores = scores[:, isobjective].detach().numpy()
    ref_point = compute_ref_point(objective_scores)
    df = pd.Series(ref_point, index=objective_names)
    df.to_csv("obj_ref_point.csv")
    return ref_point

def get_ref_point(evaluator, objective_names, isobjective):
    if not os.path.exists("obj_ref_point.csv"):
        ref_point = recompute_ref_point(evaluator, objective_names, isobjective)
    else:
        ref_point = pd.read_csv("obj_ref_point.csv", index_col=0).values.flatten()
        if len(ref_point) != len(objective_names):
            ref_point = recompute_ref_point(evaluator, objective_names, isobjective)
        elif not np.all(np.isin(objective_names, ref_point)):
            ref_point = recompute_ref_point(evaluator, objective_names, isobjective)
        elif not np.all(np.isin(ref_point, objective_names)):
            ref_point = recompute_ref_point(evaluator, objective_names, isobjective)
    return ref_point

class Hypervolume(ScoringFunction):
    def __init__(self):
        super().__init__()

    def return_names(self) -> List[str]:
        return ["Hypervolume"]

    def evaluate(self, designs, objective_scores, constraint_scores, objective_names, constraint_names, ref_point):
        #if ref_point exists, use it, otherwise compute it
        
        validity_mask = np.all(constraint_scores <= 0, axis=1)
        valid_objective_scores = objective_scores[validity_mask]
        if valid_objective_scores.size == 0:
            return 0.0
        valid_objective_scores[np.isnan(valid_objective_scores)] = float("inf")
        valid_objective_scores = valid_objective_scores/ref_point
        valid_objective_scores = np.clip(valid_objective_scores, a_min=0, a_max=1)
        scaled_ref_point = np.ones_like(ref_point)

        hv = pg.hypervolume(valid_objective_scores)
        hv_value = hv.compute(ref_point=scaled_ref_point)
        return hv_value

class ConstraintSatisfactionRate(ScoringFunction):
    def __init__(self):
        super().__init__()

    def return_names(self) -> List[str]:
        return ["Constraint Satisfaction Rate"]
    
    def evaluate(self, designs, objective_scores, constraint_scores, objective_names, constraint_names, ref_point):
        return np.mean(constraint_scores <=0)


class MMD(ScoringFunction): 

    def __init__(self, gamma=None):
        super().__init__()
        raw_ref  = pd.read_csv(split_datasets_path("bike_bench.csv"), index_col=0).values.astype(np.float32)
        
        self.scaler = StandardScaler()
        self.scaler.fit(raw_ref)
        self.reference_designs = self.scaler.transform(raw_ref)

        if gamma is None:
            gamma = self.compute_gamma(self.reference_designs)
        self.gamma = gamma

    def return_names(self) -> List[str]:
        return ["Maximum Mean Discrepancy"]

    def compute_gamma(self, ref: np.ndarray) -> float:
        dists = np.sum((ref[:, None, :] - ref[None, :, :])**2, axis=2)
        med = np.median(dists)
        return 1.0 / (2 * med) if med > 0 else 1.0

    def rbf_kernel(self, A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
        dists = np.sum((A[:, None, :] - B[None, :, :])**2, axis=2)
        return np.exp(-gamma * dists)

    def mmd(self, gen: np.ndarray, ref: np.ndarray) -> float:
        K_GG = self.rbf_kernel(gen, gen, self.gamma)
        K_RR = self.rbf_kernel(ref, ref, self.gamma)
        K_GR = self.rbf_kernel(gen, ref, self.gamma)

        n, m = gen.shape[0], ref.shape[0]
        return (K_GG.sum() / (n * n)
              + K_RR.sum() / (m * m)
              - 2 * K_GR.sum() / (n * m))

    def evaluate(self, designs, objective_scores, constraint_scores, objective_names, constraint_names, ref_point):
        scaled_designs = self.scaler.transform(designs)
        return self.mmd(scaled_designs, self.reference_designs)
    

class MinimumObjective(ScoringFunction):
    def __init__(self):
        super().__init__()

    def return_names(self) -> List[str]:
        return self.names

    def evaluate(self, designs, objective_scores, constraint_scores, objective_names, constraint_names, ref_point):
        self.names = [f"Min Objective Score: {name}" for name in objective_names]
        validity_mask = np.all(constraint_scores <= 0, axis=1)
        valid_objective_scores = objective_scores[validity_mask]
        if valid_objective_scores.size == 0:
            return np.ones_like(objective_scores[0]) * ref_point
        valid_subset = valid_objective_scores[:, validity_mask]
        minscores = np.min(valid_subset, axis=0)
        return minscores
    
class MeanObjective(ScoringFunction):
    def __init__(self):
        super().__init__()

    def return_names(self) -> List[str]:
        return self.names

    def evaluate(self, designs, objective_scores, constraint_scores, objective_names, constraint_names, ref_point):
        self.names = [f"Mean Objective Score: {name}" for name in objective_names]
        validity_mask = np.all(constraint_scores <= 0, axis=1)
        valid_objective_scores = objective_scores[validity_mask]
        if valid_objective_scores.size == 0:
            return np.ones_like(objective_scores[0]) * ref_point
        valid_subset = valid_objective_scores[:, validity_mask]
        meanscores = np.mean(valid_subset, axis=0)
        return meanscores
    
class ConstraintViolationRate(ScoringFunction):
    def __init__(self):
        super().__init__()

    def return_names(self) -> List[str]:
        return self.names

    def evaluate(self, designs, objective_scores, constraint_scores, objective_names, constraint_names, ref_point):
        self.names = [f"Constraint Violation Rate: {name}" for name in constraint_names]
        validity_boolean = constraint_scores > 0
        return np.mean(validity_boolean, axis=0)

def construct_scorer(scoring_functions: List[ScoringFunction], evaluation_functions: List[EvaluationFunction], column_names: List[str]):

    evaluator, requirement_names, requirement_types = construct_tensor_evaluator(evaluation_functions, column_names)
    requirement_names = np.array(requirement_names)
    isobjective = torch.tensor(requirement_types) == 1
    objective_names = requirement_names[isobjective]
    constraint_names = requirement_names[~isobjective]

    ref_point = get_ref_point(evaluator, objective_names, isobjective) #1D numpy array

    def scorer(designs: torch.Tensor, condition: dict = {}) -> pd.Series:
        score_names = []
        scores = []
        evaluation_scores = evaluator(designs, condition)
        objective_scores = evaluation_scores[:, isobjective].detach().numpy()
        ref_point_exp = np.expand_dims(ref_point, axis=0)
        ref_point_exp = np.repeat(ref_point_exp, objective_scores.shape[0], axis=0)
        objective_scores[np.isnan(objective_scores)] = ref_point_exp[np.isnan(objective_scores)]
        constraint_scores = evaluation_scores[:, ~isobjective].detach().numpy()
        for scoring_function in scoring_functions:
            raw = scoring_function.evaluate(designs, objective_scores, constraint_scores, objective_names, constraint_names, ref_point)

            arr = np.atleast_1d(raw)

            names = scoring_function.return_names()

            for n, val in zip(names, arr):
                score_names.append(n)
                scores.append(val)
        scores = np.array(scores)
        scores = pd.Series(scores, index=score_names)
        return scores
    return scorer

MainScores: List[ScoringFunction] = [
    Hypervolume(),
    ConstraintSatisfactionRate(),
    MMD(),
]

DetailedScores: List[ScoringFunction] = [
    MinimumObjective(),
    MeanObjective(),
    ConstraintViolationRate(),
]



