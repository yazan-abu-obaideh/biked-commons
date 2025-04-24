import torch
import pandas as pd
import numpy as np
import pygmo as pg
import os
from biked_commons.conditioning import conditioning
from biked_commons.resource_utils import split_datasets_path
from biked_commons.design_evaluation.design_evaluation import construct_tensor_evaluator, StandardEvaluations


def compute_ref_point(ref_scores):
    ref_scores[np.isnan(ref_scores)] = -float("inf")
    ref_point = np.max(ref_scores, axis=0)
    return ref_point

def recompute_ref_point():
    data = pd.read_csv(split_datasets_path("CLIP_X_test.csv"), index_col=0)
    num_data = data.shape[0]
    rider_condition = conditioning.sample_riders(num_data, split="test")
    use_case_condition = conditioning.sample_use_case(num_data, split="test")
    text_condition = conditioning.sample_text(num_data, split="test")

    condition = {"Rider": rider_condition, "Use Case": use_case_condition, "Text": text_condition}

    evaluator, requirement_names, requirement_types = construct_tensor_evaluator(StandardEvaluations, data.columns)
    isobjective = torch.tensor(requirement_types) == 1
    scores = evaluator(torch.tensor(data.values, dtype=torch.float32), condition)
    objective_scores = scores[:, isobjective]
    constraint_scores = scores[:, ~isobjective]
    objective_scores = scores[:, isobjective].detach().numpy()
    constraint_scores = scores[:, ~isobjective].detach().numpy()
    ref_point = compute_ref_point(objective_scores)
    np.save("HV_ref_point.npy", ref_point)

def hypervolume(objective_scores, constraint_scores):
    #if ref_point exists, use it, otherwise compute it
    ref_point_dir = "HV_ref_point.npy"
    if os.path.exists(ref_point_dir):
        ref_point = np.load(ref_point_dir)
    else:
        ref_point = recompute_ref_point()
        ref_point = np.load(ref_point_dir)
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

def constraint_satisfaction_rate(objective_scores, constraint_scores):
    return np.mean(constraint_scores <=0)

