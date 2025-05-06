import os
import torch
from biked_commons.design_evaluation.scoring import construct_scorer, MainScores, DetailedScores
from biked_commons.design_evaluation.design_evaluation import get_standard_evaluations
from biked_commons.conditioning import conditioning

def get_condition(idx=0):
    rider_condition = conditioning.sample_riders(10, split="test")
    use_case_condition = conditioning.sample_use_case(10, split="test")
    image_embeddings = conditioning.sample_image_embedding(10, split="test")
    condition = {"Rider": rider_condition[idx], "Use Case": use_case_condition[idx], "Embedding": image_embeddings[idx]}
    return condition

def evaluate_uncond(result_tens, name, cond_idx, data_columns, device):

    condition = get_condition(cond_idx)

    result_dir = os.path.join("results", "unconditional", f"cond_{cond_idx}", name)
    os.makedirs(result_dir, exist_ok=True)
    
    main_scorer = construct_scorer(MainScores, get_standard_evaluations(device), data_columns)
    detailed_scorer = construct_scorer(DetailedScores, get_standard_evaluations(device), data_columns)

    main_scores = main_scorer(result_tens, condition)
    detailed_scores = detailed_scorer(result_tens, condition)

    # Save result_tens as .pt
    result_tens = result_tens.cpu()
    torch.save(result_tens, os.path.join(result_dir, "result_tens.pt"))

    main_scores.to_csv(os.path.join(result_dir, "main_scores.csv"), index_label=False, header=False)
    detailed_scores.to_csv(os.path.join(result_dir, "detailed_scores.csv"), index_label=False, header=False)