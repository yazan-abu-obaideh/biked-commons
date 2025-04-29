import os
import sys
sys.path.append("../../")
print("Current working directory:", os.getcwd())
from biked_commons.resource_utils import split_datasets_path
from biked_commons.conditioning import conditioning
from biked_commons.design_evaluation.design_evaluation import *

# from biked_commons.design_evaluation.scoring import *
import argparse
import torch
import numpy as np
from matplotlib import pyplot as plt

from libmoon.solver.gradient import GradAggSolver
from libmoon.util_global.constant import problem_dict
from libmoon.util_global.weight_factor.funs import uniform_pref
from libmoon.visulization.view_res import vedio_res

from libmoon.problem.mop import mop