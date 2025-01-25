import unittest

import pandas as pd

from biked_commons.validation.biked_validator import BikedValidator
from utils_for_tests import path_of_test_resource


class ValidationsTest(unittest.TestCase):
    def setUp(self):
        self.validator = BikedValidator()
        with open(path_of_test_resource("clip_sBIKED_processed.csv"), "r") as file:
            self.df = pd.read_csv(file)

    def test_validate_clip(self):
        validation_result = self.validator.validate_clip(self.df)
        print(validation_result)
