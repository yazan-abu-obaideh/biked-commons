import unittest

import pandas as pd

from biked_commons.validation.biked_validator import BikedValidator


class ValidationsTest(unittest.TestCase):
    def setUp(self):
        self.validator = BikedValidator()

    def test_validate_clip(self):
        validation_result = self.validator.validate_clip(pd.DataFrame.from_records([{}]))
        print(validation_result)
