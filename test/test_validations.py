import unittest

import pandas as pd

from biked_commons.api.validations import validate_raw_BikeCad
from biked_commons.resource_utils import resource_path
from biked_commons.validation.biked_validator import BikedValidator
from biked_commons.xml_handling.algebraic_parser import AlgebraicParser
from biked_commons.xml_handling.bike_xml_handler import BikeXmlHandler
from utils_for_tests import path_of_test_resource


class ValidationsTest(unittest.TestCase):
    def setUp(self):
        self.validator = BikedValidator()
        with open(path_of_test_resource("clip_sBIKED_processed.csv"), "r") as file:
            self.df = pd.read_csv(file)

    def test_validate_clip(self):
        validation_result = self.validator.validate_clip(self.df)
        print(validation_result)

    def test_validate_biked(self):
        data = self._generate_data()
        validation_results = validate_raw_BikeCad(data, verbose=True)

        failed = []
        self.assertEqual(0, len(failed), f"Some validations failed")
        incorrect_dimensions = []
        for index, result in enumerate(validation_results):
            if result.encountered_exception:
                failed.append(result)
            if result.per_design_result.shape != (10,):
                incorrect_dimensions.append(result)

        self.assertEqual(0, len(failed), f"Some validations failed")
        self.assertEqual(0, len(incorrect_dimensions), f"Some results have the wrong dimensions")

    def _generate_data(self):
        handler = BikeXmlHandler()
        with open(resource_path("PlainRoadBikeStandardized.txt"), "r") as file:
            handler.set_xml(file.read())
        data = pd.DataFrame.from_records([self._to_dict_data(handler) for _ in range(10)])
        return data

    def _to_dict_data(self, handler):
        res = {k: AlgebraicParser().attempt_parse(v) for k, v in handler.get_entries_dict().items()}
        res["DT Length"] = 600
        return res
