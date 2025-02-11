import unittest
from typing import List

import pandas as pd

from biked_commons.api.validations import validate_raw_BikeCad, validate_clip
from biked_commons.resource_utils import resource_path
from biked_commons.validation.validation_result import ValidationResult
from biked_commons.xml_handling.algebraic_parser import AlgebraicParser
from biked_commons.xml_handling.bike_xml_handler import BikeXmlHandler
from utils_for_tests import path_of_test_resource


class ValidationsTest(unittest.TestCase):
    def setUp(self):
        with open(path_of_test_resource("clip_sBIKED_processed.csv"), "r") as file:
            self.df = pd.read_csv(file)

    def test_validate_clip(self):
        validation_results = validate_clip(self.df)
        data_length = len(self.df)
        self.assertCleanResults(data_length, validation_results)

    def test_validate_biked(self):
        data_length = 31
        data = self._generate_data(data_length)
        validation_results = validate_raw_BikeCad(data)
        self.assertCleanResults(data_length, validation_results)

    def test_exceptional_cases_have_correct_dimensions(self):
        data_length = 31
        results = validate_raw_BikeCad(pd.DataFrame.from_records({"FIELD": 15} for _ in range(data_length)))
        not_exceptional = []
        incorrect_dimensions = []
        for res in results:
            if res.encountered_exception is False:
                not_exceptional.append(res.validation_name)
            if res.per_design_result.shape != (data_length, 1):
                incorrect_dimensions.append((res.validation_name, res.per_design_result.shape))

        self.assertEqual(0, len(not_exceptional),
                         f"Some validations did not encounter exceptions {not_exceptional}")
        self.assertEqual(0, len(incorrect_dimensions),
                         f"Some results have the wrong dimensions: {incorrect_dimensions}")

    def assertCleanResults(self, data_length: int, validation_results: List[ValidationResult]):
        encountered_exceptions = []
        incorrect_dimensions = []
        for index, result in enumerate(validation_results):
            if result.encountered_exception:
                encountered_exceptions.append(result.validation_name)
            if result.per_design_result.shape != (data_length, 1):
                incorrect_dimensions.append((result.validation_name, result.per_design_result.shape))
        self.assertEqual(0, len(encountered_exceptions),
                         f"Some validations encountered exceptions: {encountered_exceptions}")
        self.assertEqual(0, len(incorrect_dimensions),
                         f"Some results have the wrong dimensions: {incorrect_dimensions}")

    def _generate_data(self, data_length):
        handler = BikeXmlHandler()
        with open(resource_path("PlainRoadBikeStandardized.txt"), "r") as file:
            handler.set_xml(file.read())
        data = pd.DataFrame.from_records([self._to_dict_data(handler) for _ in range(data_length)])
        return data

    def _to_dict_data(self, handler):
        res = {k: AlgebraicParser().attempt_parse(v) for k, v in handler.get_entries_dict().items()}
        res["DT Length"] = 600
        return res
