import unittest

import pandas as pd

from biked_commons.api.validations import validate_raw_BikeCad
from biked_commons.resource_utils import resource_path
from biked_commons.validation.biked_validator import BikedValidator
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
        handler = BikeXmlHandler()
        with open(resource_path("PlainRoadBikeStandardized.txt"), "r")  as file:
            handler.set_xml(file.read())
        data = pd.DataFrame.from_records([handler.get_entries_dict()])
        print(data)
        validation_results = validate_raw_BikeCad(data, verbose=True)
        for res in validation_results:
            print(res.encountered_exception)
