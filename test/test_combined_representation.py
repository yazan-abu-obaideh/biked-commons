import unittest

import pandas as pd

from biked_commons.combined_representation.combined_representation import CombinedRepresentation, DatasetDescription, \
    Conversion


class CombinedRepresentationTest(unittest.TestCase):

    def test_no_shared_keys(self):
        first = pd.DataFrame.from_records([{"A": 17} for _ in range(13)])
        second = pd.DataFrame.from_records([{"B": 19} for _ in range(13)])
        representation = CombinedRepresentation([DatasetDescription.from_data(data) for data in [first, second]])
        self.assertEqual(13, len(representation.data))
        self.assertTrue((17 == representation.data["A"]).all())
        self.assertTrue((19 == representation.data["B"]).all())

    def test_one_shared_key(self):
        first = pd.DataFrame.from_records([{"A": 17, "C-in-first": 13} for _ in range(13)])
        second = pd.DataFrame.from_records([{"B": 19, "C-in-second": 1301} for _ in range(13)])
        first_description = DatasetDescription(
            first, [
                Conversion(
                    name="C-in-first",
                    master_name="C-in-master",
                    conversion_function=lambda x: x
                )
            ]
        )
        second_description = DatasetDescription(second, conversions=[
            Conversion(
                name="C-in-second",
                master_name="C-in-master",
                conversion_function=lambda x: x / 1000
            )
        ])
        representation = CombinedRepresentation([first_description, second_description])

        print("***")
        print(representation.data)

        self.assertEqual(13, len(representation.data))
