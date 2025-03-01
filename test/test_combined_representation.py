import unittest

import pandas as pd

from biked_commons.combined_representation.combined_representation import CombinedRepresentation, DatasetDescription, \
    Conversion


class CombinedRepresentationTest(unittest.TestCase):

    def test_no_shared_keys(self):
        first = pd.DataFrame.from_records([{"A": 17} for _ in range(13)])
        second = pd.DataFrame.from_records([{"B": 19} for _ in range(13)])
        representation = CombinedRepresentation({
            "first": DatasetDescription.from_data(first),
            "second": DatasetDescription.from_data(second)
        })
        self.assertEqual(13, len(representation._merged_data))
        self.assertEqual({"A", "B"}, set(representation.get_data().columns))
        self.assertTrue((17 == representation._merged_data["A"]).all())
        self.assertTrue((19 == representation._merged_data["B"]).all())

    def test_one_shared_key(self):
        first = pd.DataFrame.from_records([{"A": 17, "C-in-first": 13} for _ in range(13)])
        second = pd.DataFrame.from_records([{"B": 19, "C-in-second": 1301} for _ in range(13)])
        first_description = DatasetDescription(
            first, [Conversion(
                name="C-in-first",
                master_name="C-in-master",
                conversion_function=lambda x: x / 10,
                inverse_conversion_function=lambda x: x * 10
            )
            ]
        )
        second_description = DatasetDescription(second, conversions=[
            Conversion(
                name="C-in-second",
                master_name="C-in-master",
                conversion_function=lambda x: x,
                inverse_conversion_function=lambda x: x
            )
        ])
        representation = CombinedRepresentation(
            {"first": first_description,
             "second": second_description
             }
        )

        self.assertEqual(13, len(representation._merged_data))
        self.assertEqual({"A", "B", "C-in-master"}, set(representation._merged_data.columns))
        self.assertTrue((17 == representation._merged_data["A"]).all())
        self.assertTrue((19 == representation._merged_data["B"]).all())
        self.assertTrue((1.3 == representation._merged_data["C-in-master"]).all())

    def test_get_as_representation(self):
        first = pd.DataFrame.from_records([{"A": 17, "C-in-first": 13} for _ in range(13)])
        second = pd.DataFrame.from_records([{"B": 19, "C-in-second": 1301} for _ in range(13)])
        first_description = DatasetDescription(
            first, [
                Conversion(
                    name="C-in-first",
                    master_name="C-in-master",
                    conversion_function=lambda x: x / 10,
                    inverse_conversion_function=lambda x: x * 10
                )
            ]
        )
        second_description = DatasetDescription(second, conversions=[
            Conversion(
                name="C-in-second",
                master_name="C-in-master",
                conversion_function=lambda x: x,
                inverse_conversion_function=lambda x: x
            )
        ])
        representation = CombinedRepresentation({
            "first": first_description,
            "second": second_description
        })

        self.assertEqual({"A", "C-in-first"}, set(representation.get_as_representation("first").columns))
        self.assertEqual({"B", "C-in-second"}, set(representation.get_as_representation("second").columns))
