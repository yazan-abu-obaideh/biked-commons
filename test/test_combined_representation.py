import unittest
from typing import Dict

import pandas as pd

from biked_commons.combined_representation.combined_representation import CombinedRepresentation, DatasetDescription, \
    Conversion, DEFAULT_STRICT_INTERSECTION, DEFAULT_KEEP_FIRST
from biked_commons.combined_representation.merging import DuplicateColumnRemovalStrategy, \
    DuplicateColumnRemovalStrategies, RowMergeStrategy, RowMergeStrategies

SHOULD_BE_CLEANED = "Description has not been cleaned. This could lead to memory issues."


class CombinedRepresentationTest(unittest.TestCase):

    def test_no_shared_keys(self):
        first = pd.DataFrame.from_records([{"A": 17} for _ in range(13)])
        second = pd.DataFrame.from_records([{"B": 19} for _ in range(13)])

        representation = self.build_representation({
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
        representation = self.build_representation({
            "first": first_description,
            "second": second_description
        })

        self.assertEqual(13, len(representation._merged_data))
        self.assertEqual({"A", "B", "C-in-master"}, set(representation._merged_data.columns))
        self.assertTrue((17 == representation._merged_data["A"]).all())
        self.assertTrue((19 == representation._merged_data["B"]).all())
        self.assertTrue((1.3 == representation._merged_data["C-in-master"]).all())
        self.assertEqual(["A", "C-in-first"], list(representation.get_as_representation("first").columns))
        self.assertEqual(["B", "C-in-second"], list(representation.get_as_representation("second").columns))

    def test_column_merge_strategy(self):
        descriptions = {
            "first": DatasetDescription.from_data(
                pd.DataFrame.from_records([{"A": 15, "B": 16} for _ in range(10)])),
            "second": DatasetDescription.from_data(
                pd.DataFrame.from_records([{"B": 15, "C": 16} for _ in range(10)])),
        }

        representation = self.build_representation(descriptions,
                                                   column_removal_strategy=DuplicateColumnRemovalStrategies.IGNORE
                                                   )
        merged = representation.get_data()
        self.assertEqual(len(merged), 10)
        self.assertEqual(["A", "B", "B", "C"], list(merged.columns))

        representation = self.build_representation(descriptions,
                                                   column_removal_strategy=DuplicateColumnRemovalStrategies.KEEP_FIRST
                                                   )
        merged = representation.get_data()
        self.assertEqual(len(merged), 10)
        self.assertEqual(["A", "B", "C"], list(merged.columns))

    def test_row_merge_strategy_no_intersection(self):
        first_data = pd.DataFrame.from_records(data=[{"A": 15} for _ in range(10)],
                                               index=[i for i in range(10, 20)])
        second_data = pd.DataFrame.from_records(data=[{"B": 15} for _ in range(10)],
                                                index=[i for i in range(10)])
        descriptions = {
            "first": DatasetDescription.from_data(first_data),
            "second": DatasetDescription.from_data(second_data),
        }
        representation = self.build_representation(descriptions,
                                                   row_merge_strategy=RowMergeStrategies.STRICT_INTERSECTION)
        self.assertEqual(len(representation.get_data()), 0)
        self.assertEqual(len(representation.get_as_representation("first")), 0)
        self.assertEqual(len(representation.get_as_representation("second")), 0)

    def test_row_merge_strategy_some_intersection(self):
        first_data = pd.DataFrame.from_records(data=[{"A": 15} for _ in range(10)],
                                               index=[i for i in range(10)])
        second_data = pd.DataFrame.from_records(data=[{"B": 15} for _ in range(10)],
                                                index=[i for i in range(3, 13)])
        third_data = pd.DataFrame.from_records(data=[{"C": 15} for _ in range(10)],
                                               index=[i for i in range(6, 16)])
        descriptions = {
            "first": DatasetDescription.from_data(first_data),
            "second": DatasetDescription.from_data(second_data),
            "third": DatasetDescription.from_data(third_data),
        }
        representation = self.build_representation(descriptions,
                                                   row_merge_strategy=RowMergeStrategies.STRICT_INTERSECTION)
        self.assertEqual(len(representation.get_data()), 4)
        self.assertEqual(set(representation.get_data().index), {6, 7, 8, 9})
        self.assertEqual(set(representation.get_as_representation("first").index), {6, 7, 8, 9})
        self.assertEqual(set(representation.get_as_representation("second").index), {6, 7, 8, 9})
        self.assertEqual(set(representation.get_as_representation("third").index), {6, 7, 8, 9})

    def build_representation(self, id_to_description: Dict[str, DatasetDescription],
                             column_removal_strategy: DuplicateColumnRemovalStrategy = DEFAULT_KEEP_FIRST,
                             row_merge_strategy: RowMergeStrategy = DEFAULT_STRICT_INTERSECTION
                             ) -> CombinedRepresentation:
        representation = CombinedRepresentation(id_to_description,
                                                column_removal_strategy=column_removal_strategy,
                                                row_merge_strategy=row_merge_strategy
                                                )
        for desc in representation._id_to_description.values():
            self.assertTrue(desc.has_been_cleaned(), SHOULD_BE_CLEANED)
        return representation
