import unittest

import numpy.testing as np_test
import pandas as pd

from biked_commons.combined_representation.conversions import RenameColumn, ReversibleConversion, ScaleColumn


class ReversibleConversionsTest(unittest.TestCase):
    def test_rename_column(self):
        data = pd.DataFrame.from_records([{"A": i} for i in range(10)])
        expected_converted = pd.DataFrame.from_records([{"B": i} for i in range(10)])
        conversion = RenameColumn("A", "B")
        self.assert_conversion(
            starting_data=data,
            conversion=conversion,
            expected_converted=expected_converted
        )

    def test_scale_column(self):
        data = pd.DataFrame.from_records([{"A": i} for i in range(10)])
        expected_converted = pd.DataFrame.from_records([{"A": i * 1000} for i in range(10)])
        conversion = ScaleColumn("A", 1000)
        self.assert_conversion(
            starting_data=data,
            expected_converted=expected_converted,
            conversion=conversion
        )

    def assert_conversion(self,
                          starting_data: pd.DataFrame,
                          conversion: ReversibleConversion,
                          expected_converted: pd.DataFrame):
        starting_copy = starting_data.copy(deep=True)
        actual_converted = conversion.apply(starting_data)

        self.assert_dataframes_equal(actual_converted, expected_converted)
        self.assert_original_not_modified(starting_copy, starting_data)
        converted_copy = actual_converted.copy(deep=True)
        self.assert_dataframes_equal(conversion.reverse(actual_converted), starting_copy)
        self.assert_original_not_modified(converted_copy, actual_converted)

    def assert_original_not_modified(self, starting_copy: pd.DataFrame, starting_data: pd.DataFrame):
        self.assert_dataframes_equal(starting_copy, starting_data)

    def assert_dataframes_equal(self,
                                first: pd.DataFrame,
                                second: pd.DataFrame):
        self.assertEqual(list(first.columns), list(second.columns))
        np_test.assert_array_equal(first.index.values, second.index.values)
        np_test.assert_array_equal(first.values, second.values)
