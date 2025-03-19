import unittest

from biked_commons.combined_representation.combined_bike_representations import COMBINED


class CombinedBikeRepresentationsTest(unittest.TestCase):
    def test_passes(self):
        data = COMBINED.get_data()
        print(data)
