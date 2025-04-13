import json
import unittest

from biked_commons.api.rendering import RenderingEngine, DEFAULT_RIDER
from biked_commons.resource_utils import STANDARD_BIKE_RESOURCE
from utils_for_tests import path_of_test_resource


def load_samples() -> dict:
    with open(path_of_test_resource("sample_objects.json"), "r") as file:
        return json.loads(file.read())


SAMPLES = load_samples()
SAMPLE_BIKE_OBJECT = SAMPLES["SAMPLE_BIKE_OBJECT"]
SAMPLE_CLIPS_OBJECT = SAMPLES["SAMPLE_CLIPS_OBJECT"]


class RenderingTest(unittest.TestCase):
    def setUp(self):
        self.renderer = RenderingEngine(1, 60)
        with open(STANDARD_BIKE_RESOURCE, "r") as file:
            self.standard_bike_xml = file.read()

    def test_render_biked(self):
        actual_result = self.renderer.render_biked(SAMPLE_BIKE_OBJECT)
        self.assertIsNotNone(actual_result.image_bytes)

    def test_render_with_rider(self):
        self.assertIsNotNone(self.renderer.render_biked(SAMPLE_BIKE_OBJECT, DEFAULT_RIDER))
        self.assertIsNotNone(self.renderer.render_clip(SAMPLE_CLIPS_OBJECT, DEFAULT_RIDER))

    def test_render_bike_xml_file(self):
        actual_result = self.renderer.render_xml(self.standard_bike_xml)
        self.assertIsNotNone(actual_result.image_bytes)

    def test_render_clip(self):
        actual_result = self.renderer.render_clip(SAMPLE_CLIPS_OBJECT)
        self.assertIsNotNone(actual_result.image_bytes)
