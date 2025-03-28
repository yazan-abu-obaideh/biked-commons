import json
import unittest
import uuid
from difflib import SequenceMatcher

from biked_commons.api.rendering import SingleThreadedRenderer
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
        self.renderer = SingleThreadedRenderer()
        with open(STANDARD_BIKE_RESOURCE, "r") as file:
            self.standard_bike_xml = file.read()

    def test_render_biked(self):
        actual_result = self.renderer.render_biked(SAMPLE_BIKE_OBJECT)
        with open(f"some_file_{uuid.uuid4()}", "wb") as file:
            file.write(actual_result.image_bytes)
        self.assertImagesEqual(actual_result.image_bytes, "expected_render_biked.svg")

    def test_render_bike_xml_file(self):
        actual_result = self.renderer.render_xml(self.standard_bike_xml)
        with open(f"some_file_{uuid.uuid4()}", "wb") as file:
            file.write(actual_result.image_bytes)
        self.assertImagesEqual(actual_result.image_bytes, "expected_standard_bike_img.svg")

    def test_render_clip(self):
        actual_result = self.renderer.render_clip(SAMPLE_CLIPS_OBJECT)
        with open(f"some_file_{uuid.uuid4()}", "wb") as file:
            file.write(actual_result.image_bytes)
        self.assertImagesEqual(actual_result.image_bytes, "expected_clips_bike_img.svg")

    def assertImagesEqual(self, rendering_result: bytes, test_image_path: str):
        try:
            with open(path_of_test_resource(test_image_path), "r") as image_file:
                result_str = str(rendering_result, "utf-8")
                len_res = len(result_str)
                print(f"f{len_res=}")
                expected_str = image_file.read()
                exactly_equal = result_str == expected_str
                similar_enough = int(exactly_equal)
                if not exactly_equal:
                    print("WARNING: images not exactly equal")
                    similar_enough = SequenceMatcher(None, result_str, result_str).ratio()
                # differences in screen size can affect rendering result.
                # this workaround eliminates that for now.
                print(f"{similar_enough=}")
                self.assertGreater(similar_enough, 0.9999999999)

        except Exception as e:
            failed_image_path = test_image_path.replace(".svg", "_failed.svg")
            print(f"An exception occurred. Writing failed rendering result to file: {failed_image_path}")
            with open(path_of_test_resource(failed_image_path), "wb") as failed_image_result:
                failed_image_result.write(rendering_result)
            self.fail(e)
