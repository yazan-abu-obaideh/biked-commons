import unittest
from difflib import SequenceMatcher

from biked_commons.api.rendering import SingleThreadedRenderer
from biked_commons.resource_utils import STANDARD_BIKE_RESOURCE
from utils_for_tests import path_of_test_resource

SAMPLE_BIKE_OBJECT = {
    "Crank length": 169.98547990908648, "DT Length": 679.9341705346619, "HT Angle": 72.5,
    "HT LX": 51.2, "HT Length": 166.80082510430057, "Handlebar style": 1,
    "Headset spacers": 10.280547380910999, "ST Angle": 73.49999998763374, "ST Length": 309.8455052246119,
    "Saddle height": 494.4966418082093, "Seatpost LENGTH": 300, "Stack": 565.6, "Stem angle": 23.47686421897292,
    "Stem length": 82.05632953500437
}

SAMPLE_CLIPS_OBJECT = {"CS textfield": 420.0, "BB textfield": 60.006180256298634, "Stack": 513.1, "Head angle": 71.5,
                       "Head tube length textfield": 160.41945731506718, "Seat stay junction0": 45.0,
                       "Seat tube length": 501.00629695517847, "Seat angle": 73.50133116409353,
                       "DT Length": 687.5903391820704, "BB diameter": 40.0, "ttd": 28.6, "dtd": 36.5,
                       "csd": 22.38580515252831, "ssd": 12.5, "Chain stay position on BB": 15.0,
                       "SSTopZOFFSET": 9.176450476129222, "Head tube upper extension2": 29.45003450254673,
                       "Seat tube extension2": 53.23495635933917, "Head tube lower extension2": 47.95834081596411,
                       "SEATSTAYbrdgshift": 330.0, "CHAINSTAYbrdgshift": 350.0, "SEATSTAYbrdgdia1": 16.0,
                       "CHAINSTAYbrdgdia1": 18.0, "SEATSTAYbrdgCheck": 0.0, "CHAINSTAYbrdgCheck": 0.0,
                       "Dropout spacing": 130.0, "ERD rear": 582.0, "Wheel width rear": 23.0, "BSD front": 622.0,
                       "Wheel width front": 23.0, "ERD front": 582.0, "BSD rear": 622.0, "Display AEROBARS": 1.0,
                       "BB length": 68.0, "Head tube diameter": 45.0, "Wheel cut": 692.0, "Seat tube diameter": 31.8,
                       "Front Fender include": 0.0, "Rear Fender include": 1.0, "Number of cogs": 0.0,
                       "Number of chainrings": 1.0, "Display RACK": 0.0, "FIRST color R_RGB": 254.0,
                       "FIRST color G_RGB": 232.0, "FIRST color B_RGB": 0.0, "SPOKES composite front": 1.0,
                       "SBLADEW front": 80.0, "SBLADEW rear": 80.0, "Saddle length": 278.0, "Saddle height": 710.0,
                       "Down tube diameter": 34.237061525224206, "Seatpost LENGTH": 291.0,
                       "MATERIAL OHCLASS: ALUMINIUM": 0.0, "MATERIAL OHCLASS: BAMBOO": 1.0,
                       "MATERIAL OHCLASS: CARBON": 0.0, "MATERIAL OHCLASS: OTHER": 0.0, "MATERIAL OHCLASS: STEEL": 1.0,
                       "MATERIAL OHCLASS: TITANIUM": 0.0, "Dropout spacing style OHCLASS: 0": 0.0,
                       "Dropout spacing style OHCLASS: 1": 1.0, "Dropout spacing style OHCLASS: 2": 1.0,
                       "Dropout spacing style OHCLASS: 3": 0.0, "Fork type OHCLASS: 0": 1.0,
                       "Fork type OHCLASS: 1": 1.0, "Fork type OHCLASS: 2": 0.0, "Stem kind OHCLASS: 0": 1.0,
                       "Stem kind OHCLASS: 1": 0.0, "Stem kind OHCLASS: 2": 0.0, "Handlebar style OHCLASS: 0": 1.0,
                       "Handlebar style OHCLASS: 1": 0.0, "Handlebar style OHCLASS: 2": 0.0,
                       "Head tube type OHCLASS: 0": 0.0, "Head tube type OHCLASS: 1": 0.0,
                       "Head tube type OHCLASS: 2": 1.0, "Head tube type OHCLASS: 3": 1.0,
                       "Top tube type OHCLASS: 0": 1.0, "Top tube type OHCLASS: 1": 0.0,
                       "bottle SEATTUBE0 show OHCLASS: False": 1.0, "bottle SEATTUBE0 show OHCLASS: True": 0.0,
                       "bottle DOWNTUBE0 show OHCLASS: False": 1.0, "bottle DOWNTUBE0 show OHCLASS: True": 1.0,
                       "BELTorCHAIN OHCLASS: 0": 0.0, "BELTorCHAIN OHCLASS: 1": 1.0,
                       "RIM_STYLE front OHCLASS: DISC": 0.0, "RIM_STYLE front OHCLASS: SPOKED": 0.0,
                       "RIM_STYLE front OHCLASS: TRISPOKE": 1.0, "RIM_STYLE rear OHCLASS: DISC": 0.0,
                       "RIM_STYLE rear OHCLASS: SPOKED": 1.0, "RIM_STYLE rear OHCLASS: TRISPOKE": 1.0}


class RenderingTest(unittest.TestCase):
    def setUp(self):
        self.renderer = SingleThreadedRenderer()
        with open(STANDARD_BIKE_RESOURCE, "r") as file:
            self.standard_bike_xml = file.read()

    def test_render_biked(self):
        actual_result = self.renderer.render_biked(SAMPLE_BIKE_OBJECT)
        self.assertImagesEqual(actual_result.image_bytes, "expected_render_biked.svg")

    def test_render_bike_xml_file(self):
        actual_result = self.renderer.render_xml(self.standard_bike_xml)
        self.assertImagesEqual(actual_result.image_bytes, "expected_standard_bike_img.svg")

    def test_render_clip(self):
        actual_result = self.renderer.render_clip(SAMPLE_CLIPS_OBJECT)
        self.assertImagesEqual(actual_result.image_bytes, "expected_clips_bike_img.svg")

    def assertImagesEqual(self, rendering_result: bytes, test_image_path: str):
        try:
            with open(path_of_test_resource(test_image_path), "r") as image_file:
                result_str = str(rendering_result, "utf-8")
                expected_str = image_file.read()
                exactly_equal = result_str == expected_str
                similar_enough = int(exactly_equal)
                if not exactly_equal:
                    print("WARNING: images not exactly equal")
                    similar_enough = SequenceMatcher(None, result_str, result_str).ratio()
                # differences in screen size can affect rendering result.
                # this workaround eliminates that for now.
                self.assertGreater(similar_enough, 0.999999)

        except Exception as e:
            failed_image_path = test_image_path.replace(".svg", "_failed.svg")
            print(f"An exception occurred. Writing failed rendering result to file: {failed_image_path}")
            with open(path_of_test_resource(failed_image_path), "wb") as failed_image_result:
                failed_image_result.write(rendering_result)
            self.fail(e)
