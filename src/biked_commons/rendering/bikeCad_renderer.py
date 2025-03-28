import requests

from biked_commons.exceptions import InternalError
from biked_commons.resource_utils import STANDARD_BIKE_RESOURCE
from biked_commons.xml_handling.cad_builder import BikeCadFileBuilder

LOGGER_NAME = "BikeCadLogger"
WINDOWS = "Windows"


class RenderingService:
    def __init__(self, cad_builder: BikeCadFileBuilder = BikeCadFileBuilder()):
        self.cad_builder = cad_builder
        self._xml_transformer = BikeCadFileBuilder()

    def render_object(self, bike_object, seed_bike_xml: str):
        return self.render(self._xml_transformer.build_cad_from_biked(bike_object, seed_bike_xml))

    def render_clips(self, target_bike: dict, seed_bike_xml: str):
        return self.render(self._xml_transformer.build_cad_from_clips_object(target_bike, seed_bike_xml))

    def render(self, bike_xml: str):
        result = requests.post("http://localhost:8080/api/v1/render", data=bike_xml)
        if result.status_code == 200:
            return result.content
        raise InternalError(f"Rendering request failed {result}")

    def _read_standard_bike_xml(self, handler):
        with open(STANDARD_BIKE_RESOURCE) as file:
            handler.set_xml(file.read())
