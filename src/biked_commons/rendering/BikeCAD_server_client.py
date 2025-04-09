import requests

from biked_commons.exceptions import InternalError
from biked_commons.rendering.BikeCAD_server_manager import SingleThreadedBikeCadServerManager, ServerManager
from biked_commons.resource_utils import STANDARD_BIKE_RESOURCE
from biked_commons.xml_handling.cad_builder import BikeCadFileBuilder


class RenderingClient:

    def __init__(self,
                 server_manager: ServerManager):
        self._xml_transformer = BikeCadFileBuilder()
        self._server_manager = server_manager

    def render_biked(self, target_bike: dict, seed_bike_xml: str) -> bytes:
        return self.render(self._xml_transformer.build_cad_from_biked(target_bike, seed_bike_xml))

    def render_clips(self, target_bike: dict, seed_bike_xml: str) -> bytes:
        return self.render(self._xml_transformer.build_cad_from_clips_object(target_bike, seed_bike_xml))

    def render(self, bike_xml: str):
        endpoint = self._server_manager.endpoint("/api/v1/render")
        result = requests.post(endpoint, data=bike_xml)
        if result.status_code == 200:
            return result.content
        raise InternalError(f"Rendering request failed {result}")

    def _read_standard_bike_xml(self, handler):
        with open(STANDARD_BIKE_RESOURCE) as file:
            handler.set_xml(file.read())
