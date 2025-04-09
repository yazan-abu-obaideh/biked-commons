import os

import requests

from biked_commons.exceptions import InternalError
from biked_commons.rendering.BikeCAD_server_manager import SingleThreadedBikeCadServerManager
from biked_commons.resource_utils import STANDARD_BIKE_RESOURCE
from biked_commons.xml_handling.cad_builder import BikeCadFileBuilder

SERVER_START_TIMEOUT_SECONDS = int(os.getenv("RENDERING_SERVER_START_TIMEOUT_SECONDS", 60))


def get_java_binary():
    b = os.getenv("JAVA_HOME", "java")
    if b.endswith("java"):
        res = b
    else:
        res = os.path.join(b, "bin", "java")
    print(f"Using {res} as the Java binary")
    return res


JAVA_BINARY = get_java_binary()


class RenderingClient:

    def __init__(self, cad_builder: BikeCadFileBuilder = BikeCadFileBuilder()):
        self.cad_builder = cad_builder
        self._xml_transformer = BikeCadFileBuilder()
        self._server_manager = SingleThreadedBikeCadServerManager()

    def render_object(self, bike_object, seed_bike_xml: str):
        return self.render(self._xml_transformer.build_cad_from_biked(bike_object, seed_bike_xml))

    def render_clips(self, target_bike: dict, seed_bike_xml: str):
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


RENDERING_CLIENT_INSTANCE = RenderingClient()
