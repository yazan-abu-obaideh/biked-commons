import attrs

from biked_commons.rendering.BikeCAD_server_client import RenderingClient
from biked_commons.rendering.BikeCAD_server_manager import SingleThreadedBikeCadServerManager, \
    MultiThreadedBikeCadServerManager
from biked_commons.resource_utils import STANDARD_BIKE_RESOURCE
from biked_commons.xml_handling.cad_builder import BikeCadFileBuilder

FILE_BUILDER = BikeCadFileBuilder()


@attrs.define(frozen=True)
class RenderingResult:
    image_bytes: bytes


class RenderingEngine:
    def __init__(self,
                 number_rendering_servers: int,
                 timeout_seconds: int
                 ):
        with open(STANDARD_BIKE_RESOURCE, "r") as file:
            self.standard_bike_xml = file.read()
        self._rendering_client = self._init_rendering_client(number_rendering_servers, timeout_seconds)

    def render_xml(self, bike_xml: str) -> RenderingResult:
        return RenderingResult(image_bytes=(self._render(bike_xml)))

    def render_biked(self, biked: dict, show_rider=False) -> RenderingResult:
        xml = FILE_BUILDER.build_cad_from_biked(biked, self.standard_bike_xml, show_rider)
        return RenderingResult(image_bytes=(self._render(xml)))

    def render_clip(self, clip: dict, show_rider=False) -> RenderingResult:
        xml = FILE_BUILDER.build_cad_from_clip(clip, self.standard_bike_xml, show_rider)
        return RenderingResult(image_bytes=(self._render(xml)))

    def _render(self, xml: str) -> bytes:
        return self._rendering_client.render(xml)

    def _init_rendering_client(self,
                               number_rendering_servers: int,
                               timeout_seconds: int
                               ):
        if number_rendering_servers > 1:
            manager = MultiThreadedBikeCadServerManager(number_servers=number_rendering_servers,
                                                        timeout_seconds=timeout_seconds)
            return RenderingClient(server_manager=manager)
        else:
            return RenderingClient(server_manager=SingleThreadedBikeCadServerManager(timeout_seconds))
