import attrs

from biked_commons.rendering.BikeCAD_server_client import RenderingClient
from biked_commons.rendering.BikeCAD_server_manager import SingleThreadedBikeCadServerManager, \
    MultiThreadedBikeCadServerManager
from biked_commons.resource_utils import STANDARD_BIKE_RESOURCE


@attrs.define(frozen=True)
class RenderingResult:
    image_bytes: bytes


class RenderingEngine:
    def __init__(self, multi_threaded_rendering=False):

        with open(STANDARD_BIKE_RESOURCE, "r") as file:
            self.standard_bike_xml = file.read()
        if multi_threaded_rendering:
            self._rendering_client = RenderingClient(
                server_manager=MultiThreadedBikeCadServerManager(
                    number_servers=5,
                    timeout_seconds=75
                )
            )
        else:
            self._rendering_client = RenderingClient(
                server_manager=SingleThreadedBikeCadServerManager()
            )

    def render_xml(self, bike_xml: str) -> RenderingResult:
        res = self._rendering_client.render(bike_xml)
        return RenderingResult(image_bytes=res)

    def render_biked(self, biked: dict) -> RenderingResult:
        return RenderingResult(image_bytes=self._rendering_client.render_biked(target_bike=biked,
                                                                               seed_bike_xml=self.standard_bike_xml))

    def render_clip(self, clip: dict) -> RenderingResult:
        return RenderingResult(image_bytes=self._rendering_client.render_clips(target_bike=clip,
                                                                               seed_bike_xml=self.standard_bike_xml))
