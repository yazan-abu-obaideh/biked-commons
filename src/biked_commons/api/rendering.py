import attrs

from biked_commons.rendering.BikeCAD_server_client import RENDERING_CLIENT_INSTANCE
from biked_commons.resource_utils import STANDARD_BIKE_RESOURCE


@attrs.define(frozen=True)
class RenderingResult:
    image_bytes: bytes


class SingleThreadedRenderer:
    def __init__(self):
        with open(STANDARD_BIKE_RESOURCE, "r") as file:
            self.standard_bike_xml = file.read()

    def render_xml(self, bike_xml: str) -> RenderingResult:
        res = RENDERING_CLIENT_INSTANCE.render(bike_xml)
        return RenderingResult(image_bytes=res)

    def render_biked(self, biked: dict) -> RenderingResult:
        return RenderingResult(image_bytes=RENDERING_CLIENT_INSTANCE.render_object(bike_object=biked,
                                                                                   seed_bike_xml=self.standard_bike_xml))

    def render_clip(self, clip: dict) -> RenderingResult:
        return RenderingResult(image_bytes=RENDERING_CLIENT_INSTANCE.render_clips(target_bike=clip,
                                                                                  seed_bike_xml=self.standard_bike_xml))
