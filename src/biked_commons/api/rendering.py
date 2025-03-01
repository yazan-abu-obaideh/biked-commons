import attrs

from biked_commons.rendering.bikeCad_renderer import RenderingService
from biked_commons.resource_utils import STANDARD_BIKE_RESOURCE


@attrs.define(frozen=True)
class RenderingResult:
    image_bytes: bytes


class SingleThreadedRenderer:
    """

    """

    def __init__(self,
                 renderer_timeout_seconds: int = 30,
                 timeout_granularity_seconds: int = 1):
        self.renderer = RenderingService(renderer_pool_size=1,
                                         renderer_timeout=renderer_timeout_seconds,
                                         timeout_granularity=timeout_granularity_seconds)
        with open(STANDARD_BIKE_RESOURCE, "r") as file:
            self.standard_bike_xml = file.read()

    def render_xml(self, bike_xml: str) -> RenderingResult:
        res = self.renderer.render(bike_xml)
        return RenderingResult(image_bytes=res)

    def render_biked(self, biked: dict) -> RenderingResult:
        return RenderingResult(image_bytes=self.renderer.render_object(bike_object=biked,
                                                                       seed_bike_xml=self.standard_bike_xml))

    def render_clip(self, clip: dict) -> RenderingResult:
        return RenderingResult(image_bytes=self.renderer.render_clips(target_bike=clip,
                                                                      seed_bike_xml=self.standard_bike_xml))
