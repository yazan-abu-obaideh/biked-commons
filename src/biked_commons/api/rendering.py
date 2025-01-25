from biked_commons.rendering.bikeCad_renderer import RenderingService


class RenderingResult:
    pass


class SingleThreadedRenderer:
    """

    """

    def __init__(self,
                 renderer_timeout_seconds: int = 30,
                 timeout_granularity_seconds: int = 1):
        self.renderer = RenderingService(renderer_pool_size=1,
                                         renderer_timeout=renderer_timeout_seconds,
                                         timeout_granularity=timeout_granularity_seconds)

    def render_from_xml(self, bike_xml: str) -> RenderingResult:
        res = self.renderer.render(bike_xml)
        return res
