import concurrent
import io
import unittest
from concurrent.futures import ThreadPoolExecutor, Future

import pandas as pd

from biked_commons.api.rendering import RenderingEngine, RenderingResult
from biked_commons.bike_embedding.clip_embedding_calculator import ClipEmbeddingCalculatorImpl
from biked_commons.resource_utils import resource_path
import PIL.Image


class ClipEmbeddingGenerationScriptTest(unittest.TestCase):
    def test_gen_script(self):
        n_servers = 1
        clip_data = pd.read_csv(resource_path("datasets/split_datasets/CLIP_X_test.csv"), index_col=0)
        records = clip_data.sample(10).to_dict(orient="records")
        executor = ThreadPoolExecutor(max_workers=n_servers)
        rendering_engine = RenderingEngine(number_rendering_servers=n_servers, server_init_timeout_seconds=60)
        future_results = []

        embedding_calculator = ClipEmbeddingCalculatorImpl()

        def render_record(clip_record: dict):
            rendering_result = rendering_engine.render_clip(clip_record)
            bytes_stream = io.BytesIO(rendering_result.image_bytes)
            image = PIL.Image.open(bytes_stream)
            # embedding_tensor = embedding_calculator.from_image(image)

        for record in records:
            future_result: Future[RenderingResult] = executor.submit(render_record, record)
            future_results.append(future_result)
        concurrent.futures.wait(future_results)
