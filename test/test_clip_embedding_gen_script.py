import concurrent.futures
import unittest
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import PIL.Image
import cairosvg
import numpy as np
import pandas as pd

from biked_commons.api.rendering import RenderingEngine
from biked_commons.bike_embedding.clip_embedding_calculator import ClipEmbeddingCalculatorImpl
from biked_commons.resource_utils import resource_path

CHECKPOINT_SIZE = 100


def to_tensor(image):
    raise Exception("Unimplemented function...")


def generate_embeddings(number_rendering_servers: int):
    clip_data = pd.read_csv(resource_path("datasets/split_datasets/CLIP_X_test.csv"), index_col=0)
    records = clip_data.sample(12).to_dict(orient="records")
    executor = ThreadPoolExecutor(max_workers=number_rendering_servers * 2)
    rendering_engine = RenderingEngine(number_rendering_servers=number_rendering_servers,
                                       server_init_timeout_seconds=90)
    future_results = []
    embedding_calculator = ClipEmbeddingCalculatorImpl()

    def render_record(clip_record: dict):
        rendering_result = rendering_engine.render_clip(clip_record)
        print("Rendering result received from server...")
        png_data = cairosvg.svg2png(rendering_result.image_bytes)
        image = PIL.Image.open(BytesIO(png_data))
        print("Image loaded...")
        # image_tensor = to_tensor(image)
        # augmented = get_augmented_views_gpu(image_tensor)
        embedding_tensor = embedding_calculator.from_image(image)
        print("Embedding tensor obtained...")
        return embedding_tensor

    for record in records:
        future_results.append(executor.submit(render_record, record))

    count = 0
    numpy_result_array = np.ndarray(shape=(1, 512))
    for result in concurrent.futures.as_completed(future_results):
        latest_result = result.result().detach().numpy().reshape((1, 512))
        numpy_result_array = np.concat([numpy_result_array, latest_result])
        count += 1
        if count % CHECKPOINT_SIZE == 0:
            check_point_csv = f"./checkpoint_{count}.csv"
            pd.DataFrame(numpy_result_array).to_csv(check_point_csv)
            print(f"Check point reached, saved to csv {check_point_csv}")

    data_frame = pd.DataFrame(numpy_result_array)
    data_frame.to_csv("./clip_embeddings.csv")


class ClipEmbeddingGenerationScriptTest(unittest.TestCase):
    def test_gen_script(self):
        generate_embeddings(1)
