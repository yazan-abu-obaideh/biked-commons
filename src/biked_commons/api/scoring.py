import numpy as np

from biked_commons.bike_embedding.clip_embedding_calculator import ClipEmbeddingCalculatorImpl

embedding_calculator = ClipEmbeddingCalculatorImpl()


def get_ergonomic_score():
    pass


def get_aerodynamic_score():
    pass


def embedding_from_text(bike_description: str) -> np.ndarray:
    return embedding_calculator.from_text(bike_description)


def embedding_from_image_path(image_path: str) -> np.ndarray:
    return embedding_calculator.from_image_path(image_path)
