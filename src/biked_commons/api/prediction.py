import numpy as np
import pandas as pd

from biked_commons.bike_embedding.embedding_predictor import EmbeddingPredictor

predictor = EmbeddingPredictor()


def embeddings_from_bike(designs: pd.DataFrame) -> np.ndarray:
    return predictor.predict(designs)
