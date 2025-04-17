import os


def resource_path(rel_path: str):
    return os.path.join(os.path.dirname(__file__), "..", "resources", rel_path)

def models_and_scalers_path(rel_path: str):
    return os.path.join(os.path.dirname(__file__), "..", "resources/models_and_scalers", rel_path)

STANDARD_BIKE_RESOURCE = resource_path("PlainRoadBikeStandardized.txt")
