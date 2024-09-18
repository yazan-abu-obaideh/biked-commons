import os


def resource_path(rel_path: str):
    return os.path.join(os.path.dirname(__file__), "resources", rel_path)


STANDARD_BIKE_RESOURCE = "PlainRoadBikeStandardized.txt"
