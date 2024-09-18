import os.path


def test_resource_path(path_suffix: str):
    return os.path.join(os.path.dirname(__file__), "resources", path_suffix)
