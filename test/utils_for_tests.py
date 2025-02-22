import os.path


def path_of_test_resource(path_suffix: str):
    return os.path.join(os.path.dirname(__file__), "resources", path_suffix)
