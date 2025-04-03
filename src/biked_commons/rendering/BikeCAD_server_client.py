import atexit
import os
import subprocess
import time

import psutil
import requests

from biked_commons.exceptions import InternalError, check_internal_precondition
from biked_commons.resource_utils import STANDARD_BIKE_RESOURCE, resource_path
from biked_commons.xml_handling.cad_builder import BikeCadFileBuilder

SERVER_START_TIMEOUT_SECONDS = int(os.getenv("RENDERING_SERVER_START_TIMEOUT_SECONDS", 60))


def get_java_binary():
    b = os.getenv("JAVA_HOME", "java")
    if b.endswith("java"):
        res = b
    else:
        res = os.path.join(b, "bin", "java")
    print(f"Using {res} as the Java binary")
    return res


JAVA_BINARY = get_java_binary()


class RenderingClient:
    SERVER_PORT = 8080

    def __init__(self, cad_builder: BikeCadFileBuilder = BikeCadFileBuilder()):
        self.cad_builder = cad_builder
        self._xml_transformer = BikeCadFileBuilder()
        self._server_pid = None
        self._start_server()
        atexit.register(self._kill_server)

    def _start_server(self):
        if not self.check_server_health():
            print("Starting BikeCAD server...")
            process = subprocess.Popen(
                [JAVA_BINARY, "-jar", resource_path("BikeCAD-server.jar"), f"--server.port={self.SERVER_PORT}"])
            self._server_pid = process.pid
            seconds_waited = 0
            while not self.check_server_health():
                time.sleep(1)
                seconds_waited += 1
                if seconds_waited > SERVER_START_TIMEOUT_SECONDS:
                    raise InternalError("Could not start server...")
            print("BikeCAD server started.")

    def _kill_server(self):
        if self._server_pid and psutil.pid_exists(self._server_pid):
            print("BikeCAD Server exists. Killing...")
            psutil.Process(pid=self._server_pid).kill()
            print("BikeCAD server killed successfully.")

    def check_server_health(self) -> bool:
        try:
            health_response = requests.get(self._endpoint("/actuator/serverInformation"), timeout=1)
        except Exception as ignored:
            return False
        if health_response.status_code != 200:
            return False
        return health_response.json()["serverName"] == "BikeCAD-server"

    def render_object(self, bike_object, seed_bike_xml: str):
        return self.render(self._xml_transformer.build_cad_from_biked(bike_object, seed_bike_xml))

    def render_clips(self, target_bike: dict, seed_bike_xml: str):
        return self.render(self._xml_transformer.build_cad_from_clips_object(target_bike, seed_bike_xml))

    def render(self, bike_xml: str):
        result = requests.post(self._endpoint("/api/v1/render"), data=bike_xml)
        if result.status_code == 200:
            return result.content
        raise InternalError(f"Rendering request failed {result}")

    def _endpoint(self, suffix: str):
        url = f"http://localhost:{self.SERVER_PORT}{suffix}"
        check_internal_precondition(suffix.startswith("/"), f"Invalid url {url}")
        return url

    def _read_standard_bike_xml(self, handler):
        with open(STANDARD_BIKE_RESOURCE) as file:
            handler.set_xml(file.read())


RENDERING_CLIENT_INSTANCE = RenderingClient()
