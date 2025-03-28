import atexit
import os
import subprocess
import time

import psutil
import requests

from biked_commons.exceptions import InternalError
from biked_commons.resource_utils import STANDARD_BIKE_RESOURCE, resource_path
from biked_commons.xml_handling.cad_builder import BikeCadFileBuilder

JAVA_BINARY = os.getenv("JAVA_PATH", "java")


class RenderingClient:
    def __init__(self, cad_builder: BikeCadFileBuilder = BikeCadFileBuilder()):
        self.cad_builder = cad_builder
        self._xml_transformer = BikeCadFileBuilder()
        self._server_pid = None
        self._start_server()
        atexit.register(self._kill_server)

    def _start_server(self):
        if not self.check_server_health():
            process = subprocess.Popen([JAVA_BINARY, "-jar", resource_path("BikeCAD-server.jar")])
            self._server_pid = process.pid
            wait_count = 0
            while not self.check_server_health():
                time.sleep(1)
                wait_count += 1
                if wait_count > 10:
                    raise InternalError("Could not start server...")

    def _kill_server(self):
        if self._server_pid and psutil.pid_exists(self._server_pid):
            print("Server exists. Killing...")
            psutil.Process(pid=self._server_pid).kill()

    def check_server_health(self) -> bool:
        try:
            health_response = requests.get("http://localhost:8080/actuator/serverInformation")
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
        result = requests.post("http://localhost:8080/api/v1/render", data=bike_xml)
        if result.status_code == 200:
            return result.content
        raise InternalError(f"Rendering request failed {result}")

    def _read_standard_bike_xml(self, handler):
        with open(STANDARD_BIKE_RESOURCE) as file:
            handler.set_xml(file.read())
