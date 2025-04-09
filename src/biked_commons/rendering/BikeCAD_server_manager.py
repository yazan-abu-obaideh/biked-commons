import atexit
import os
import subprocess
import time
from typing import Callable

import psutil
import requests

from biked_commons.exceptions import InternalError, check_internal_precondition
from biked_commons.resource_utils import resource_path

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


class ServerManager:

    def start_server(self, port: int, pid_consumer: Callable):
        if not self.check_server_health(port):
            print(f"Starting BikeCAD server on port {port}...")
            process = subprocess.Popen(
                [JAVA_BINARY, "-jar", resource_path("BikeCAD-server.jar"), f"--server.port={port}"])
            pid_consumer(process.pid)
            seconds_waited = 0
            while not self.check_server_health(port):
                time.sleep(1)
                seconds_waited += 1
                if seconds_waited > SERVER_START_TIMEOUT_SECONDS:
                    raise InternalError(f"Could not start server on port {port}...")
            print(f"BikeCAD server started on port {port}.")

    def _kill_server(self, server_pid: int):
        if server_pid and psutil.pid_exists(server_pid):
            print(f"BikeCAD Server with pid {server_pid} exists. Killing...")
            psutil.Process(pid=server_pid).kill()
            print(f"BikeCAD server with pid {server_pid} killed successfully.")
        else:
            print(f"Pid {server_pid} does not exist")

    def check_server_health(self, port: int) -> bool:
        try:
            health_response = requests.get(self.endpoint(port, "/actuator/serverInformation"), timeout=1)
        except Exception as ignored:
            return False
        if health_response.status_code != 200:
            return False
        return health_response.json()["serverName"] == "BikeCAD-server"

    def endpoint(self, port: int, suffix: str):
        url = f"http://localhost:{port}{suffix}"
        check_internal_precondition(suffix.startswith("/"), f"Invalid url {url}")
        return url


class SingleThreadedBikeCadServerManager(ServerManager):
    SERVER_PORT = 8080

    def __init__(self):
        super().__init__()
        self._server_pid = -1
        self.start_server(self.SERVER_PORT, self._set_pid)
        atexit.register(self._kill_server, self._server_pid)

    def _set_pid(self, pid: int):
        print(f"Setting PID to {pid}")
        self._server_pid = pid
