import unittest

import requests

from biked_commons.rendering.BikeCAD_server_manager import MultiThreadedBikeCadServerManager


class MultiThreadedServerManagerTest(unittest.TestCase):
    def test_starts_and_load_balances(self):
        manager = MultiThreadedBikeCadServerManager(number_servers=3, timeout_seconds=90)
        ports = [8080, 8081, 8082]

        self.assertEqual(manager.STARTING_PORT, 8080)
        self.assertEqual(manager._port_range, ports)

        for port in ports:
            self.assert_healthy(port)

        self.assertEqual(manager.endpoint("/hello"), "http://localhost:8080/hello")
        self.assertEqual(manager.endpoint("/hello"), "http://localhost:8081/hello")
        self.assertEqual(manager.endpoint("/hello"), "http://localhost:8082/hello")
        self.assertEqual(manager.endpoint("/hello"), "http://localhost:8080/hello")
        self.assertEqual(manager.endpoint("/hello"), "http://localhost:8081/hello")

    def assert_healthy(self, port: int):
        res = requests.get(f"http://localhost:{port}/actuator/serverInformation")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["serverName"], "BikeCAD-server")
