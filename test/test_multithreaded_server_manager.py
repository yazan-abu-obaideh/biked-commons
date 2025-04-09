import unittest

from biked_commons.rendering.BikeCAD_server_manager import MultiThreadedBikeCadServerManager


class MultiThreadedServerManagerTest(unittest.TestCase):
    def test_load_balancing(self):
        manager = MultiThreadedBikeCadServerManager(number_servers=3, timeout_seconds=90)
        self.assertEqual(manager.STARTING_PORT, 8080)
        self.assertEqual(manager.endpoint("/hello"), "http://localhost:8080/hello")
        self.assertEqual(manager.endpoint("/hello"), "http://localhost:8081/hello")
        self.assertEqual(manager.endpoint("/hello"), "http://localhost:8082/hello")
        self.assertEqual(manager.endpoint("/hello"), "http://localhost:8080/hello")
        self.assertEqual(manager.endpoint("/hello"), "http://localhost:8081/hello")
