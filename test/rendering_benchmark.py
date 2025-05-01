import logging
import os.path
import time
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Dict

from biked_commons.api.rendering import RenderingEngine, FILE_BUILDER
from biked_commons.resource_utils import resource_path, STANDARD_BIKE_RESOURCE

# Configure the logging
logging.basicConfig(level=logging.DEBUG,  # Set the logging level to DEBUG
                    format='%(asctime)s - %(levelname)s - %(message)s')  # Customize the log format


def read_standard_xml():
    with open(STANDARD_BIKE_RESOURCE, "r") as file:
        return file.read()


NUMBER_SERVERS = 10
standard_bike_xml = read_standard_xml()


def get_records_with_id() -> Dict[str, dict]:
    """
    Return records in a dictionary of the form {
    (record_id: str) : (record: dict)
    }
    """
    ...


def record_to_xml(save_path: str, record_id: str, record: dict):
    try:
        file_path = os.path.join(save_path, f"{record_id}.xml")
        with open(file_path, "w") as file:
            xml_data = FILE_BUILDER.build_cad_from_clip(record, standard_bike_xml, False)
            file.write(xml_data)
    except Exception as e:
        print(f"Failed with exception {e}")


def convert_to_xml(records_with_id: Dict[str, dict],
                   process_pool_workers: int,
                   save_dir: str
                   ):
    executor = ProcessPoolExecutor(max_workers=process_pool_workers)
    path = resource_path(save_dir)
    os.makedirs(path, exist_ok=True)
    for record_id, record in records_with_id.items():
        executor.submit(record_to_xml, path, record_id, record)
    executor.shutdown()  # waits for all submitted tasks to finish


def run_rendering_benchmark(
        number_rendering_servers: int,
        thread_pool_workers: int,
        records_with_id: Dict[str, dict],
        save_dir: str,
):
    executor = ThreadPoolExecutor(max_workers=thread_pool_workers)
    rendering_engine = RenderingEngine(number_rendering_servers=number_rendering_servers, server_init_timeout_seconds=3)
    base_path = resource_path(save_dir)

    def render_record(xml: str):
        try:
            xml_path = os.path.join(base_path, xml)
            with open(xml_path, "r") as xml_file:
                print("Sending request to server...")
                read_file = xml_file.read()
                print("Read file...")
                rendering_result = rendering_engine.render_xml(read_file)
                print("Rendering result received from server...")
                with open(xml_path.replace(".xml", ".svg"), "wb") as image_file:
                    image_file.write(rendering_result.image_bytes)
                    print("Image file written to disk.")
                return True
        except Exception as e:
            print(f"Rendering failed: {e}")
            return False, e

    for record_id, _ in records_with_id.items():
        executor.submit(render_record, record_id)

    executor.shutdown()
