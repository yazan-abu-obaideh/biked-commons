import os

import pandas as pd

from biked_commons.exceptions import UserInputException
from biked_commons.rendering.one_hot_clips import ONE_HOT_ENCODED_CLIPS_COLUMNS
from biked_commons.xml_handling.bike_xml_handler import BikeXmlHandler
from biked_commons.xml_handling.clips_to_bcad import clips_to_cad

OPTIMIZED_TO_CAD = {
    "ST Angle": "Seat angle",
    "HT Length": "Head tube length textfield",
    "HT Angle": "Head angle",
    "HT LX": "Head tube lower extension2",
    'Stack': 'Stack',
    "ST Length": "Seat tube length",
    "Seatpost LENGTH": "Seatpost LENGTH",
    "Saddle height": "Saddle height",
    "Stem length": "Stem length",
    "Crank length": "Crank length",
    "Headset spacers": "Headset spacers",
    "Stem angle": "Stem angle",
    "Handlebar style": "Handlebar style",
}


def one_hot_decode(bike: pd.Series) -> dict:
    result = {}
    for encoded_value in ONE_HOT_ENCODED_CLIPS_COLUMNS:
        for column in bike.index:
            if encoded_value in column and bike[column] == 1:
                result[encoded_value] = column.split('OHCLASS:')[1].strip()
    return result


def _get_valid_seed_bike(seed_image_id):
    if str(seed_image_id) not in [str(_) for _ in range(1, 14)]:
        raise UserInputException(f"Invalid seed bike ID [{seed_image_id}]")
    return f"bike{seed_image_id}.bcad"


def _build_bike_path(seed_bike_id):
    seed_image = _get_valid_seed_bike(seed_bike_id)
    return os.path.join(os.path.dirname(__file__), "../resources", "seed-bikes", seed_image)


class BikeCadFileBuilder:
    def build_cad_from_biked(self, bike_object, seed_bike_xml: str) -> str:
        xml_handler = BikeXmlHandler()
        xml_handler.set_xml(seed_bike_xml)
        for response_key, cad_key in OPTIMIZED_TO_CAD.items():
            self._update_xml(xml_handler, cad_key, bike_object[response_key])
        # self._update_xml(xml_handler, "Display RIDER", "true")
        return xml_handler.get_content_string()

    def build_cad_from_clips_object(self, target_bike, seed_bike_xml: str) -> str:
        xml_handler = BikeXmlHandler()
        xml_handler.set_xml(seed_bike_xml)
        target_dict = self._to_cad_dict(target_bike)
        self._update_values(xml_handler, target_dict)
        return xml_handler.get_content_string()

    def _to_cad_dict(self, bike: dict):
        bike_complete = clips_to_cad(pd.DataFrame.from_records([bike])).iloc[0]
        decoded_values = one_hot_decode(bike_complete)
        bike_dict = bike_complete.to_dict()
        bike_dict.update(decoded_values)
        return self._remove_encoded_values(bike_dict)

    def _update_xml(self, xml_handler, cad_key, desired_value):
        entry = xml_handler.find_entry_by_key(cad_key)
        if entry:
            xml_handler.update_entry_value(entry, str(desired_value))
        else:
            xml_handler.add_new_entry(cad_key, str(desired_value))

    def _remove_encoded_values(self, bike_dict: dict) -> dict:
        to_delete = []
        for k, _ in bike_dict.items():
            for encoded_key in ONE_HOT_ENCODED_CLIPS_COLUMNS:
                if "OHCLASS" in k and encoded_key in k:
                    to_delete.append(k)
        return {
            k: v for k, v in bike_dict.items() if k not in to_delete
        }

    def _update_values(self, handler, bike_dict):
        num_updated = 0
        for k, v in bike_dict.items():
            parsed = self._parse(v)
            if parsed is not None:
                num_updated += 1
                self._update_value(parsed, handler, k)

    def _parse(self, v):
        handled = self._handle_numeric(v)
        handled = self._handle_bool(str(handled))
        return handled

    def _update_value(self, handled, xml_handler, k):
        xml_handler.add_or_update(k, handled)

    def _handle_numeric(self, v):
        if str(v).lower() == 'nan':
            return None
        if type(v) in [int, float]:
            v = int(v)
        return v

    def _handle_bool(self, param):
        if param.lower().title() in ['True', 'False']:
            return param.lower()
        return param
