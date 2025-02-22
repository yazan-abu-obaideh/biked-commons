import pandas as pd

from biked_commons.xml_handling.cad_builder import BikeCadFileBuilder
from biked_commons.rendering.one_hot_clips import ONE_HOT_ENCODED_CLIPS_COLUMNS


def one_hot_decode(bike: pd.Series) -> dict:
    result = {}
    for encoded_value in ONE_HOT_ENCODED_CLIPS_COLUMNS:
        for column in bike.index:
            if encoded_value in column and bike[column] == 1:
                result[encoded_value] = column.split('OHCLASS:')[1].strip()
    return result


class XmlTransformer:
    def __init__(self):
        self.cad_builder = BikeCadFileBuilder()

    def clip_to_xml(self, template_xml: str, clips_object: dict) -> str:
        return self.cad_builder.build_cad_from_clips_object(clips_object,
                                                            template_xml)

    def biked_to_xml(self, template_xml: str, biked_object: dict) -> str:
        return self.cad_builder.build_cad_from_biked(biked_object,
                                                     template_xml)
