from biked_commons.xml_handling.bike_xml_handler import BikeXmlHandler


def to_attributes_dict(bike_xml: str):
    handler = BikeXmlHandler()
    handler.set_xml(bike_xml)
    return handler.get_entries_dict()
