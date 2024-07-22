from biked_commons.xml_handling.bike_xml_handler import BikeXmlHandler
from biked_commons.xml_handling.algebraic_parser import AlgebraicParser
import unittest
import os

file_path = os.path.join(os.path.dirname(__file__), "resources/test.xml")


class XmlHandlerTest(unittest.TestCase):
    def setUp(self):
        with open(file_path, "r") as file:
            self.xml_handler = BikeXmlHandler()
            self.xml_handler.set_xml(file.read())
        self.ENTRY_TAG = self.xml_handler.XML_TAG
        self.ENTRY_KEY = self.xml_handler.ATTRIBUTE
        self.PARENT_TAG = self.xml_handler.PARENT_TAG

    def test_duplicate_keys(self):
        xml_handler = BikeXmlHandler()
        xml_handler.set_xml("""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
        <!DOCTYPE properties SYSTEM "http://java.sun.com/dtd/properties.dtd">
        <properties>
        <comment> Made with care! </comment>
        <entry key="first">1</entry>
        <entry key="first">2</entry>
        </properties>""")
        entries = xml_handler.get_entries_dict()
        self.assertEqual(1, len(entries))
        self.assertEqual("2", entries["first"])

    def test_parse_xml(self):
        xml_handler = BikeXmlHandler()
        xml_handler.set_xml("""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE properties SYSTEM "http://java.sun.com/dtd/properties.dtd">
<properties>
<comment> Made with care! </comment>
<entry key="first">12.5</entry>
<entry key="second">TRUE</entry>
</properties>""")
        parsed_entries = xml_handler.get_parsed_entries(AlgebraicParser().attempt_parse)
        self.assertEqual(2, len(parsed_entries))
        self.assertEqual(12.5, parsed_entries["first"])
        self.assertIs(float, type(parsed_entries["first"]))
        self.assertEqual(1, parsed_entries["second"])
        self.assertEqual(float, type(parsed_entries["second"]))

    def test_xml_tree_contains_entries(self):
        self.assertEqual(2, self.xml_handler.get_entries_count())

    def test_can_copy(self):
        self.assertEqual("ready", self.xml_handler.copy_first_entry()[self.ENTRY_KEY])
        self.assertEqual("3", self.xml_handler.copy_first_entry().text)

    def test_can_add_new_entries(self):
        new_key = "key"
        new_value = "value"
        self.xml_handler.add_new_entry(new_key, new_value)
        self.assertEqual(3, self.xml_handler.get_entries_count())
        self.assertEqual('[<entry key="ready">3</entry>, <entry key="stuff">5</entry>, '
                         f'<entry key="{new_key}">{new_value}</entry>]', self.xml_handler.get_all_entries_string())

    def test_can_get_specific_entry(self):
        stuff_entry = self.get_stuff_entry()
        ready_entry = self.get_ready_entry()
        none_entry = self.xml_handler.find_entry_by_key("does not exist")
        self.assertEqual('<entry key="ready">3</entry>', ready_entry.__str__())
        self.assertEqual('<entry key="stuff">5</entry>', stuff_entry.__str__())
        self.assertIsNone(none_entry)

    def test_can_update_entry(self):
        new_stuff_key = "new_stuff"
        new_ready_key = "new_ready"
        self.xml_handler.update_entry_key(self.get_stuff_entry(), new_stuff_key)
        self.xml_handler.update_entry_key(self.get_ready_entry(), new_ready_key)
        self.assertEqual(f'[<entry key="{new_ready_key}">3</entry>, <entry key="{new_stuff_key}">5</entry>]',
                         self.xml_handler.get_all_entries_string())

    def test_can_update_value(self):
        new_stuff_value = "NEW VALUE"
        self.xml_handler.update_entry_value(self.get_stuff_entry(), new_stuff_value)
        self.assertEqual(f'[<entry key="ready">3</entry>, <entry key="stuff">{new_stuff_value}</entry>]',
                         self.xml_handler.get_all_entries_string())

    def test_modifying_copy_does_not_modify_original(self):
        original_entries = self.xml_handler.get_all_entries_string()
        copy = self.xml_handler.copy_first_entry()
        self.xml_handler.update_entry_value(copy, "DUMMY")
        self.xml_handler.update_entry_key(copy, "DUMMY")
        self.assertEqual(original_entries,
                         self.xml_handler.get_all_entries_string())

    def test_end_to_end(self):
        expected_final_content = \
            '<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE properties SYSTEM ' \
            '"http://java.sun.com/dtd/properties.dtd">\n<properties>\n<comment> Made with care! ' \
            '</comment>\n<entry key="ready-updated-key">3</entry>\n<entry ' \
            'key="stuff">new-stuff</entry>\n<entry ' \
            'key="new_key">10</entry></properties>'
        self.xml_handler.add_new_entry(key="new_key", value="10")
        self.xml_handler.update_entry_key(self.get_ready_entry(), "ready-updated-key")
        self.xml_handler.update_entry_value(self.get_stuff_entry(), "new-stuff")
        self.assertEqual(self.xml_handler.get_content_string(), expected_final_content)

    def test_can_get_entries_dict(self):
        self.assertEqual(self.xml_handler.get_entries_dict(), {"ready": "3", "stuff": "5"})

    def test_does_entry_exist(self):
        self.assertFalse(self.xml_handler.does_entry_exist("does not exist"))
        self.assertTrue(self.xml_handler.does_entry_exist("ready"))

    def test_remove_entry(self):
        self.xml_handler.remove_entry(self.get_stuff_entry())
        self.assertEqual('[<entry key="ready">3</entry>]',
                         self.xml_handler.get_all_entries_string())
        self.xml_handler.remove_entry(self.get_ready_entry())
        self.assertEqual(0, self.xml_handler.get_entries_count())

    def test_remove_all_entries(self):
        self.xml_handler.remove_all_entries()
        self.assertEqual(0, self.xml_handler.get_entries_count())

    def test_empty_xml(self):
        self.xml_handler.set_xml("")
        self.assertEqual('<?xml version="1.0" encoding="utf-8"?>\n', self.xml_handler.get_content_string())

    def test_generate_xml_from_dict(self):
        handler = BikeXmlHandler()
        handler.set_entries_from_dict({num: num for num in range(3)})
        self.assertEqual('''<?xml version="1.0" encoding="utf-8"?>
<properties><entry key="0">0</entry><entry key="1">1</entry><entry key="2">2</entry></properties>''',
                         handler.get_content_string())

    def test_set_xml_does_not_throw(self):
        garbage = "fewfwefew"
        self.xml_handler.set_xml(garbage)
        self.assertEqual(0, self.xml_handler.get_entries_count())

    def test_fill_entries_from_dict(self):
        self.xml_handler.set_entries_from_dict({"first": "1", "third": "3"})
        self.assertEqual('[<entry key="first">1</entry>, <entry key="third">3</entry>]',
                         self.xml_handler.get_all_entries_string())

    def test_update_xml_from_dict(self):
        self.xml_handler.update_entries_from_dict({"ready": "ready-new", "new": "new-value"})
        self.assertEqual(
            '[<entry key="ready">ready-new</entry>, <entry key="stuff">5</entry>, <entry key="new">new-value</entry>]',
            self.xml_handler.get_all_entries_string())

    def get_ready_entry(self):
        return self.xml_handler.find_entry_by_key("ready")

    def get_stuff_entry(self):
        return self.xml_handler.find_entry_by_key("stuff")
