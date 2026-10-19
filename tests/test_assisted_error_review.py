"""Fictitious XML controls only; no protected text or workbook authoring."""
import sys
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from verify_assisted_error_review import native_controls


def controls(scroll="C9", split="8", choice="Incertain", width="22"):
    return native_controls(ET.fromstring(f'''<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
      <sheetViews><sheetView><pane xSplit="2" ySplit="{split}" topLeftCell="{scroll}" state="frozen"/></sheetView></sheetViews>
      <cols><col min="5" max="5" width="{width}"/></cols>
      <dataValidations><dataValidation sqref="E9:E38" type="list"><formula1>"{choice}"</formula1></dataValidation></dataValidations>
    </worksheet>'''))


class NativeReviewControlTests(unittest.TestCase):
    def test_scroll_is_distinct_from_frozen_boundary(self):
        self.assertEqual(controls(scroll="C34"), controls(scroll="C9"))

    def test_changed_boundary_is_detected(self):
        self.assertNotEqual(controls(split="8"), controls(split="7"))

    def test_changed_choices_are_detected(self):
        self.assertNotEqual(controls(choice="Incertain"), controls(choice="Other"))

    def test_changed_width_is_detected(self):
        self.assertNotEqual(controls(width="22"), controls(width="12"))


if __name__ == "__main__":
    unittest.main()
