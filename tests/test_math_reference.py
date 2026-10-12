from html.parser import HTMLParser

from fer_rf_sim.math_reference import MATH_SECTIONS, math_reference_html
from fer_rf_sim.physics import E_F, F_BLEND, F_SWITCH, a
from fer_rf_sim.provenance import MODEL_VERSION


class ReferenceParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.anchors = []
        self.links = []
        self.text = []
        self.resources = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "a":
            if "name" in attributes:
                self.anchors.append(attributes["name"])
            if "href" in attributes:
                self.links.append(attributes["href"])
        if "src" in attributes:
            self.resources.append(attributes["src"])

    def handle_data(self, data):
        self.text.append(data)


def test_reference_has_unique_topics_and_only_local_links():
    parser = ReferenceParser()
    parser.feed(math_reference_html())
    expected = [anchor for anchor, _, _ in MATH_SECTIONS]
    assert len(expected) == len(set(expected))
    assert parser.anchors == expected
    assert parser.links == [f"#{anchor}" for anchor in expected]
    assert not parser.resources
    assert {"notation", "barrier", "wkb", "airy", "hybrid", "current", "rf", "lut",
            "constant-current", "peaks", "broadening", "experimental-units",
            "recipe-fits", "fer-index", "empirical-models", "image-charge", "limits"} == set(expected)


def test_reference_constants_match_the_numerical_model():
    sections = {anchor: body for anchor, _, body in MATH_SECTIONS}
    assert f"E<sub>F</sub> = {E_F:g} eV" in sections["notation"]
    assert f"a = {a:g} nm" in sections["notation"]
    assert f"F<sub>1</sub> = {F_SWITCH:g} V/nm" in sections["hybrid"]
    assert f"b = {F_BLEND:g} V/nm" in sections["hybrid"]
    assert MODEL_VERSION in sections["hybrid"]
    assert "10 Z" not in sections["notation"]


def test_reference_explains_scope_and_current_averaging_order():
    parser = ReferenceParser()
    parser.feed(math_reference_html())
    text = " ".join(" ".join(parser.text).split())
    for phrase in ("complete currents", "fixed-height partial derivative",
                   "not calibrated amperes", "half-prominence width",
                   "Existing inconsistency", "not enabled", "quasistatic",
                   "does not extrapolate", "Lambert W", "scaled Airy boundary matching",
                   "not the active default transmission"):
        assert phrase in text