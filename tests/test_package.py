"""Test the package."""

import re

import disinfo_relation_checker as drc


def test_package_has_version() -> None:
    """Test that the package has a version."""
    assert re.match(r"^0\.\d+\.\d+$", drc.__version__)
