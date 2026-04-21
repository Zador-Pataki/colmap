import pycolmap
import pytest


@pytest.mark.parametrize(
    "name",
    [
        "run_rotation_averaging_extended",
        "run_global_positioning_extended",
    ],
)
def test_extended_pipeline_functions_exist(name):
    assert hasattr(pycolmap, name)
    assert callable(getattr(pycolmap, name))
