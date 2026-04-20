"""§07 smoke tests for pycolmap.glomap options structs."""

from __future__ import annotations

import pickle

import pytest

pycolmap = pytest.importorskip("pycolmap")
if not hasattr(pycolmap, "glomap") or not hasattr(pycolmap.glomap,
                                                   "GlobalPositionerOptions"):
    pytest.skip(
        "pycolmap.glomap.GlobalPositionerOptions missing — rebuild required",
        allow_module_level=True,
    )

gl = pycolmap.glomap

OPTIONS_CLASSES = [
    "GlobalPositionerOptions",
    "RotationEstimatorOptions",
    "GlobalMapperOptions",
    "ViewGraphCalibratorOptions",
    "TriangulatorOptions",
    "TrackEstablishmentOptions",
    "BundleAdjustmentOptions",
    "InlierThresholdOptions",
    "OptimizationBaseOptions",
]


@pytest.mark.parametrize("name", OPTIONS_CLASSES)
def test_options_makedataclass_installed(name):
    cls = getattr(gl, name)
    obj = cls()
    assert hasattr(obj, "summary")
    assert hasattr(obj, "todict")
    assert hasattr(obj, "mergedict")
    assert repr(obj)


def test_options_dict_kwargs_construction():
    opts = gl.GlobalPositionerOptions(random_init_scale=250.0, seed=42)
    assert opts.random_init_scale == 250.0
    assert opts.seed == 42


def test_nested_loss_function_config_dict():
    opts = gl.GlobalPositionerOptions(
        loss_normal_geometry={"name": "huber", "scale": 1.0, "weight": 2.0},
    )
    assert opts.loss_normal_geometry.name == "huber"
    assert opts.loss_normal_geometry.scale == 1.0
    assert opts.loss_normal_geometry.weight == 2.0


def test_nested_loss_function_config_explicit():
    cfg = gl.GlobalPositionerOptions.LossFunctionConfig()
    cfg.name = "huber"
    cfg.scale = 0.5
    cfg.weight = 3.0
    opts = gl.GlobalPositionerOptions(loss_normal_geometry=cfg)
    assert opts.loss_normal_geometry.name == "huber"
    assert opts.loss_normal_geometry.scale == 0.5


def test_nested_enum_constraint_type():
    assert gl.GlobalPositionerOptions.ConstraintType.ONLY_POINTS is not None


def test_nested_enum_point_constraint_type():
    PCT = gl.GlobalPositionerOptions.PointConstraintType
    assert PCT.GEOMETRY_ONLY is not None
    assert PCT.SPLIT_METRIC_DEPTH is not None


def test_rotation_estimator_solver_type_enum():
    ST = gl.RotationEstimatorOptions.SolverType
    assert ST.L1_IRLS is not None
    assert ST.CERES_GRAVITY is not None


def test_rotation_estimator_weight_type_enum():
    WT = gl.RotationEstimatorOptions.WeightType
    assert WT.GEMAN_MCCLURE is not None
    assert WT.HALF_NORM is not None


def test_mergedict_updates_fields():
    opts = gl.GlobalPositionerOptions()
    opts.mergedict({"seed": 99, "min_num_view_per_track": 5})
    assert opts.seed == 99
    assert opts.min_num_view_per_track == 5


def test_mergedict_nested_loss_config():
    opts = gl.GlobalPositionerOptions()
    opts.mergedict({"loss_lc_geometry": {"name": "cauchy", "scale": 2.0}})
    assert opts.loss_lc_geometry.name == "cauchy"
    assert opts.loss_lc_geometry.scale == 2.0


def test_global_mapper_options_composes_subopts():
    gm = gl.GlobalMapperOptions()
    assert isinstance(gm.opt_gp, gl.GlobalPositionerOptions)
    assert isinstance(gm.opt_ra, gl.RotationEstimatorOptions)
    assert isinstance(gm.opt_ba, gl.BundleAdjustmentOptions)
    assert isinstance(gm.inlier_thresholds, gl.InlierThresholdOptions)


def test_pickle_dumps_succeeds_for_all():
    # MakeDataclass+pickle is fine for POD options since there are no const id
    # fields (contrast with §05 scene types).
    for name in OPTIONS_CLASSES:
        obj = getattr(gl, name)()
        blob = pickle.dumps(obj)
        assert len(blob) > 0
