import pytest

import pycolmap


def test_rotation_weight_type_enum():
    assert {
        k: int(v) for k, v in pycolmap.RotationWeightType.__members__.items()
    } == {
        "GEMAN_MCCLURE": 0,
        "HALF_NORM": 1,
    }


def test_gravity_refiner_options_max_outlier_ratio_readwrite():
    options = pycolmap.GravityRefinerOptions()
    assert isinstance(options.max_outlier_ratio, float)
    options.max_outlier_ratio = 0.5
    assert options.max_outlier_ratio == 0.5


def test_gravity_refiner_options_max_gravity_error_readwrite():
    options = pycolmap.GravityRefinerOptions()
    assert isinstance(options.max_gravity_error, float)
    options.max_gravity_error = 10.0
    assert options.max_gravity_error == 10.0


def test_gravity_refiner_options_min_num_neighbors_readwrite():
    options = pycolmap.GravityRefinerOptions()
    assert isinstance(options.min_num_neighbors, int)
    options.min_num_neighbors = 5
    assert options.min_num_neighbors == 5


@pytest.mark.parametrize(
    "name",
    [
        "run_rotation_averaging",
        "run_gravity_refinement",
        "run_global_positioning",
    ],
)
def test_public_api_callable(name):
    assert callable(getattr(pycolmap, name))


# --- Fork extension tests ---

import numpy as np


def test_loss_function_config_default():
    cfg = pycolmap.LossFunctionConfig()
    assert cfg.name == "trivial"
    assert cfg.scale == 1.0
    assert cfg.weight == 1.0


def test_loss_function_config_init_with_args():
    cfg = pycolmap.LossFunctionConfig("huber", 0.5, 2.0)
    assert cfg.name == "huber"
    assert cfg.scale == 0.5
    assert cfg.weight == 2.0


def test_loss_function_config_fields_mutable():
    cfg = pycolmap.LossFunctionConfig()
    cfg.name = "cauchy"
    cfg.scale = 0.1
    cfg.weight = 3.0
    assert cfg.name == "cauchy"
    assert cfg.scale == 0.1
    assert cfg.weight == 3.0


def test_point_constraint_type_enum():
    assert hasattr(pycolmap.PointConstraintType, "GEOMETRY_ONLY")
    assert hasattr(pycolmap.PointConstraintType, "SPLIT_METRIC_DEPTH")
    assert pycolmap.PointConstraintType.GEOMETRY_ONLY != pycolmap.PointConstraintType.SPLIT_METRIC_DEPTH


def test_constraint_type_enum():
    assert hasattr(pycolmap.ConstraintType, "ONLY_POINTS")
    assert hasattr(pycolmap.ConstraintType, "ALL")
    assert pycolmap.ConstraintType.ONLY_POINTS != pycolmap.ConstraintType.ALL


def test_global_positioner_options_fork_flags():
    opt = pycolmap.GlobalPositionerOptions()
    assert opt.use_log_residual_for_depth is False
    assert opt.filter_depth_outliers is False
    assert opt.use_relative_pose_constraints is False


def test_global_positioner_options_point_constraint_type_roundtrip():
    opt = pycolmap.GlobalPositionerOptions()
    opt.point_constraint_type = pycolmap.PointConstraintType.SPLIT_METRIC_DEPTH
    assert opt.point_constraint_type == pycolmap.PointConstraintType.SPLIT_METRIC_DEPTH


def test_global_positioner_options_constraint_type_roundtrip():
    opt = pycolmap.GlobalPositionerOptions()
    opt.constraint_type = pycolmap.ConstraintType.ONLY_POINTS
    assert opt.constraint_type == pycolmap.ConstraintType.ONLY_POINTS


def test_global_positioner_options_loss_config_roundtrip():
    opt = pycolmap.GlobalPositionerOptions()
    cfg = pycolmap.LossFunctionConfig("huber", 0.1, 1.0)
    opt.loss_normal_geometry = cfg
    assert opt.loss_normal_geometry.name == "huber"
    assert opt.loss_normal_geometry.scale == 0.1


@pytest.mark.parametrize(
    "field,value",
    [
        ("random_init_scale", 50.0),
        ("use_init", True),
        ("optimize_depth_map_scales", False),
        ("regularize_rotations", False),
        ("rotation_prior_sigma", 0.02),
        ("scale_regularization_sigma", 2.0),
        ("scale_prior_stddev", 0.5),
        ("regularize_depth_map_scales", False),
        ("use_log_scale_for_depth_map_scales", False),
        ("zero_residual_behind_camera", True),
        ("smooth_log_linear_transition", True),
        ("log_linear_threshold", 0.2),
        ("consecutive_trivial", False),
    ],
)
def test_global_positioner_options_fork_field_settable(field, value):
    opt = pycolmap.GlobalPositionerOptions()
    setattr(opt, field, value)
    assert getattr(opt, field) == value


def test_rotation_estimator_options_fork_axis_roundtrip():
    opt = pycolmap.RotationEstimatorOptions()
    opt.axis = np.array([0.0, 0.0, 1.0])
    assert np.allclose(opt.axis, [0.0, 0.0, 1.0])


def test_rotation_estimator_options_gravity_world_roundtrip():
    opt = pycolmap.RotationEstimatorOptions()
    opt.gravity_world = np.array([0.0, -9.81, 0.0])
    assert np.allclose(opt.gravity_world, [0.0, -9.81, 0.0])


def test_rotation_estimator_options_gravity_loss_type_roundtrip():
    opt = pycolmap.RotationEstimatorOptions()
    opt.gravity_loss_type = "huber"
    assert opt.gravity_loss_type == "huber"


@pytest.mark.parametrize(
    "field,value",
    [
        ("use_precomputed_weights", True),
        ("fix_non_lc_weights", True),
        ("fixed_non_lc_weight", 100.0),
        ("skip_risky_LC_pairs", True),
        ("use_video_constraints", True),
        ("video_tracking_huber_scale", 0.2),
        ("video_lc_cauchy_scale", 0.1),
        ("use_gravity_prior", True),
        ("default_gravity_sigma", 0.05),
        ("gravity_loss_scale", 0.1),
    ],
)
def test_rotation_estimator_options_fork_field_settable(field, value):
    opt = pycolmap.RotationEstimatorOptions()
    setattr(opt, field, value)
    assert getattr(opt, field) == value
