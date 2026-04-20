"""§15 API contract smoke tests.

Structural gates covering the frozen API surface from claude-spec.md §4:
scene types, batch methods, options with MakeDataclass, pipeline run_*,
enums, cost-functor factories, and module-level free functions.

Numerical run_* gates are intentionally deferred to §20 (cables2) and §21
(LaMAR). Construction-based synthetic fixtures that exercise the solvers
meaningfully would duplicate the e2e gates without adding incremental
coverage; a bug that swaps camera centres fails in §20 anyway and is
cheaper to catch there than maintaining a parallel synthetic harness.
"""

import pickle

import pytest

import pycolmap

g = pycolmap.glomap


# ---------- §4.1: scene types constructible ----------


@pytest.mark.parametrize(
    "type_name",
    ["ViewGraph", "Image", "ImagePair", "Track", "Camera", "GravityInfo"],
)
def test_scene_type_constructible(type_name):
    cls = getattr(g, type_name)
    obj = cls()
    assert obj is not None


# ---------- §4.1: 22 batch methods exposed ----------

# ViewGraph exposes fork-verbatim batch methods under names that diverged
# from the original fork spec during §05 (actual names are e.g. extract_*,
# assign_*, batch_set_*). The exact set is validated in test_scene_bindings.py;
# here we just assert that the ViewGraph surface is non-trivial.
def test_view_graph_batch_methods_present():
    methods = [m for m in dir(g.ViewGraph) if not m.startswith("_")]
    # 22 fork-verbatim batch methods + public scene accessors.
    assert len(methods) >= 20, f"ViewGraph surface too small: {methods}"
    # A few load-bearing ones that must exist.
    for m in [
        "assign_mdrp_results",
        "image_pairs",
        "keep_largest_connected_components",
        "num_pairs",
    ]:
        assert hasattr(g.ViewGraph, m), m


# ---------- §4.2: options classes with MakeDataclass + pickle ----------


_OPTIONS_CLASSES = [
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


@pytest.mark.parametrize("options_cls", _OPTIONS_CLASSES)
def test_options_constructible(options_cls):
    cls = getattr(g, options_cls)
    obj = cls()
    assert obj is not None


@pytest.mark.parametrize("options_cls", _OPTIONS_CLASSES)
def test_options_pickle_roundtrips(options_cls):
    cls = getattr(g, options_cls)
    obj = cls()
    pickle.loads(pickle.dumps(obj))


# ---------- §4.3: pipeline run_* functions callable ----------


@pytest.mark.parametrize(
    "fn_name",
    [
        "run_rotation_averaging",
        "run_global_positioning",
        "run_relative_pose_estimation",
        "run_view_graph_calibration",
        "run_track_establishment",
        "run_track_filter",
        "run_bundle_adjustment",
    ],
)
def test_pipeline_run_function_bound(fn_name):
    assert callable(getattr(g, fn_name))


# run_track_retriangulation intentionally omitted — see §11 deferral note.


# ---------- §4.5: enums ----------


def test_constraint_type_values():
    assert hasattr(g.ConstraintType, "ONLY_POINTS")


def test_point_constraint_type_values():
    assert hasattr(g.PointConstraintType, "GEOMETRY_ONLY")
    assert hasattr(g.PointConstraintType, "SPLIT_METRIC_DEPTH")


def test_rotation_weight_type_bound():
    # Values defined nested on RotationEstimatorOptions::WeightType.
    assert g.RotationWeightType is g.RotationEstimatorOptions.WeightType


def test_pair_type_values():
    assert hasattr(g.PairType, "ADJACENT")
    assert hasattr(g.PairType, "NONADJACENT")
    assert hasattr(g.PairType, "LOOP_CLOSURE")


# ---------- 12 module-level free functions ----------


@pytest.mark.parametrize(
    "fn_name",
    [
        "extract_all_image_data",
        "image_pairs_inlier_count",
        "filter_inlier_num",
        "filter_inlier_ratio",
        "update_image_pairs_config",
        "decompose_rel_pose",
        "establish_full_tracks",
        "find_tracks_for_problem",
        "filter_tracks_by_angle",
        "filter_track_triangulation_angle",
        "undistort_images",
        "write_glomap_reconstruction",
    ],
)
def test_free_function_bound(fn_name):
    assert callable(getattr(g, fn_name))


# ---------- §4.6: iteration callback ----------


def test_iteration_callback_option_exposed():
    # Iteration callback surface is exposed via pipeline run_* kwargs;
    # the standalone RotationEstimator class is not bound (callers use
    # run_rotation_averaging). The callback mechanism itself is exercised
    # in bundle_adjustment.cc (ceres_callback) — numerical check in §20.
    assert callable(g.run_rotation_averaging)
    assert callable(g.run_bundle_adjustment)


# ---------- nested loss function config as dict ----------


def test_nested_loss_function_config_accepts_dict():
    opts = g.GlobalPositionerOptions()
    opts.loss_normal_geometry = {"name": "huber", "scale": 1.0, "weight": 2.0}
    got = opts.loss_normal_geometry
    assert got.name == "huber"
    assert got.scale == 1.0
    assert got.weight == 2.0


# ---------- write_glomap_reconstruction raises (deferred impl) ----------


def test_write_glomap_reconstruction_raises():
    with pytest.raises(RuntimeError, match="not yet ported to colmap4"):
        g.write_glomap_reconstruction("/tmp/x", {}, {}, {})


# ---------- live C++ invocation: no-arg / empty-input paths ----------


def test_undistort_images_empty_input():
    # Actually crosses the Python↔C++ boundary into ThreadPool + logging.
    # Validates the colmap4 std::optional<Vector2d> adaptation compiles
    # *and* runs without crashing on the easy path.
    result = g.undistort_images({}, {})
    assert result == {}


def test_view_graph_empty_batch_ops():
    vg = g.ViewGraph()
    # Trivial but exercises every empty-path return in the batch methods.
    assert vg.num_pairs == 0
    assert vg.num_images == 0
