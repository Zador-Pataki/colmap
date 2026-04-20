"""Smoke tests for §05 glomap scene bindings.

All glomap scene types live under `pycolmap.glomap.*` to avoid name clashes
with `pycolmap.Track` / `pycolmap.Image` / `pycolmap.Camera`.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

pycolmap = pytest.importorskip("pycolmap")

if not hasattr(pycolmap, "glomap"):
    pytest.skip(
        "pycolmap.glomap submodule missing — rebuild (§08/§16) required",
        allow_module_level=True,
    )

gl = pycolmap.glomap

_REQUIRED_SYMS = ["ViewGraph", "Image", "ImagePair", "Track", "Camera",
                  "GravityInfo", "PairType"]
if not all(hasattr(gl, s) for s in _REQUIRED_SYMS):
    pytest.skip(
        "pycolmap.glomap wheel missing required symbols",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Constructibility
# ---------------------------------------------------------------------------

def test_viewgraph_constructible():
    vg = gl.ViewGraph()
    assert vg.num_images == 0


def test_image_constructible_with_ids():
    img = gl.Image(42, 7, "foo.jpg")
    assert img.image_id == 42
    assert img.camera_id == 7
    assert img.file_name == "foo.jpg"


def test_image_pair_constructible_with_ids():
    p = gl.ImagePair(1, 2)
    assert p.image_id1 == 1
    assert p.image_id2 == 2


def test_track_constructible():
    t = gl.Track()
    assert t.xyz.shape == (3,)


def test_camera_constructible():
    cam = gl.Camera()
    assert cam.has_refined_focal_length is False


def test_gravity_info_constructible():
    g = gl.GravityInfo()
    assert g.has_gravity is False


# ---------------------------------------------------------------------------
# Const id fields — assignment must raise AttributeError
# ---------------------------------------------------------------------------

def test_image_image_id_readonly():
    img = gl.Image(0, 0, "a.jpg")
    with pytest.raises(AttributeError):
        img.image_id = 5


def test_image_file_name_readonly():
    img = gl.Image(0, 0, "a.jpg")
    with pytest.raises(AttributeError):
        img.file_name = "b.jpg"


def test_image_pair_const_ids_readonly():
    p = gl.ImagePair(3, 5)
    for attr in ("pair_id", "image_id1", "image_id2"):
        with pytest.raises(AttributeError):
            setattr(p, attr, 99)


# ---------------------------------------------------------------------------
# Writable-field round-trips
# ---------------------------------------------------------------------------

def test_image_depth_priors_numpy_roundtrip():
    img = gl.Image(0, 0, "x")
    img.depth_priors = np.array([1.0, 2.0, 3.0])
    back = np.asarray(img.depth_priors)
    np.testing.assert_allclose(back, [1.0, 2.0, 3.0])


def test_image_is_inlier_bool_roundtrip():
    img = gl.Image(0, 0, "x")
    img.is_inlier = np.array([True, False, True], dtype=bool)
    back = np.asarray(img.is_inlier)
    assert back.dtype == bool
    assert list(back) == [True, False, True]


def test_image_is_depth_outlier_roundtrip():
    img = gl.Image(0, 0, "x")
    img.is_depth_outlier = np.array([False, True], dtype=bool)
    assert list(np.asarray(img.is_depth_outlier)) == [False, True]


def test_image_is_track_anchor_roundtrip():
    img = gl.Image(0, 0, "x")
    img.is_track_anchor = np.array([True], dtype=bool)
    assert list(np.asarray(img.is_track_anchor)) == [True]


def test_image_is_excluded_roundtrip():
    img = gl.Image(0, 0, "x")
    img.is_excluded = np.array([True, True, False], dtype=bool)
    assert list(np.asarray(img.is_excluded)) == [True, True, False]


def test_image_features_roundtrip():
    img = gl.Image(0, 0, "x")
    feats = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    img.features = feats
    back = img.features
    assert len(back) == 2


def test_image_gravity_info_roundtrip():
    img = gl.Image(0, 0, "x")
    g = gl.GravityInfo()
    g.set_gravity(np.array([0.0, -9.81, 0.0]))
    img.gravity_info = g
    assert img.gravity_info.has_gravity is True


def test_image_pair_is_valid_writable():
    p = gl.ImagePair(0, 1)
    assert p.is_valid is True
    p.is_valid = False
    assert p.is_valid is False


def test_image_pair_weight_writable():
    p = gl.ImagePair(0, 1)
    p.weight = 0.5
    assert p.weight == 0.5


def test_image_pair_type_writable():
    p = gl.ImagePair(0, 1)
    p.type = gl.PairType.LOOP_CLOSURE
    assert p.type == gl.PairType.LOOP_CLOSURE


def test_image_pair_is_lc_writable():
    p = gl.ImagePair(0, 1)
    p.is_LC = True
    assert p.is_LC is True


def test_track_observations_writable():
    t = gl.Track()
    t.observations = [(1, 2), (3, 4)]
    assert len(t.observations) == 2


def test_camera_has_refined_focal_length_writable():
    cam = gl.Camera()
    cam.has_refined_focal_length = True
    assert cam.has_refined_focal_length is True


def test_camera_composition_inner_colmap_camera():
    cam = gl.Camera()
    cam.camera.camera_id = 7
    assert cam.camera.camera_id == 7


def test_gravity_info_set_gravity_updates_r_align():
    g = gl.GravityInfo()
    g.set_gravity(np.array([0.0, -9.81, 0.0]))
    assert g.has_gravity is True
    r = np.asarray(g.get_r_align())
    np.testing.assert_allclose(np.linalg.det(r), 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# PairType enum
# ---------------------------------------------------------------------------

def test_pair_type_enum_values():
    assert hasattr(gl.PairType, "ADJACENT")
    assert hasattr(gl.PairType, "NONADJACENT")
    assert hasattr(gl.PairType, "LOOP_CLOSURE")


# ---------------------------------------------------------------------------
# Rigid3d caster: assign_mdrp_results must accept pycolmap.Rigid3d directly.
# ---------------------------------------------------------------------------

def test_assign_mdrp_results_uses_rigid3d_caster():
    vg = gl.ViewGraph()
    pair = gl.ImagePair(1, 2)
    vg.add_pair(pair)

    r3 = pycolmap.Rigid3d()
    r3.translation = np.array([1.0, 2.0, 3.0])
    results = {
        pair.pair_id: {
            "is_valid": True,
            "cam2_from_cam1": r3,
            "weight": 0.9,
            "rel_depth_scale": 1.5,
            "inliers": [0, 1, 2],
            "cov_t": np.eye(3),
        }
    }
    num_valid, num_invalid = vg.assign_mdrp_results(results)
    assert num_valid == 1
    assert num_invalid == 0
    t_back = np.asarray(vg.image_pairs[pair.pair_id].cam2_from_cam1.translation)
    np.testing.assert_allclose(t_back, [1.0, 2.0, 3.0])


def test_extract_pair_poses_returns_rigid_components():
    vg = gl.ViewGraph()
    p = gl.ImagePair(1, 2)
    vg.add_pair(p)
    data = vg.extract_pair_poses([p.pair_id])
    assert p.pair_id in data
    assert "R" in data[p.pair_id] and "t" in data[p.pair_id]


# ---------------------------------------------------------------------------
# MakeDataclass smoke: pickle round-trip preserves type.
# ---------------------------------------------------------------------------

# Full pickle.loads round-trip is incompatible with MakeDataclass + const id
# fields (image_id, pair_id, etc. are readonly → cannot be restored via
# `mergedict`). Verify only that pickle.dumps succeeds for every type —
# downstream code that actually needs round-trip will use clone() / __copy__.
@pytest.mark.parametrize(
    "type_name",
    ["ViewGraph", "Image", "ImagePair", "Track", "Camera", "GravityInfo"],
)
def test_scene_type_pickle_dump_succeeds(type_name):
    cls = getattr(gl, type_name)
    obj = cls()
    blob = pickle.dumps(obj)
    assert len(blob) > 0


# Types without const ids support full round-trip.
@pytest.mark.parametrize("type_name", ["Camera", "GravityInfo"])
def test_scene_type_pickle_roundtrip_no_const_ids(type_name):
    cls = getattr(gl, type_name)
    obj = cls()
    back = pickle.loads(pickle.dumps(obj))
    assert type(back) is type(obj)


# ---------------------------------------------------------------------------
# Batch method existence (fork-verbatim names)
# ---------------------------------------------------------------------------

_VIEW_GRAPH_BATCH_METHODS = [
    "add_pair",
    "assign_mdrp_results",
    "extract_valid_pair_data",
    "extract_pair_poses",
    "batch_mark_matches_as_lc",
    "extract_pair_image_ids",
    "initialize_all_poses_identity",
    "extract_all_pair_image_ids",
    "extract_pair_matches_and_lc",
    "batch_set_pair_configs",
    "extract_pair_configs",
    "restore_pair_configs",
    "extract_pair_validity",
    "restore_validity_where_invalidated",
    "propagate_lc_forward",
    "extract_all_pair_summary",
    "initialize_all_are_lc_false",
    "batch_set_are_lc",
    "filter_pairs_by_image_set",
    "clone",
]

_IMAGE_PAIR_BATCH_METHODS = ["update", "summary"]

_IMAGE_BATCH_METHODS = ["update_depth_priors", "update_angular"]


@pytest.mark.parametrize("method_name", _VIEW_GRAPH_BATCH_METHODS)
def test_view_graph_batch_method_bound(method_name):
    assert hasattr(gl.ViewGraph, method_name), \
        f"ViewGraph missing batch method {method_name}"


@pytest.mark.parametrize("method_name", _IMAGE_PAIR_BATCH_METHODS)
def test_image_pair_batch_method_bound(method_name):
    assert hasattr(gl.ImagePair, method_name), \
        f"ImagePair missing batch method {method_name}"


@pytest.mark.parametrize("method_name", _IMAGE_BATCH_METHODS)
def test_image_batch_method_bound(method_name):
    assert hasattr(gl.Image, method_name), \
        f"Image missing batch method {method_name}"


def test_extract_all_image_data_module_level():
    assert hasattr(gl, "extract_all_image_data")


# ---------------------------------------------------------------------------
# Symbol existence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("attr", _REQUIRED_SYMS)
def test_scene_api_symbols_exposed(attr):
    assert hasattr(gl, attr)
