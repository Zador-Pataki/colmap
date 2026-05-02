import json
from pathlib import Path

import numpy as np
import pytest

import pycolmap


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, records) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_f64(path: Path, values) -> None:
    np.asarray(values, dtype="<f8").tofile(path)


def _artifact(filename: str, ids, shape) -> dict:
    return {
        "file": filename,
        "dtype": "float64",
        "byte_order": "little_endian",
        "ids": list(ids),
        "shape": list(shape),
    }


def _make_trace(
    tmp_path: Path, *, with_jacobians: bool, with_loss_rho: bool = False
) -> Path:
    trace_dir = tmp_path / (
        "trace_jacobians" if with_jacobians else "trace_values"
    )
    residual_values_dir = trace_dir / "residual_values"
    residual_values_dir.mkdir(parents=True)

    _write_json(
        trace_dir / "manifest.json",
        {
            "schema_version": 1,
            "run_id": "run",
            "status": "finished",
            "trace_level": (
                "residual_jacobians" if with_jacobians else "residual_values"
            ),
        },
    )
    _write_jsonl(
        trace_dir / "residual_blocks.jsonl",
        [
            {"event_type": "residual_added", "attrs": {"residual_id": "r0"}},
            {"event_type": "residual_added", "attrs": {"residual_id": "r1"}},
        ],
    )

    metadata = {
        "schema_version": 1,
        "run_id": "run",
        "iteration": 0,
        "dtype": "float64",
        "byte_order": "little_endian",
        "num_residual_blocks": 2,
        "total_scalar_residuals": 3,
        "has_raw_jacobians": with_jacobians,
        "residual_ids": ["r0", "r1"],
        "residual_dims": [2, 1],
        "residual_offsets": [0, 2],
        "evaluation_success": [True, True],
        "artifacts": {
            "raw_residuals": {
                "file": "iter_000000_raw_residuals_f64.bin",
                "dtype": "float64",
                "byte_order": "little_endian",
                "shape": [3],
            },
            "raw_costs": {
                "file": "iter_000000_raw_costs_f64.bin",
                "dtype": "float64",
                "byte_order": "little_endian",
                "shape": [2],
            },
            "robust_costs": {
                "file": "iter_000000_robust_costs_f64.bin",
                "dtype": "float64",
                "byte_order": "little_endian",
                "shape": [2],
            },
        },
    }
    if with_jacobians:
        metadata.update(
            {
                "total_jacobian_scalars": 9,
                "parameter_block_sizes": [[3, 1], [1]],
                "raw_jacobian_offsets": [[0, 6], [8]],
                "parameter_blocks": [
                    [
                        {
                            "role": "frame_center",
                            "kind": "frame_center",
                            "id": 10,
                        },
                        {"role": "bata_scale", "kind": "bata_scale", "id": 0},
                    ],
                    [{"role": "dmap_scale", "kind": "dmap_scale", "id": 20}],
                ],
                "parameter_block_is_constant": [[False, True], [False]],
                "parameter_block_lower_bounds": [
                    [[-1.0, -1.0, -1.0], [1e-5]],
                    [[1e-5]],
                ],
                "raw_jacobian_layout": (
                    "residual_block_major/parameter_block_major/row_major"
                ),
                "jacobian_domain": "raw_cost_function_ambient_parameters",
                "loss_applied_to_jacobians": False,
                "manifold_applied_to_jacobians": False,
                "constant_parameter_blocks_included": True,
            }
        )
        metadata["artifacts"]["raw_jacobians"] = {
            "file": "iter_000000_raw_jacobians_f64.bin",
            "dtype": "float64",
            "byte_order": "little_endian",
            "shape": [9],
        }
    if with_loss_rho:
        metadata["loss_rho_layout"] = "residual_block_major/rho0_rho1_rho2"
        metadata["artifacts"]["loss_rho_values"] = {
            "file": "iter_000000_loss_rho_values_f64.bin",
            "dtype": "float64",
            "byte_order": "little_endian",
            "shape": [2, 3],
        }

    _write_json(residual_values_dir / "iter_000000.json", metadata)
    _write_f64(
        residual_values_dir / "iter_000000_raw_residuals_f64.bin",
        [1.0, 2.0, 3.0],
    )
    _write_f64(
        residual_values_dir / "iter_000000_raw_costs_f64.bin", [2.5, 4.5]
    )
    _write_f64(
        residual_values_dir / "iter_000000_robust_costs_f64.bin", [2.25, 4.25]
    )
    if with_loss_rho:
        _write_f64(
            residual_values_dir / "iter_000000_loss_rho_values_f64.bin",
            [[4.5, 0.25, -0.125], [8.5, 1.0, 0.0]],
        )
    if with_jacobians:
        _write_f64(
            residual_values_dir / "iter_000000_raw_jacobians_f64.bin",
            np.arange(9, dtype=np.float64),
        )
    return trace_dir


def _make_snapshot_trace(
    tmp_path: Path, *, include_optional: bool = True
) -> Path:
    trace_dir = tmp_path / "trace_snapshots"
    snapshots_dir = trace_dir / "snapshots"
    snapshots_dir.mkdir(parents=True)
    _write_json(
        trace_dir / "manifest.json",
        {
            "schema_version": 1,
            "run_id": "run",
            "status": "finished",
            "trace_level": "parameter_snapshots",
        },
    )

    iteration = 3
    prefix = f"iter_{iteration:06d}"
    frame_centers = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64
    )
    points3D = np.arange(15, dtype=np.float64).reshape(5, 3)
    scales = np.array([0.5, 1.5], dtype=np.float64)
    dmap_scales = np.array([2.5, 3.5], dtype=np.float64)
    cams_in_rig = np.array([[0.1, 0.2, 0.3]], dtype=np.float64)

    frame_ids = [10, 20]
    point3D_ids = [100, 101, 102, 103, 104]
    scale_ids = [30, 40]
    dmap_ids = [50, 60]
    cam_ids = [70]

    _write_f64(snapshots_dir / f"{prefix}_frame_centers_f64.bin", frame_centers)
    _write_f64(snapshots_dir / f"{prefix}_points3D_f64.bin", points3D)
    _write_f64(snapshots_dir / f"{prefix}_scales_f64.bin", scales)

    artifacts = {
        "frame_centers": _artifact(
            f"{prefix}_frame_centers_f64.bin", frame_ids, frame_centers.shape
        ),
        "points3D": _artifact(
            f"{prefix}_points3D_f64.bin", point3D_ids, points3D.shape
        ),
        "scales": _artifact(
            f"{prefix}_scales_f64.bin", scale_ids, scales.shape
        ),
    }
    dmap_shape = [0]
    if include_optional:
        _write_f64(snapshots_dir / f"{prefix}_dmap_scales_f64.bin", dmap_scales)
        _write_f64(snapshots_dir / f"{prefix}_cams_in_rig_f64.bin", cams_in_rig)
        artifacts["dmap_scales"] = _artifact(
            f"{prefix}_dmap_scales_f64.bin", dmap_ids, dmap_scales.shape
        )
        artifacts["cams_in_rig"] = _artifact(
            f"{prefix}_cams_in_rig_f64.bin", cam_ids, cams_in_rig.shape
        )
        dmap_shape = list(dmap_scales.shape)
    else:
        dmap_ids = []

    _write_json(
        snapshots_dir / f"{prefix}.json",
        {
            "schema_version": 1,
            "run_id": "run",
            "iteration": iteration,
            "dtype": "float64",
            "byte_order": "little_endian",
            "frame_ids": frame_ids,
            "frame_centers_world_shape": list(frame_centers.shape),
            "point3D_ids": point3D_ids,
            "points3D_world_shape": list(points3D.shape),
            "bata_residual_ids": [],
            "bata_scale_ids": scale_ids,
            "bata_scales_shape": list(scales.shape),
            "dmap_image_ids": dmap_ids,
            "dmap_scales_stored_shape": dmap_shape,
            "artifacts": artifacts,
        },
    )
    return trace_dir


def test_global_positioning_trace_loads_residual_values(tmp_path: Path) -> None:
    trace = pycolmap.GlobalPositioningTrace.load(
        _make_trace(tmp_path, with_jacobians=False)
    )

    assert trace.status == "finished"
    assert trace.trace_level == "residual_values"
    assert trace.residual_value_iterations == (0,)

    residual_values = trace.residual_values()
    assert isinstance(residual_values.raw_residuals, np.memmap)
    assert residual_values.has_raw_jacobians is False
    assert residual_values.raw_jacobians is None
    assert residual_values.has_loss_rho_values is False

    residual = residual_values.residual("r0")
    np.testing.assert_allclose(residual.raw_residuals, [1.0, 2.0])
    assert residual.raw_cost == 2.5
    assert residual.robust_cost == 2.25
    assert residual.parameter_blocks == ()
    assert residual.jacobian_blocks == ()
    with pytest.raises(ValueError, match="loss_rho_values artifact"):
        _ = residual_values.loss_rho_values
    with pytest.raises(ValueError, match="loss_rho_values artifact"):
        _ = residual.loss_rho


def test_global_positioning_trace_loads_loss_rho_values(
    tmp_path: Path,
) -> None:
    trace = pycolmap.GlobalPositioningTrace.load(
        _make_trace(tmp_path, with_jacobians=False, with_loss_rho=True)
    )

    residual_values = trace.residual_values()
    assert residual_values.has_loss_rho_values is True
    assert isinstance(residual_values.loss_rho_values, np.memmap)
    np.testing.assert_allclose(
        residual_values.loss_rho_values,
        [[4.5, 0.25, -0.125], [8.5, 1.0, 0.0]],
    )

    residual = residual_values.residual("r0")
    np.testing.assert_allclose(residual.loss_rho, [4.5, 0.25, -0.125])
    assert residual.loss_rho0 == 4.5
    assert residual.loss_rho1 == 0.25
    assert residual.loss_rho2 == -0.125
    assert residual.loss_derivative_scale == 0.25


def test_global_positioning_trace_rejects_bad_loss_rho_shape(
    tmp_path: Path,
) -> None:
    trace_dir = _make_trace(tmp_path, with_jacobians=False, with_loss_rho=True)
    metadata_path = trace_dir / "residual_values" / "iter_000000.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["artifacts"]["loss_rho_values"]["shape"] = [2, 2]
    _write_json(metadata_path, metadata)

    trace = pycolmap.GlobalPositioningTrace.load(trace_dir)
    with pytest.raises(ValueError, match="loss_rho_values.*shape"):
        trace.residual_values(0)


def test_global_positioning_trace_rejects_bad_loss_rho_layout(
    tmp_path: Path,
) -> None:
    trace_dir = _make_trace(tmp_path, with_jacobians=False, with_loss_rho=True)
    metadata_path = trace_dir / "residual_values" / "iter_000000.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["loss_rho_layout"] = "wrong"
    _write_json(metadata_path, metadata)

    trace = pycolmap.GlobalPositioningTrace.load(trace_dir)
    with pytest.raises(ValueError, match="loss_rho_layout"):
        trace.residual_values(0)


def test_global_positioning_trace_validates_loss_rho_costs_on_access(
    tmp_path: Path,
) -> None:
    trace_dir = _make_trace(tmp_path, with_jacobians=False, with_loss_rho=True)
    _write_f64(
        trace_dir / "residual_values" / "iter_000000_robust_costs_f64.bin",
        [999.0, 4.25],
    )

    residual_values = pycolmap.GlobalPositioningTrace.load(
        trace_dir
    ).residual_values(0)
    with pytest.raises(ValueError, match=r"0\.5 \* loss_rho_values"):
        _ = residual_values.loss_rho_values


def test_global_positioning_trace_allows_failed_loss_rho_nan_rows(
    tmp_path: Path,
) -> None:
    trace_dir = _make_trace(tmp_path, with_jacobians=False, with_loss_rho=True)
    metadata_path = trace_dir / "residual_values" / "iter_000000.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["evaluation_success"] = [False, True]
    _write_json(metadata_path, metadata)
    _write_f64(
        trace_dir / "residual_values" / "iter_000000_robust_costs_f64.bin",
        [float("nan"), 4.25],
    )
    _write_f64(
        trace_dir / "residual_values" / "iter_000000_loss_rho_values_f64.bin",
        [[float("nan"), float("nan"), float("nan")], [8.5, 1.0, 0.0]],
    )

    residual_values = pycolmap.GlobalPositioningTrace.load(
        trace_dir
    ).residual_values(0)

    assert residual_values.has_loss_rho_values is True
    np.testing.assert_allclose(
        residual_values.loss_rho_values[1],
        [8.5, 1.0, 0.0],
    )
    assert np.isnan(residual_values.loss_rho_values[0, 0])


def test_global_positioning_trace_loads_raw_jacobians(tmp_path: Path) -> None:
    trace = pycolmap.GlobalPositioningTrace.load(
        _make_trace(tmp_path, with_jacobians=True)
    )
    residual_values = trace.residual_values(iteration=0)

    assert residual_values.has_raw_jacobians is True
    assert isinstance(residual_values.raw_jacobians, np.memmap)

    residual = residual_values.residual(0)
    assert [block.role for block in residual.parameter_blocks] == [
        "frame_center",
        "bata_scale",
    ]
    assert residual.parameter_blocks[1].is_constant is True
    assert residual.parameter_blocks[1].lower_bounds == (1e-5,)

    np.testing.assert_allclose(
        residual.jacobian(0), np.arange(6, dtype=np.float64).reshape(2, 3)
    )
    np.testing.assert_allclose(residual.jacobian("bata_scale"), [[6.0], [7.0]])
    np.testing.assert_allclose(
        residual_values.residual("r1").jacobian("dmap_scale", id=20), [[8.0]]
    )


def test_global_positioning_trace_rejects_bad_sidecar_size(
    tmp_path: Path,
) -> None:
    trace_dir = _make_trace(tmp_path, with_jacobians=False)
    (
        trace_dir / "residual_values" / "iter_000000_raw_costs_f64.bin"
    ).write_bytes(b"short")

    trace = pycolmap.GlobalPositioningTrace.load(trace_dir)
    with np.testing.assert_raises(ValueError):
        trace.residual_values(0)


def test_global_positioning_trace_rejects_bad_residual_metadata(
    tmp_path: Path,
) -> None:
    trace_dir = _make_trace(tmp_path, with_jacobians=False)
    metadata_path = trace_dir / "residual_values" / "iter_000000.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["residual_ids"] = ["same", "same"]
    _write_json(metadata_path, metadata)

    trace = pycolmap.GlobalPositioningTrace.load(trace_dir)
    with pytest.raises(ValueError, match="residual_ids must be unique"):
        trace.residual_values(0)

    trace_dir = _make_trace(tmp_path / "offsets", with_jacobians=False)
    metadata_path = trace_dir / "residual_values" / "iter_000000.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["residual_offsets"] = [0, 0]
    _write_json(metadata_path, metadata)

    trace = pycolmap.GlobalPositioningTrace.load(trace_dir)
    with pytest.raises(ValueError, match="residual_offsets"):
        trace.residual_values(0)


def test_global_positioning_trace_rejects_residual_ledger_mismatch(
    tmp_path: Path,
) -> None:
    trace_dir = _make_trace(tmp_path, with_jacobians=False)
    metadata_path = trace_dir / "residual_values" / "iter_000000.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["residual_ids"] = ["r1", "r0"]
    _write_json(metadata_path, metadata)

    trace = pycolmap.GlobalPositioningTrace.load(trace_dir)
    with pytest.raises(ValueError, match="residual_blocks.jsonl order"):
        trace.residual_values(0)


def test_global_positioning_trace_rejects_bad_jacobian_metadata(
    tmp_path: Path,
) -> None:
    trace_dir = _make_trace(tmp_path, with_jacobians=True)
    metadata_path = trace_dir / "residual_values" / "iter_000000.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["raw_jacobian_offsets"] = [[0, 5], [8]]
    _write_json(metadata_path, metadata)

    trace = pycolmap.GlobalPositioningTrace.load(trace_dir)
    with pytest.raises(ValueError, match="raw_jacobian_offsets"):
        trace.residual_values(0)

    trace_dir = _make_trace(tmp_path / "bounds", with_jacobians=True)
    metadata_path = trace_dir / "residual_values" / "iter_000000.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["parameter_block_lower_bounds"][0][0] = [-1.0, -1.0]
    _write_json(metadata_path, metadata)

    trace = pycolmap.GlobalPositioningTrace.load(trace_dir)
    with pytest.raises(ValueError, match="parameter_block_lower_bounds"):
        trace.residual_values(0)


def test_global_positioning_trace_rejects_bad_jacobian_contract(
    tmp_path: Path,
) -> None:
    trace_dir = _make_trace(tmp_path, with_jacobians=True)
    metadata_path = trace_dir / "residual_values" / "iter_000000.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["raw_jacobian_layout"] = "wrong"
    _write_json(metadata_path, metadata)

    trace = pycolmap.GlobalPositioningTrace.load(trace_dir)
    with pytest.raises(ValueError, match="raw_jacobian_layout"):
        trace.residual_values(0)

    trace_dir = _make_trace(tmp_path / "domain", with_jacobians=True)
    metadata_path = trace_dir / "residual_values" / "iter_000000.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["jacobian_domain"] = "wrong"
    _write_json(metadata_path, metadata)

    trace = pycolmap.GlobalPositioningTrace.load(trace_dir)
    with pytest.raises(ValueError, match="jacobian_domain"):
        trace.residual_values(0)

    trace_dir = _make_trace(tmp_path / "loss", with_jacobians=True)
    metadata_path = trace_dir / "residual_values" / "iter_000000.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["loss_applied_to_jacobians"] = True
    _write_json(metadata_path, metadata)

    trace = pycolmap.GlobalPositioningTrace.load(trace_dir)
    with pytest.raises(ValueError, match="loss_applied_to_jacobians"):
        trace.residual_values(0)


def test_global_positioning_trace_rejects_residual_filename_iteration_mismatch(
    tmp_path: Path,
) -> None:
    trace_dir = _make_trace(tmp_path, with_jacobians=False)
    metadata_path = trace_dir / "residual_values" / "iter_000000.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["iteration"] = 1
    _write_json(metadata_path, metadata)

    with pytest.raises(
        ValueError, match="filename does not match metadata iteration"
    ):
        pycolmap.GlobalPositioningTrace.load(trace_dir)


def test_global_positioning_trace_rejects_residual_path_traversal(
    tmp_path: Path,
) -> None:
    trace_dir = _make_trace(tmp_path, with_jacobians=False)
    metadata_path = trace_dir / "residual_values" / "iter_000000.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["artifacts"]["raw_residuals"]["file"] = "../escape.bin"
    _write_json(metadata_path, metadata)

    trace = pycolmap.GlobalPositioningTrace.load(trace_dir)
    with pytest.raises(ValueError, match="bare relative filename"):
        trace.residual_values(0)


def test_global_positioning_trace_loads_snapshot_with_all_artifacts(
    tmp_path: Path,
) -> None:
    trace = pycolmap.GlobalPositioningTrace.load(
        _make_snapshot_trace(tmp_path, include_optional=True)
    )

    assert trace.snapshot_iterations == (3,)
    snapshot = trace.snapshot(3)

    assert snapshot.iteration == 3
    assert snapshot.frame_centers.ids == (10, 20)
    assert snapshot.frame_centers.shape == (2, 3)
    np.testing.assert_allclose(
        snapshot.frame_centers.values,
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    )
    assert isinstance(snapshot.points3D.values, np.memmap)
    assert snapshot.points3D.ids == (100, 101, 102, 103, 104)
    np.testing.assert_allclose(
        snapshot.points3D.values,
        np.arange(15, dtype=np.float64).reshape(5, 3),
    )
    assert snapshot.scales.ids == (30, 40)
    np.testing.assert_allclose(snapshot.scales.values, [0.5, 1.5])

    assert snapshot.dmap_scales is not None
    assert snapshot.dmap_scales.ids == (50, 60)
    np.testing.assert_allclose(snapshot.dmap_scales.values, [2.5, 3.5])
    assert snapshot.cams_in_rig is not None
    assert snapshot.cams_in_rig.ids == (70,)
    np.testing.assert_allclose(snapshot.cams_in_rig.values, [[0.1, 0.2, 0.3]])
    assert isinstance(snapshot, pycolmap.GlobalPositioningParameterSnapshot)
    assert isinstance(
        snapshot.points3D, pycolmap.GlobalPositioningSnapshotArray
    )


def test_global_positioning_trace_snapshot_optional_artifacts_missing(
    tmp_path: Path,
) -> None:
    trace = pycolmap.GlobalPositioningTrace.load(
        _make_snapshot_trace(tmp_path, include_optional=False)
    )

    snapshot = trace.snapshot(3)

    assert snapshot.dmap_scales is None
    assert snapshot.cams_in_rig is None
    np.testing.assert_allclose(snapshot.scales.values, [0.5, 1.5])


def test_global_positioning_trace_snapshot_accepts_legacy_missing_scales(
    tmp_path: Path,
) -> None:
    trace_dir = _make_snapshot_trace(tmp_path, include_optional=False)
    metadata_path = trace_dir / "snapshots" / "iter_000003.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    del metadata["bata_scale_ids"]
    del metadata["bata_scales_shape"]
    del metadata["artifacts"]["scales"]
    _write_json(metadata_path, metadata)

    snapshot = pycolmap.GlobalPositioningTrace.load(trace_dir).snapshot(3)

    assert snapshot.scales.ids == ()
    assert snapshot.scales.shape == (0,)
    np.testing.assert_allclose(snapshot.scales.values, [])


def test_global_positioning_trace_snapshot_max_points(tmp_path: Path) -> None:
    trace = pycolmap.GlobalPositioningTrace.load(
        _make_snapshot_trace(tmp_path, include_optional=True)
    )

    snapshot = trace.snapshot(3, max_points=2)

    assert snapshot.points3D.ids == (100, 101)
    assert snapshot.points3D.shape == (2, 3)
    assert isinstance(snapshot.points3D.values, np.memmap)
    np.testing.assert_allclose(
        snapshot.points3D.values,
        np.arange(6, dtype=np.float64).reshape(2, 3),
    )


def test_global_positioning_trace_snapshot_rejects_malformed_shape_and_size(
    tmp_path: Path,
) -> None:
    trace_dir = _make_snapshot_trace(tmp_path / "shape", include_optional=True)
    metadata_path = trace_dir / "snapshots" / "iter_000003.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["artifacts"]["points3D"]["shape"] = [4, 3]
    _write_json(metadata_path, metadata)

    trace = pycolmap.GlobalPositioningTrace.load(trace_dir)
    with pytest.raises(ValueError, match="points3D.*shape"):
        trace.snapshot(3)

    trace_dir = _make_snapshot_trace(tmp_path / "size", include_optional=True)
    (trace_dir / "snapshots" / "iter_000003_points3D_f64.bin").write_bytes(
        b"short"
    )

    trace = pycolmap.GlobalPositioningTrace.load(trace_dir)
    with pytest.raises(ValueError, match="byte size"):
        trace.snapshot(3)


def test_global_positioning_trace_snapshot_rejects_filename_iteration_mismatch(
    tmp_path: Path,
) -> None:
    trace_dir = _make_snapshot_trace(tmp_path, include_optional=True)
    metadata_path = trace_dir / "snapshots" / "iter_000003.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["iteration"] = 4
    _write_json(metadata_path, metadata)

    with pytest.raises(
        ValueError, match="filename does not match metadata iteration"
    ):
        pycolmap.GlobalPositioningTrace.load(trace_dir)
