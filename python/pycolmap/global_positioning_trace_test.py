import json
import struct
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


def _write_raw_string(stream, value: str) -> None:
    payload = value.encode("utf-8")
    stream.write(struct.pack("<I", len(payload)))
    stream.write(payload)


def _write_raw_json(stream, value: dict) -> None:
    _write_raw_string(stream, json.dumps(value, sort_keys=True, separators=(",", ":")))


def _write_raw_array(path: Path, name: str, ids, values) -> None:
    values = np.asarray(values, dtype="<f8")
    ids = np.asarray(ids, dtype="<i8")
    if values.ndim == 1:
        rows = values.shape[0]
        cols = 1
    else:
        rows, cols = values.shape
    assert rows == ids.shape[0]
    with path.open("wb") as stream:
        stream.write(b"GPTRARR1")
        stream.write(struct.pack("<IQQ", 1, rows, cols))
        _write_raw_string(stream, name)
        ids.tofile(stream)
        values.reshape(-1).tofile(stream)


def _write_raw_residual_ledger(path: Path, records: list[dict]) -> None:
    with path.open("wb") as stream:
        stream.write(b"GPTRLGR1")
        stream.write(struct.pack("<IQ", 1, len(records)))
        for record in records:
            attrs = record["attrs"]
            _write_raw_string(stream, attrs["residual_id"])
            _write_raw_string(stream, attrs["residual_type"])
            _write_raw_string(stream, attrs["loss_bucket"])
            stream.write(
                struct.pack(
                    "<qqq?",
                    -1 if attrs.get("frame_id") is None else attrs["frame_id"],
                    -1 if attrs.get("image_id") is None else attrs["image_id"],
                    -1 if attrs.get("point3D_id") is None else attrs["point3D_id"],
                    attrs.get("is_lc_observation", False),
                )
            )
            _write_raw_json(stream, attrs)


def _write_raw_residual_values(
    path: Path,
    *,
    iteration: int,
    residual_ids: list[str],
    residual_dims: list[int],
    evaluation_success: list[bool],
    raw_residuals,
    raw_costs,
    robust_costs,
    loss_rho_values=None,
    parameter_blocks: list[list[dict]] | None = None,
    raw_jacobians=None,
    force_version: int | None = None,
) -> None:
    residual_offsets = []
    offset = 0
    for dim in residual_dims:
        residual_offsets.append(offset)
        offset += dim
    raw_residuals = np.asarray(raw_residuals, dtype="<f8")
    raw_costs = np.asarray(raw_costs, dtype="<f8")
    robust_costs = np.asarray(robust_costs, dtype="<f8")
    has_raw_jacobians = parameter_blocks is not None
    if has_raw_jacobians:
        assert raw_jacobians is not None
        raw_jacobians = np.asarray(raw_jacobians, dtype="<f8")
    with path.open("wb") as stream:
        stream.write(b"GPTRRSV1")
        version = force_version or (2 if has_raw_jacobians else 1)
        if version == 2:
            stream.write(
                struct.pack(
                    "<IqQQ??",
                    2,
                    iteration,
                    len(residual_ids),
                    raw_residuals.size,
                    loss_rho_values is not None,
                    has_raw_jacobians,
                )
            )
        else:
            assert version == 1
            assert not has_raw_jacobians
            stream.write(
                struct.pack(
                    "<IqQQ?",
                    1,
                    iteration,
                    len(residual_ids),
                    raw_residuals.size,
                    loss_rho_values is not None,
                )
            )
        for residual_id, residual_dim, residual_offset, success in zip(
            residual_ids,
            residual_dims,
            residual_offsets,
            evaluation_success,
            strict=True,
        ):
            _write_raw_string(stream, residual_id)
            stream.write(struct.pack("<IQ?", residual_dim, residual_offset, success))
        raw_residuals.tofile(stream)
        raw_costs.tofile(stream)
        robust_costs.tofile(stream)
        if loss_rho_values is not None:
            np.asarray(loss_rho_values, dtype="<f8").reshape((-1, 3)).tofile(stream)
        if has_raw_jacobians:
            assert parameter_blocks is not None
            stream.write(struct.pack("<Q", raw_jacobians.size))
            expected_offset = 0
            for residual_idx, residual_blocks in enumerate(parameter_blocks):
                stream.write(struct.pack("<I", len(residual_blocks)))
                for block in residual_blocks:
                    block_size = int(block["size"])
                    _write_raw_string(stream, block["role"])
                    _write_raw_string(stream, block["kind"])
                    stream.write(
                        struct.pack(
                            "<QIQ?",
                            int(block["id"]),
                            block_size,
                            expected_offset,
                            bool(block.get("is_constant", False)),
                        )
                    )
                    np.asarray(block["lower_bounds"], dtype="<f8").tofile(stream)
                    expected_offset += residual_dims[residual_idx] * block_size
            assert expected_offset == raw_jacobians.size
            raw_jacobians.tofile(stream)


def _raw_support_records() -> list[dict]:
    return [
        {
            "event_type": "residual_added",
            "attrs": {
                "residual_id": "r10",
                "residual_type": "bata_ref_frame",
                "loss_bucket": "geometry_normal_inlier",
                "frame_id": 10,
                "image_id": 1010,
                "point3D_id": 100,
                "is_lc_observation": False,
            },
        },
        {
            "event_type": "residual_added",
            "attrs": {
                "residual_id": "r20",
                "residual_type": "bata_ref_frame",
                "loss_bucket": "geometry_normal_inlier",
                "frame_id": 20,
                "image_id": 1020,
                "point3D_id": 100,
                "is_lc_observation": False,
            },
        },
    ]


def _make_raw_binary_trace(
    tmp_path: Path,
    *,
    with_jacobians: bool = False,
    force_residual_version: int | None = None,
) -> Path:
    trace_dir = tmp_path / "trace_raw_binary"
    static_dir = trace_dir / "static"
    iteration_dir = trace_dir / "iterations" / "iter_000000"
    static_dir.mkdir(parents=True)
    iteration_dir.mkdir(parents=True)

    records = _raw_support_records()
    _write_raw_residual_ledger(static_dir / "residual_ledger.bin", records)
    _write_raw_array(
        iteration_dir / "frame_centers.bin",
        "frame_centers",
        [10, 20],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    )
    _write_raw_array(
        iteration_dir / "point_xyz.bin", "point_xyz", [100], [[0.0, 0.0, 2.0]]
    )
    _write_raw_array(iteration_dir / "scales.bin", "scales", [0], [1.0])
    parameter_blocks = None
    raw_jacobians = None
    if with_jacobians:
        parameter_blocks = [
            [
                {
                    "role": "frame_center",
                    "kind": "frame_center",
                    "id": 10,
                    "size": 3,
                    "is_constant": False,
                    "lower_bounds": [-np.inf, -np.inf, -np.inf],
                },
                {
                    "role": "bata_scale",
                    "kind": "bata_scale",
                    "id": 0,
                    "size": 1,
                    "is_constant": True,
                    "lower_bounds": [1e-5],
                },
            ],
            [
                {
                    "role": "dmap_scale",
                    "kind": "dmap_scale",
                    "id": 20,
                    "size": 1,
                    "is_constant": False,
                    "lower_bounds": [1e-5],
                }
            ],
        ]
        raw_jacobians = np.arange(9, dtype=np.float64)
    _write_raw_residual_values(
        iteration_dir / "residual_values.bin",
        iteration=0,
        residual_ids=["r10", "r20"],
        residual_dims=[2, 1],
        evaluation_success=[True, True],
        raw_residuals=[1.0, 2.0, 3.0],
        raw_costs=[2.5, 4.5],
        robust_costs=[2.25, 4.25],
        loss_rho_values=[[4.5, 0.25, -0.125], [8.5, 1.0, 0.0]],
        parameter_blocks=parameter_blocks,
        raw_jacobians=raw_jacobians,
        force_version=force_residual_version,
    )
    _write_json(
        trace_dir / "manifest.json",
        {
            "schema_version": 1,
            "storage_format": "global_positioning_raw_binary_v1",
            "run_id": "run",
            "status": "finished",
            "trace_level": "raw_binary_minimal",
            "byte_order": "little_endian",
            "dtype": "float64",
            "static": {"residual_ledger": "static/residual_ledger.bin"},
            "iterations": [
                {
                    "iteration": 0,
                    "directory": "iterations/iter_000000",
                    "frame_centers": "frame_centers.bin",
                    "point_xyz": "point_xyz.bin",
                    "scales": "scales.bin",
                    "residual_values": "residual_values.bin",
                }
            ],
        },
    )
    return trace_dir


def _artifact(filename: str, ids, shape) -> dict:
    return {
        "file": filename,
        "dtype": "float64",
        "byte_order": "little_endian",
        "ids": list(ids),
        "shape": list(shape),
    }


def _ledger_attrs(residual_id: str) -> dict:
    return {
        "residual_id": residual_id,
        "replay_schema_version": 1,
        "parameter_blocks": [
            {
                "role": "frame_center",
                "kind": "frame_center",
                "id": 10,
                "size": 3,
            },
            {
                "role": "point3D",
                "kind": "point3D",
                "id": 20,
                "size": 3,
            },
        ],
        "loss": {
            "bucket": "track",
            "type": "cauchy",
            "scale": 1.5,
            "weight": None,
            "source": "config",
            "observation_count_weight": 0.25,
        },
        "fixed_parameters_status": "serialized",
        "fixed_parameters": {
            "image_id": 7,
            "point2D_idx": 3,
            "is_lc_observation": False,
        },
    }


def _trace_record(
    event_type: str,
    *,
    seq: int,
    stage: str = "ceres_solve",
    iteration: int | None = None,
    attrs: dict | None = None,
) -> dict:
    return {
        "schema_version": 1,
        "run_id": "run",
        "seq": seq,
        "event_type": event_type,
        "stage": stage,
        "iteration": iteration,
        "timestamp_ns": 1000 + seq,
        "attrs": attrs or {},
    }


def _iteration_metric(
    iteration: int,
    *,
    seq: int,
    attrs: dict | None = None,
) -> dict:
    metric_attrs = {
        "step_is_successful": True,
        "cost": 42.0,
        "cost_change": -3.5,
        "gradient_max_norm": 0.25,
        "step_norm": 1.5,
        "trust_region_radius": 10.0,
        "linear_solver_iterations": 7,
        "iteration_time_sec": 0.125,
        "cumulative_time_sec": 0.5,
    }
    if attrs:
        metric_attrs.update(attrs)
    return _trace_record(
        "ceres_iteration",
        seq=seq,
        iteration=iteration,
        attrs=metric_attrs,
    )


def _make_trace(
    tmp_path: Path, *, with_jacobians: bool, with_loss_rho: bool = False
) -> Path:
    trace_dir = tmp_path / ("trace_jacobians" if with_jacobians else "trace_values")
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
    _write_f64(residual_values_dir / "iter_000000_raw_costs_f64.bin", [2.5, 4.5])
    _write_f64(residual_values_dir / "iter_000000_robust_costs_f64.bin", [2.25, 4.25])
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


def _make_snapshot_trace(tmp_path: Path, *, include_optional: bool = True) -> Path:
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
    frame_centers = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
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
        "scales": _artifact(f"{prefix}_scales_f64.bin", scale_ids, scales.shape),
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


def _replay_parameter_block(role: str, kind: str, block_id: int, size: int) -> dict:
    return {"role": role, "kind": kind, "id": block_id, "size": size}


def _replay_loss(loss_type: str = "trivial", scale: float = 1.0, weight=1.0) -> dict:
    return {
        "bucket": "test",
        "type": loss_type,
        "scale": scale,
        "weight": weight,
        "source": "test",
    }


def _replay_attrs(
    residual_id: str,
    residual_type: str,
    parameter_blocks: list[dict],
    fixed_parameters: dict,
    *,
    loss: dict | None = None,
    extra: dict | None = None,
) -> dict:
    attrs = {
        "residual_id": residual_id,
        "residual_type": residual_type,
        "replay_schema_version": 1,
        "parameter_blocks": parameter_blocks,
        "loss": loss or _replay_loss(),
        "fixed_parameters_status": "serialized",
        "fixed_parameters": fixed_parameters,
    }
    if extra:
        attrs.update(extra)
    return attrs


def _make_replay_trace(tmp_path: Path, records: list[dict] | None = None) -> Path:
    trace_dir = tmp_path / "trace_replay"
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
    frame_centers = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    points3D = np.array([[4.0, 4.0, 6.0]], dtype=np.float64)
    scales = np.array([0.5], dtype=np.float64)
    dmap_scales = np.array([2.0], dtype=np.float64)
    cams_in_rig = np.array([[0.1, 0.2, 0.3]], dtype=np.float64)

    _write_f64(snapshots_dir / f"{prefix}_frame_centers_f64.bin", frame_centers)
    _write_f64(snapshots_dir / f"{prefix}_points3D_f64.bin", points3D)
    _write_f64(snapshots_dir / f"{prefix}_scales_f64.bin", scales)
    _write_f64(snapshots_dir / f"{prefix}_dmap_scales_f64.bin", dmap_scales)
    _write_f64(snapshots_dir / f"{prefix}_cams_in_rig_f64.bin", cams_in_rig)
    _write_json(
        snapshots_dir / f"{prefix}.json",
        {
            "schema_version": 1,
            "run_id": "run",
            "iteration": iteration,
            "dtype": "float64",
            "byte_order": "little_endian",
            "frame_ids": [10],
            "frame_centers_world_shape": list(frame_centers.shape),
            "point3D_ids": [20],
            "points3D_world_shape": list(points3D.shape),
            "bata_residual_ids": [],
            "bata_scale_ids": [30],
            "bata_scales_shape": list(scales.shape),
            "dmap_image_ids": [50],
            "dmap_scales_stored_shape": list(dmap_scales.shape),
            "artifacts": {
                "frame_centers": _artifact(
                    f"{prefix}_frame_centers_f64.bin", [10], frame_centers.shape
                ),
                "points3D": _artifact(
                    f"{prefix}_points3D_f64.bin", [20], points3D.shape
                ),
                "scales": _artifact(f"{prefix}_scales_f64.bin", [30], scales.shape),
                "dmap_scales": _artifact(
                    f"{prefix}_dmap_scales_f64.bin", [50], dmap_scales.shape
                ),
                "cams_in_rig": _artifact(
                    f"{prefix}_cams_in_rig_f64.bin", [70], cams_in_rig.shape
                ),
            },
        },
    )

    common_bata_blocks = [
        _replay_parameter_block("frame_center", "frame_center", 10, 3),
        _replay_parameter_block("point3D", "point3D", 20, 3),
        _replay_parameter_block("bata_scale", "bata_scale", 30, 1),
    ]
    if records is None:
        records = [
            {
                "event_type": "residual_added",
                "attrs": _replay_attrs(
                    "bata_ref",
                    "bata_ref_frame",
                    common_bata_blocks,
                    {"cam_from_point3D_dir": [2.0, 0.0, 1.0]},
                    loss=_replay_loss("cauchy", 2.0, 1.0),
                ),
            },
            {
                "event_type": "residual_added",
                "attrs": _replay_attrs(
                    "bata_const",
                    "bata_constant_rig",
                    [
                        _replay_parameter_block("point3D", "point3D", 20, 3),
                        _replay_parameter_block("frame_center", "frame_center", 10, 3),
                        _replay_parameter_block("bata_scale", "bata_scale", 30, 1),
                    ],
                    {
                        "cam_from_point3D_dir": [1.0, 1.0, 1.0],
                        "cam_from_rig_dir": [0.1, 0.2, 0.3],
                    },
                ),
            },
            {
                "event_type": "residual_added",
                "attrs": _replay_attrs(
                    "bata_var",
                    "bata_variable_rig",
                    [
                        _replay_parameter_block("point3D", "point3D", 20, 3),
                        _replay_parameter_block("frame_center", "frame_center", 10, 3),
                        _replay_parameter_block("cam_in_rig", "cam_in_rig", 70, 3),
                        _replay_parameter_block("bata_scale", "bata_scale", 30, 1),
                    ],
                    {
                        "cam_from_point3D_dir": [1.0, 1.0, 1.0],
                        "world_from_rig_rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
                    },
                ),
            },
            {
                "event_type": "residual_added",
                "attrs": _replay_attrs(
                    "metric",
                    "metric_depth",
                    [
                        _replay_parameter_block("frame_center", "frame_center", 10, 3),
                        _replay_parameter_block("point3D", "point3D", 20, 3),
                        _replay_parameter_block("dmap_scale", "dmap_scale", 50, 1),
                    ],
                    {
                        "camera_rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
                        "metric_depth_use_log_scale": False,
                        "metric_depth_residual_type": "linear",
                        "metric_depth_zero_residual_behind": False,
                        "metric_depth_log_linear_threshold": 0.1,
                    },
                    extra={"depth_prior": 1.0, "depth_sigma": 0.5},
                ),
            },
            {
                "event_type": "residual_added",
                "attrs": _replay_attrs(
                    "scale_prior",
                    "scale_prior",
                    [_replay_parameter_block("dmap_scale", "dmap_scale", 50, 1)],
                    {"scale_prior_target": 1.0, "scale_prior_stddev": 0.5},
                    loss=_replay_loss("trivial", 1.0, 3.0),
                ),
            },
        ]
    _write_jsonl(trace_dir / "residual_blocks.jsonl", records)
    return trace_dir


def _write_replay_residual_dump(trace_dir: Path) -> None:
    residual_values_dir = trace_dir / "residual_values"
    residual_values_dir.mkdir()
    iteration = 3
    prefix = f"iter_{iteration:06d}"

    residual_ids = [
        "bata_ref",
        "bata_const",
        "bata_var",
        "metric",
        "scale_prior",
    ]
    residual_dims = [3, 3, 3, 1, 1]
    residual_offsets = [0, 3, 6, 9, 10]
    raw_residuals = np.array(
        [
            0.5,
            -1.0,
            -0.5,
            -0.55,
            -0.1,
            -0.65,
            -0.45,
            0.1,
            -0.35,
            1.0,
            2.0,
        ],
        dtype=np.float64,
    )
    raw_costs = np.array([0.75, 0.3675, 0.1675, 0.5, 2.0], dtype=np.float64)
    cauchy_rho0 = 4.0 * np.log1p(1.5 / 4.0)
    loss_rho_values = np.array(
        [
            [cauchy_rho0, 1.0 / 1.375, -1.0 / (4.0 * 1.375**2)],
            [0.735, 1.0, 0.0],
            [0.335, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [12.0, 3.0, 0.0],
        ],
        dtype=np.float64,
    )
    robust_costs = 0.5 * loss_rho_values[:, 0]

    jacobian_blocks = [
        [
            0.5 * np.eye(3),
            -0.5 * np.eye(3),
            np.array([[-3.0], [-2.0], [-3.0]], dtype=np.float64),
        ],
        [
            -0.5 * np.eye(3),
            0.5 * np.eye(3),
            np.array([[-3.1], [-2.2], [-3.3]], dtype=np.float64),
        ],
        [
            -0.5 * np.eye(3),
            0.5 * np.eye(3),
            0.5 * np.eye(3),
            np.array([[-2.9], [-1.8], [-2.7]], dtype=np.float64),
        ],
        [
            np.array([[0.0, 0.0, -1.0]], dtype=np.float64),
            np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
            np.array([[-1.5]], dtype=np.float64),
        ],
        [np.array([[2.0]], dtype=np.float64)],
    ]
    raw_jacobian_offsets = []
    raw_jacobian_parts = []
    jacobian_offset = 0
    for residual_blocks in jacobian_blocks:
        residual_offsets_for_blocks = []
        for block in residual_blocks:
            residual_offsets_for_blocks.append(jacobian_offset)
            raw_jacobian_parts.append(block.ravel())
            jacobian_offset += block.size
        raw_jacobian_offsets.append(residual_offsets_for_blocks)
    raw_jacobians = np.concatenate(raw_jacobian_parts)

    parameter_blocks = [
        [
            {"role": "frame_center", "kind": "frame_center", "id": 10},
            {"role": "point3D", "kind": "point3D", "id": 20},
            {"role": "bata_scale", "kind": "bata_scale", "id": 30},
        ],
        [
            {"role": "point3D", "kind": "point3D", "id": 20},
            {"role": "frame_center", "kind": "frame_center", "id": 10},
            {"role": "bata_scale", "kind": "bata_scale", "id": 30},
        ],
        [
            {"role": "point3D", "kind": "point3D", "id": 20},
            {"role": "frame_center", "kind": "frame_center", "id": 10},
            {"role": "cam_in_rig", "kind": "cam_in_rig", "id": 70},
            {"role": "bata_scale", "kind": "bata_scale", "id": 30},
        ],
        [
            {"role": "frame_center", "kind": "frame_center", "id": 10},
            {"role": "point3D", "kind": "point3D", "id": 20},
            {"role": "dmap_scale", "kind": "dmap_scale", "id": 50},
        ],
        [{"role": "dmap_scale", "kind": "dmap_scale", "id": 50}],
    ]
    parameter_block_sizes = [
        [block.shape[1] for block in residual_blocks]
        for residual_blocks in jacobian_blocks
    ]
    parameter_block_is_constant = [
        [False] * len(residual_blocks) for residual_blocks in jacobian_blocks
    ]
    parameter_block_lower_bounds = []
    for residual_blocks in jacobian_blocks:
        parameter_block_lower_bounds.append(
            [[-1e100] * block.shape[1] for block in residual_blocks]
        )

    _write_json(
        residual_values_dir / f"{prefix}.json",
        {
            "schema_version": 1,
            "run_id": "run",
            "iteration": iteration,
            "dtype": "float64",
            "byte_order": "little_endian",
            "num_residual_blocks": len(residual_ids),
            "total_scalar_residuals": int(raw_residuals.size),
            "has_raw_jacobians": True,
            "residual_ids": residual_ids,
            "residual_dims": residual_dims,
            "residual_offsets": residual_offsets,
            "evaluation_success": [True] * len(residual_ids),
            "total_jacobian_scalars": int(raw_jacobians.size),
            "parameter_block_sizes": parameter_block_sizes,
            "raw_jacobian_offsets": raw_jacobian_offsets,
            "parameter_blocks": parameter_blocks,
            "parameter_block_is_constant": parameter_block_is_constant,
            "parameter_block_lower_bounds": parameter_block_lower_bounds,
            "raw_jacobian_layout": "residual_block_major/parameter_block_major/row_major",
            "jacobian_domain": "raw_cost_function_ambient_parameters",
            "loss_applied_to_jacobians": False,
            "manifold_applied_to_jacobians": False,
            "constant_parameter_blocks_included": True,
            "loss_rho_layout": "residual_block_major/rho0_rho1_rho2",
            "artifacts": {
                "raw_residuals": {
                    "file": f"{prefix}_raw_residuals_f64.bin",
                    "dtype": "float64",
                    "byte_order": "little_endian",
                    "shape": list(raw_residuals.shape),
                },
                "raw_costs": {
                    "file": f"{prefix}_raw_costs_f64.bin",
                    "dtype": "float64",
                    "byte_order": "little_endian",
                    "shape": list(raw_costs.shape),
                },
                "robust_costs": {
                    "file": f"{prefix}_robust_costs_f64.bin",
                    "dtype": "float64",
                    "byte_order": "little_endian",
                    "shape": list(robust_costs.shape),
                },
                "loss_rho_values": {
                    "file": f"{prefix}_loss_rho_values_f64.bin",
                    "dtype": "float64",
                    "byte_order": "little_endian",
                    "shape": list(loss_rho_values.shape),
                },
                "raw_jacobians": {
                    "file": f"{prefix}_raw_jacobians_f64.bin",
                    "dtype": "float64",
                    "byte_order": "little_endian",
                    "shape": list(raw_jacobians.shape),
                },
            },
        },
    )
    _write_f64(residual_values_dir / f"{prefix}_raw_residuals_f64.bin", raw_residuals)
    _write_f64(residual_values_dir / f"{prefix}_raw_costs_f64.bin", raw_costs)
    _write_f64(residual_values_dir / f"{prefix}_robust_costs_f64.bin", robust_costs)
    _write_f64(
        residual_values_dir / f"{prefix}_loss_rho_values_f64.bin", loss_rho_values
    )
    _write_f64(residual_values_dir / f"{prefix}_raw_jacobians_f64.bin", raw_jacobians)


def _make_replay_trace_with_residual_dump(tmp_path: Path) -> Path:
    trace_dir = _make_replay_trace(tmp_path)
    _write_replay_residual_dump(trace_dir)
    return trace_dir


def _flatten_replay_jacobians(
    replay: pycolmap.GlobalPositioningReplayEvaluation,
) -> np.ndarray:
    return np.concatenate(
        [
            block.values.ravel()
            for residual_blocks in replay.raw_jacobians
            for block in residual_blocks
        ]
    )


def test_global_positioning_trace_loads_residual_values(tmp_path: Path) -> None:
    trace = pycolmap.GlobalPositioningTrace.load(
        _make_trace(tmp_path, with_jacobians=False)
    )

    assert trace.status == "finished"
    assert trace.trace_level == "residual_values"
    assert trace.residual_value_iterations == (0,)
    assert trace.events == ()
    assert trace.solver_events == ()
    assert trace.iteration_metrics == ()
    assert trace.iteration_metrics_by_iteration == {}

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


def test_global_positioning_trace_loads_events_and_iteration_metrics(
    tmp_path: Path,
) -> None:
    trace_dir = _make_trace(tmp_path, with_jacobians=False)
    _write_jsonl(
        trace_dir / "events.jsonl",
        [
            _trace_record(
                "solve_started",
                seq=0,
                iteration=None,
                attrs={"linear_solver_type": "SPARSE_SCHUR"},
            ),
            _trace_record(
                "solve_finished",
                seq=1,
                iteration=None,
                attrs={
                    "termination_type": "CONVERGENCE",
                    "message": "done",
                    "final_cost": 12.5,
                },
            ),
        ],
    )
    _write_jsonl(
        trace_dir / "iteration_metrics.jsonl",
        [
            _iteration_metric(0, seq=2, attrs={"gradient_norm": 0.75}),
            _iteration_metric(
                1,
                seq=3,
                attrs={
                    "step_is_successful": False,
                    "cost": 38.0,
                    "linear_solver_iterations": 9,
                },
            ),
        ],
    )

    trace = pycolmap.GlobalPositioningTrace.load(trace_dir)

    assert [event.event_type for event in trace.events] == [
        "solve_started",
        "solve_finished",
    ]
    assert trace.solver_events == trace.events
    assert trace.events[0].attrs["linear_solver_type"] == "SPARSE_SCHUR"
    assert trace.events[0].iteration is None
    assert trace.events[1].attrs["termination_type"] == "CONVERGENCE"
    assert [metric.iteration for metric in trace.iteration_metrics] == [0, 1]
    first = trace.iteration_metrics[0]
    assert first.event_type == "ceres_iteration"
    assert first.step_is_successful is True
    assert first.cost == pytest.approx(42.0)
    assert first.cost_change == pytest.approx(-3.5)
    assert first.gradient_norm == pytest.approx(0.75)
    assert first.gradient_max_norm == pytest.approx(0.25)
    assert first.step_norm == pytest.approx(1.5)
    assert first.trust_region_radius == pytest.approx(10.0)
    assert first.linear_solver_iterations == 7
    assert first.iteration_time_sec == pytest.approx(0.125)
    assert first.cumulative_time_sec == pytest.approx(0.5)
    assert trace.iteration_metrics[1].step_is_successful is False
    assert trace.iteration_metrics_by_iteration[1].linear_solver_iterations == 9


def test_global_positioning_trace_rejects_bad_event_record(
    tmp_path: Path,
) -> None:
    trace_dir = _make_trace(tmp_path, with_jacobians=False)
    record = _trace_record("solve_started", seq=0)
    record["schema_version"] = 2
    _write_jsonl(trace_dir / "events.jsonl", [record])

    with pytest.raises(ValueError, match="unsupported schema_version"):
        pycolmap.GlobalPositioningTrace.load(trace_dir)


def test_global_positioning_trace_rejects_bad_iteration_metric(
    tmp_path: Path,
) -> None:
    trace_dir = _make_trace(tmp_path, with_jacobians=False)
    record = _iteration_metric(0, seq=0)
    del record["attrs"]["cost"]
    _write_jsonl(trace_dir / "iteration_metrics.jsonl", [record])

    with pytest.raises(KeyError, match="cost"):
        pycolmap.GlobalPositioningTrace.load(trace_dir)


def test_global_positioning_trace_rejects_duplicate_iteration_metrics(
    tmp_path: Path,
) -> None:
    trace_dir = _make_trace(tmp_path, with_jacobians=False)
    _write_jsonl(
        trace_dir / "iteration_metrics.jsonl",
        [_iteration_metric(0, seq=0), _iteration_metric(0, seq=1)],
    )

    with pytest.raises(ValueError, match="duplicate iteration metric 0"):
        pycolmap.GlobalPositioningTrace.load(trace_dir)


def test_global_positioning_trace_rejects_non_iteration_metric_event(
    tmp_path: Path,
) -> None:
    trace_dir = _make_trace(tmp_path, with_jacobians=False)
    _write_jsonl(
        trace_dir / "iteration_metrics.jsonl",
        [_trace_record("solve_started", seq=0, iteration=None)],
    )

    with pytest.raises(ValueError, match="ceres_iteration"):
        pycolmap.GlobalPositioningTrace.load(trace_dir)


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

    residual_values = pycolmap.GlobalPositioningTrace.load(trace_dir).residual_values(0)
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

    residual_values = pycolmap.GlobalPositioningTrace.load(trace_dir).residual_values(0)

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
    (trace_dir / "residual_values" / "iter_000000_raw_costs_f64.bin").write_bytes(
        b"short"
    )

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


def test_global_positioning_trace_loads_residual_ledger_blocks(
    tmp_path: Path,
) -> None:
    trace_dir = _make_trace(tmp_path, with_jacobians=False)
    _write_jsonl(
        trace_dir / "residual_blocks.jsonl",
        [
            {
                "event_type": "residual_added",
                "attrs": _ledger_attrs("r0"),
            },
            {
                "event_type": "residual_added",
                "attrs": _ledger_attrs("r1"),
            },
        ],
    )

    trace = pycolmap.GlobalPositioningTrace.load(trace_dir)
    ledger_blocks = trace.residual_ledger_blocks

    assert len(ledger_blocks) == 2
    assert trace.residual_blocks[0]["attrs"]["replay_schema_version"] == 1
    first = ledger_blocks[0]
    assert isinstance(first, pycolmap.GlobalPositioningResidualLedgerBlock)
    assert first.residual_id == "r0"
    assert first.event_type == "residual_added"
    assert first.replay_schema_version == 1
    assert [block.role for block in first.parameter_blocks] == [
        "frame_center",
        "point3D",
    ]
    assert first.parameter_blocks[0].size == 3
    assert isinstance(
        first.parameter_blocks[0],
        pycolmap.GlobalPositioningResidualLedgerParameterBlock,
    )
    assert first.loss.bucket == "track"
    assert isinstance(first.loss, pycolmap.GlobalPositioningResidualLedgerLoss)
    assert first.loss.type == "cauchy"
    assert first.loss.scale == 1.5
    assert first.loss.weight is None
    assert first.loss.observation_count_weight == 0.25
    assert first.fixed_parameters_status == "serialized"
    assert first.fixed_parameters["image_id"] == 7


def test_global_positioning_trace_rejects_deferred_residual_ledger_block(
    tmp_path: Path,
) -> None:
    trace_dir = _make_trace(tmp_path, with_jacobians=False)
    attrs = _ledger_attrs("r0")
    attrs["fixed_parameters_status"] = "deferred_not_serialized"
    del attrs["fixed_parameters"]
    attrs["fixed_parameters_todo"] = "GP_REPLAY_FIXED_PARAMETERS_track"
    _write_jsonl(
        trace_dir / "residual_blocks.jsonl",
        [{"event_type": "residual_added", "attrs": attrs}],
    )

    trace = pycolmap.GlobalPositioningTrace.load(trace_dir)
    assert trace.residual_blocks[0]["attrs"]["fixed_parameters_status"] == (
        "deferred_not_serialized"
    )
    with pytest.raises(ValueError, match="fixed_parameters_status"):
        _ = trace.residual_ledger_blocks


def test_global_positioning_trace_rejects_bad_ledger_parameter_block_size(
    tmp_path: Path,
) -> None:
    trace_dir = _make_trace(tmp_path, with_jacobians=False)
    attrs = _ledger_attrs("r0")
    attrs["parameter_blocks"][0]["size"] = 0
    _write_jsonl(
        trace_dir / "residual_blocks.jsonl",
        [{"event_type": "residual_added", "attrs": attrs}],
    )

    trace = pycolmap.GlobalPositioningTrace.load(trace_dir)
    with pytest.raises(ValueError, match=r"parameter_blocks\[0\].*size"):
        _ = trace.residual_ledger_blocks


def test_global_positioning_trace_rejects_unsupported_residual_ledger_schema(
    tmp_path: Path,
) -> None:
    trace_dir = _make_trace(tmp_path, with_jacobians=False)
    attrs = _ledger_attrs("r0")
    attrs["replay_schema_version"] = 2
    _write_jsonl(
        trace_dir / "residual_blocks.jsonl",
        [{"event_type": "residual_added", "attrs": attrs}],
    )

    trace = pycolmap.GlobalPositioningTrace.load(trace_dir)
    with pytest.raises(ValueError, match="unsupported replay_schema_version"):
        _ = trace.residual_ledger_blocks


def test_global_positioning_trace_rejects_bad_ledger_fixed_parameters(
    tmp_path: Path,
) -> None:
    trace_dir = _make_trace(tmp_path, with_jacobians=False)
    attrs = _ledger_attrs("r0")
    attrs["fixed_parameters"] = ["not", "an", "object"]
    _write_jsonl(
        trace_dir / "residual_blocks.jsonl",
        [{"event_type": "residual_added", "attrs": attrs}],
    )

    trace = pycolmap.GlobalPositioningTrace.load(trace_dir)
    with pytest.raises(TypeError, match="fixed_parameters must be an object"):
        _ = trace.residual_ledger_blocks


def test_global_positioning_trace_ignores_legacy_residual_ledger_rows(
    tmp_path: Path,
) -> None:
    trace = pycolmap.GlobalPositioningTrace.load(
        _make_trace(tmp_path, with_jacobians=False)
    )

    assert [record["attrs"]["residual_id"] for record in trace.residual_blocks] == [
        "r0",
        "r1",
    ]
    assert trace.residual_ledger_blocks == ()
    assert trace.residual_values(0).residual_ids == ["r0", "r1"]


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

    with pytest.raises(ValueError, match="filename does not match metadata iteration"):
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
    assert isinstance(snapshot.points3D, pycolmap.GlobalPositioningSnapshotArray)


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
    (trace_dir / "snapshots" / "iter_000003_points3D_f64.bin").write_bytes(b"short")

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

    with pytest.raises(ValueError, match="filename does not match metadata iteration"):
        pycolmap.GlobalPositioningTrace.load(trace_dir)


def test_global_positioning_trace_replay_evaluates_known_residual_families(
    tmp_path: Path,
) -> None:
    trace = pycolmap.GlobalPositioningTrace.load(_make_replay_trace(tmp_path))

    replay = trace.replay(iteration=3)

    assert isinstance(replay, pycolmap.GlobalPositioningReplayEvaluation)
    assert replay.residual_ids == (
        "bata_ref",
        "bata_const",
        "bata_var",
        "metric",
        "scale_prior",
    )
    assert replay.residual_dims == (3, 3, 3, 1, 1)
    assert replay.residual_offsets == (0, 3, 6, 9, 10)
    np.testing.assert_allclose(
        replay.raw_residuals,
        [
            0.5,
            -1.0,
            -0.5,
            -0.55,
            -0.1,
            -0.65,
            -0.45,
            0.1,
            -0.35,
            1.0,
            2.0,
        ],
    )
    np.testing.assert_allclose(
        replay.raw_costs,
        [0.75, 0.3675, 0.1675, 0.5, 2.0],
    )
    np.testing.assert_allclose(replay.evaluation_success, [True] * 5)
    assert [block.role for block in replay.parameter_blocks[0]] == [
        "frame_center",
        "point3D",
        "bata_scale",
    ]

    cauchy_rho0 = 4.0 * np.log1p(1.5 / 4.0)
    np.testing.assert_allclose(
        replay.loss_rho_values[0],
        [cauchy_rho0, 1.0 / 1.375, -1.0 / (4.0 * 1.375**2)],
    )
    assert replay.robust_costs[0] == pytest.approx(0.5 * cauchy_rho0)
    assert replay.robust_costs[-1] == pytest.approx(6.0)
    np.testing.assert_allclose(
        replay.residual("metric").raw_residuals,
        [1.0],
    )


def test_global_positioning_trace_replay_exposes_residual_values_compatibility(
    tmp_path: Path,
) -> None:
    trace = pycolmap.GlobalPositioningTrace.load(_make_replay_trace(tmp_path))

    replay = trace.replay(iteration=3)

    assert replay.has_raw_jacobians is False
    assert replay.has_loss_rho_values is True
    residual = replay.residual("bata_ref")
    assert residual.residual_id == "bata_ref"
    assert residual.residual_dim == 3
    assert residual.residual_offset == 0
    assert residual.evaluation_success is True
    np.testing.assert_allclose(residual.raw_residuals, [0.5, -1.0, -0.5])
    assert residual.raw_cost == pytest.approx(0.75)
    assert residual.robust_cost == pytest.approx(replay.robust_costs[0])
    np.testing.assert_allclose(residual.loss_rho, replay.loss_rho_values[0])
    assert residual.loss_rho0 == pytest.approx(replay.loss_rho_values[0, 0])
    assert residual.loss_rho1 == pytest.approx(replay.loss_rho_values[0, 1])
    assert residual.loss_rho2 == pytest.approx(replay.loss_rho_values[0, 2])
    assert residual.loss_derivative_scale == pytest.approx(residual.loss_rho1)
    assert [block.role for block in residual.parameter_blocks] == [
        "frame_center",
        "point3D",
        "bata_scale",
    ]
    assert residual.jacobian_blocks == ()

    replay_with_jacobians = trace.replay(iteration=3, compute_jacobians=True)

    assert replay_with_jacobians.has_raw_jacobians is True
    assert replay_with_jacobians.has_loss_rho_values is True
    residual_with_jacobians = replay_with_jacobians.residual("bata_ref")
    assert residual_with_jacobians.raw_cost == pytest.approx(residual.raw_cost)
    np.testing.assert_allclose(residual_with_jacobians.loss_rho, residual.loss_rho)
    assert len(residual_with_jacobians.jacobian_blocks) == len(
        residual_with_jacobians.parameter_blocks
    )
    jacobian_block = residual_with_jacobians.jacobian_blocks[0]
    assert jacobian_block.parameter_block.role == "frame_center"
    assert jacobian_block.parameter_block.kind == "frame_center"
    assert jacobian_block.parameter_block.id == 10
    assert jacobian_block.parameter_block.size == 3
    assert jacobian_block.offset == 0
    assert jacobian_block.residual_dim == residual_with_jacobians.residual_dim
    assert jacobian_block.values.shape == (residual_with_jacobians.residual_dim, 3)


def test_global_positioning_trace_replay_matches_synthetic_on_disk_residual_dump(
    tmp_path: Path,
) -> None:
    """Synthetic on-disk parity, not a true C++-generated golden trace."""
    trace = pycolmap.GlobalPositioningTrace.load(
        _make_replay_trace_with_residual_dump(tmp_path)
    )
    dumped = trace.residual_values(iteration=3)

    replay = trace.replay(iteration=3)

    assert replay.residual_ids == tuple(dumped.residual_ids)
    assert replay.residual_dims == tuple(dumped.residual_dims)
    assert replay.residual_offsets == tuple(dumped.residual_offsets)
    assert replay.evaluation_success == tuple(dumped.evaluation_success)
    np.testing.assert_allclose(replay.raw_residuals, dumped.raw_residuals)
    np.testing.assert_allclose(replay.raw_costs, dumped.raw_costs)
    np.testing.assert_allclose(replay.robust_costs, dumped.robust_costs)
    np.testing.assert_allclose(replay.loss_rho_values, dumped.loss_rho_values)

    for residual_id in replay.residual_ids:
        replay_residual = replay.residual(residual_id)
        dumped_residual = dumped.residual(residual_id)
        assert replay_residual.residual_id == dumped_residual.residual_id
        assert replay_residual.residual_dim == dumped_residual.residual_dim
        assert replay_residual.residual_offset == dumped_residual.residual_offset
        assert replay_residual.evaluation_success == dumped_residual.evaluation_success
        np.testing.assert_allclose(
            replay_residual.raw_residuals, dumped_residual.raw_residuals
        )
        assert replay_residual.raw_cost == pytest.approx(dumped_residual.raw_cost)
        assert replay_residual.robust_cost == pytest.approx(dumped_residual.robust_cost)
        np.testing.assert_allclose(replay_residual.loss_rho, dumped_residual.loss_rho)

    replay_with_jacobians = trace.replay(iteration=3, compute_jacobians=True)
    assert dumped.raw_jacobians is not None
    np.testing.assert_allclose(
        _flatten_replay_jacobians(replay_with_jacobians),
        dumped.raw_jacobians,
        rtol=1e-8,
        atol=1e-10,
    )

    for residual_id in replay_with_jacobians.residual_ids:
        replay_residual = replay_with_jacobians.residual(residual_id)
        dumped_residual = dumped.residual(residual_id)
        assert len(replay_residual.jacobian_blocks) == len(
            dumped_residual.jacobian_blocks
        )
        for replay_block, dumped_block in zip(
            replay_residual.jacobian_blocks,
            dumped_residual.jacobian_blocks,
            strict=True,
        ):
            assert replay_block.offset == dumped_block.offset
            assert replay_block.residual_dim == dumped_block.residual_dim
            assert (
                replay_block.parameter_block.role == dumped_block.parameter_block.role
            )
            assert (
                replay_block.parameter_block.kind == dumped_block.parameter_block.kind
            )
            assert replay_block.parameter_block.id == dumped_block.parameter_block.id
            assert (
                replay_block.parameter_block.size == dumped_block.parameter_block.size
            )
            np.testing.assert_allclose(
                replay_block.values,
                dumped_block.values,
                rtol=1e-8,
                atol=1e-10,
            )


def test_global_positioning_trace_replay_selects_residual_ids(
    tmp_path: Path,
) -> None:
    trace = pycolmap.GlobalPositioningTrace.load(_make_replay_trace(tmp_path))

    replay = trace.replay(iteration=3, residual_ids=["scale_prior", "bata_ref"])

    assert replay.residual_ids == ("scale_prior", "bata_ref")
    assert replay.residual_dims == (1, 3)
    assert replay.residual_offsets == (0, 1)
    np.testing.assert_allclose(replay.raw_residuals, [2.0, 0.5, -1.0, -0.5])

    single_replay = trace.replay(iteration=3, residual_ids="metric")
    assert single_replay.residual_ids == ("metric",)
    np.testing.assert_allclose(single_replay.raw_residuals, [1.0])

    with pytest.raises(KeyError, match="missing replay residual ids"):
        trace.replay(iteration=3, residual_ids="missing")
    with pytest.raises(ValueError, match="non-empty"):
        trace.replay(iteration=3, residual_ids=[])
    with pytest.raises(ValueError, match="duplicates"):
        trace.replay(iteration=3, residual_ids=["metric", "metric"])


def test_global_positioning_trace_replay_finite_difference_jacobian_shape(
    tmp_path: Path,
) -> None:
    trace = pycolmap.GlobalPositioningTrace.load(_make_replay_trace(tmp_path))

    replay = trace.replay(iteration=3, compute_jacobians=True)

    residual = replay.residual("bata_ref")
    assert len(residual.jacobian_blocks) == 3
    assert residual.jacobian_blocks[0].values.shape == (3, 3)
    assert residual.jacobian_blocks[2].values.shape == (3, 1)
    np.testing.assert_allclose(residual.jacobian_blocks[0].values, 0.5 * np.eye(3))
    np.testing.assert_allclose(residual.jacobian_blocks[1].values, -0.5 * np.eye(3))
    np.testing.assert_allclose(
        residual.jacobian_blocks[2].values[:, 0], [-3.0, -2.0, -3.0]
    )
    assert residual.jacobian_blocks[0].offset == 0
    assert residual.jacobian_blocks[1].offset == 9
    assert residual.jacobian_blocks[2].offset == 18


def test_global_positioning_trace_replay_fails_loudly_for_bad_inputs(
    tmp_path: Path,
) -> None:
    trace = pycolmap.GlobalPositioningTrace.load(_make_replay_trace(tmp_path))
    with pytest.raises(KeyError, match="parameter snapshot"):
        trace.replay(iteration=99)

    unknown_loss_records = [
        {
            "event_type": "residual_added",
            "attrs": _replay_attrs(
                "bad_loss",
                "scale_prior",
                [_replay_parameter_block("dmap_scale", "dmap_scale", 50, 1)],
                {"scale_prior_target": 1.0, "scale_prior_stddev": 0.5},
                loss=_replay_loss("not_a_loss", 1.0, 1.0),
            ),
        }
    ]
    trace = pycolmap.GlobalPositioningTrace.load(
        _make_replay_trace(tmp_path / "bad_loss", unknown_loss_records)
    )
    with pytest.raises(ValueError, match="unsupported loss type"):
        trace.replay(iteration=3)

    bad_quat_records = [
        {
            "event_type": "residual_added",
            "attrs": _replay_attrs(
                "bad_quat",
                "metric_depth",
                [
                    _replay_parameter_block("frame_center", "frame_center", 10, 3),
                    _replay_parameter_block("point3D", "point3D", 20, 3),
                    _replay_parameter_block("dmap_scale", "dmap_scale", 50, 1),
                ],
                {
                    "camera_rotation_wxyz": [1.0, 0.0, 0.0],
                    "metric_depth_use_log_scale": False,
                    "metric_depth_residual_type": "linear",
                    "metric_depth_zero_residual_behind": False,
                    "metric_depth_log_linear_threshold": 0.1,
                },
                extra={"depth_prior": 1.0, "depth_sigma": 0.5},
            ),
        }
    ]
    trace = pycolmap.GlobalPositioningTrace.load(
        _make_replay_trace(tmp_path / "bad_quat", bad_quat_records)
    )
    with pytest.raises(ValueError, match="camera_rotation_wxyz.*length 4"):
        trace.replay(iteration=3)

    missing_fixed_records = [
        {
            "event_type": "residual_added",
            "attrs": _replay_attrs(
                "missing_fixed",
                "bata_constant_rig",
                [
                    _replay_parameter_block("point3D", "point3D", 20, 3),
                    _replay_parameter_block("frame_center", "frame_center", 10, 3),
                    _replay_parameter_block("bata_scale", "bata_scale", 30, 1),
                ],
                {"cam_from_point3D_dir": [1.0, 1.0, 1.0]},
            ),
        }
    ]
    trace = pycolmap.GlobalPositioningTrace.load(
        _make_replay_trace(tmp_path / "missing_fixed", missing_fixed_records)
    )
    with pytest.raises(KeyError, match="cam_from_rig_dir"):
        trace.replay(iteration=3)

    missing_snapshot_records = [
        {
            "event_type": "residual_added",
            "attrs": _replay_attrs(
                "missing_snapshot",
                "scale_prior",
                [_replay_parameter_block("dmap_scale", "dmap_scale", 999, 1)],
                {"scale_prior_target": 1.0, "scale_prior_stddev": 0.5},
            ),
        }
    ]
    trace = pycolmap.GlobalPositioningTrace.load(
        _make_replay_trace(tmp_path / "missing_snapshot", missing_snapshot_records)
    )
    with pytest.raises(KeyError, match="id=999"):
        trace.replay(iteration=3)


def test_global_positioning_trace_raw_binary_fixture_contract_minimal(
    tmp_path: Path,
) -> None:
    trace = pycolmap.GlobalPositioningTrace.load(_make_raw_binary_trace(tmp_path))

    assert trace.status == "finished"
    assert trace.trace_level == "raw_binary_minimal"
    assert trace.residual_value_iterations == (0,)
    assert trace.snapshot_iterations == (0,)
    assert [record["attrs"]["residual_id"] for record in trace.residual_blocks] == [
        "r10",
        "r20",
    ]
    assert trace.residual_blocks[0]["attrs"]["frame_id"] == 10
    assert trace.residual_blocks[1]["attrs"]["point3D_id"] == 100

    residual_values = trace.residual_values(0)
    assert residual_values.residual_ids == ["r10", "r20"]
    assert residual_values.residual_dims == [2, 1]
    assert residual_values.residual_offsets == [0, 2]
    assert residual_values.evaluation_success == [True, True]
    assert residual_values.has_loss_rho_values is True
    assert residual_values.has_raw_jacobians is False
    np.testing.assert_allclose(residual_values.raw_residuals, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(residual_values.raw_costs, [2.5, 4.5])
    np.testing.assert_allclose(residual_values.robust_costs, [2.25, 4.25])

    residual = residual_values.residual("r10")
    assert residual.residual_dim == 2
    np.testing.assert_allclose(residual.raw_residuals, [1.0, 2.0])
    np.testing.assert_allclose(residual.loss_rho, [4.5, 0.25, -0.125])

    snapshot = trace.snapshot(0)
    assert snapshot.frame_centers.ids == (10, 20)
    assert snapshot.points3D.ids == (100,)
    assert snapshot.scales.ids == (0,)
    np.testing.assert_allclose(
        snapshot.frame_centers.values, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    np.testing.assert_allclose(snapshot.points3D.values, [[0.0, 0.0, 2.0]])
    np.testing.assert_allclose(snapshot.scales.values, [1.0])


def test_global_positioning_trace_raw_binary_accepts_v2_without_jacobians(
    tmp_path: Path,
) -> None:
    trace = pycolmap.GlobalPositioningTrace.load(
        _make_raw_binary_trace(tmp_path, force_residual_version=2)
    )

    residual_values = trace.residual_values(0)
    assert residual_values.has_raw_jacobians is False
    assert residual_values.raw_jacobians is None
    assert residual_values.residual("r10").jacobian_blocks == ()


def test_global_positioning_trace_raw_binary_loads_optional_jacobians(
    tmp_path: Path,
) -> None:
    trace = pycolmap.GlobalPositioningTrace.load(
        _make_raw_binary_trace(tmp_path, with_jacobians=True)
    )

    residual_values = trace.residual_values(0)
    assert residual_values.has_raw_jacobians is True
    np.testing.assert_allclose(residual_values.raw_jacobians, np.arange(9))

    residual = residual_values.residual("r10")
    assert [block.role for block in residual.parameter_blocks] == [
        "frame_center",
        "bata_scale",
    ]
    assert [block.kind for block in residual.parameter_blocks] == [
        "frame_center",
        "bata_scale",
    ]
    assert [block.id for block in residual.parameter_blocks] == [10, 0]
    assert [block.size for block in residual.parameter_blocks] == [3, 1]
    assert residual.parameter_blocks[0].is_constant is False
    assert residual.parameter_blocks[1].is_constant is True
    np.testing.assert_allclose(
        residual.parameter_blocks[0].lower_bounds,
        [-np.inf, -np.inf, -np.inf],
    )
    np.testing.assert_allclose(residual.parameter_blocks[1].lower_bounds, [1e-5])
    np.testing.assert_allclose(
        residual.jacobian("frame_center"), np.arange(6).reshape(2, 3)
    )
    np.testing.assert_allclose(residual.jacobian("bata_scale"), [[6.0], [7.0]])

    residual_2 = residual_values.residual("r20")
    assert residual_2.parameter_blocks[0].kind == "dmap_scale"
    assert residual_2.parameter_blocks[0].id == 20
    np.testing.assert_allclose(residual_2.parameter_blocks[0].lower_bounds, [1e-5])
    np.testing.assert_allclose(residual_2.jacobian("dmap_scale"), [[8.0]])


def test_global_positioning_trace_raw_binary_rejects_bad_artifact_contract(
    tmp_path: Path,
) -> None:
    trace_dir = _make_raw_binary_trace(tmp_path)
    residual_path = trace_dir / "iterations" / "iter_000000" / "residual_values.bin"
    residual_path.write_bytes(residual_path.read_bytes()[:-1])

    trace = pycolmap.GlobalPositioningTrace.load(trace_dir)
    with pytest.raises(ValueError, match="truncated binary file"):
        trace.residual_values(0)
