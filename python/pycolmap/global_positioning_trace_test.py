import json
from pathlib import Path

import numpy as np

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


def _make_trace(tmp_path: Path, *, with_jacobians: bool) -> Path:
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
            "trace_level": "residual_jacobians"
            if with_jacobians
            else "residual_values",
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
    if with_jacobians:
        _write_f64(
            residual_values_dir / "iter_000000_raw_jacobians_f64.bin",
            np.arange(9, dtype=np.float64),
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

    residual = residual_values.residual("r0")
    np.testing.assert_allclose(residual.raw_residuals, [1.0, 2.0])
    assert residual.raw_cost == 2.5
    assert residual.robust_cost == 2.25
    assert residual.parameter_blocks == ()
    assert residual.jacobian_blocks == ()


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
