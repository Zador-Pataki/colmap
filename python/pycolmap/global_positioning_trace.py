from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_RAW_BINARY_STORAGE_FORMAT = "global_positioning_raw_binary_v1"
_RAW_LEDGER_MAGIC = b"GPTRLGR1"
_RAW_ARRAY_MAGIC = b"GPTRARR1"
_RAW_RESIDUAL_VALUES_MAGIC = b"GPTRRSV1"
_RAW_ENDIAN = "<"
_RAW_NONE_ID = -1


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _read_exact(stream: Any, size: int, label: str) -> bytes:
    payload = stream.read(size)
    if len(payload) != size:
        raise ValueError(f"{label}: truncated binary file")
    return payload


def _read_struct(stream: Any, fmt: str, label: str) -> tuple[Any, ...]:
    size = struct.calcsize(_RAW_ENDIAN + fmt)
    return struct.unpack(_RAW_ENDIAN + fmt, _read_exact(stream, size, label))


def _read_binary_string(stream: Any, label: str) -> str:
    (size,) = _read_struct(stream, "I", f"{label}.size")
    payload = _read_exact(stream, size, label)
    return payload.decode("utf-8")


def _read_binary_json(stream: Any, label: str) -> dict[str, Any]:
    text = _read_binary_string(stream, label)
    value = json.loads(text) if text else {}
    if not isinstance(value, dict):
        raise ValueError(f"{label}: expected encoded JSON object")
    return value


def _read_magic(stream: Any, expected: bytes, label: str) -> None:
    actual = _read_exact(stream, len(expected), label)
    if actual != expected:
        raise ValueError(f"{label}: bad magic {actual!r}, expected {expected!r}")


def _resolve_raw_trace_path(root: Path, relative_path: Path, label: str) -> Path:
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"{label}: invalid path {str(relative_path)!r}")
    root_resolved = root.resolve()
    resolved = (root / relative_path).resolve(strict=False)
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"{label}: path escapes trace root: {relative_path}") from exc
    return resolved


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def _require_key(mapping: dict[str, Any], key: str, label: str) -> Any:
    if key not in mapping:
        raise KeyError(f"{label}: missing key {key!r}")
    return mapping[key]


def _require_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label}: expected int, got {type(value).__name__}")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label}: expected bool, got {type(value).__name__}")
    return value


def _require_optional_iteration(value: Any, label: str) -> int | None:
    if value is None:
        return None
    iteration = _require_int(value, label)
    if iteration < 0:
        raise ValueError(f"{label}: expected non-negative iteration")
    return iteration


def _require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label}: expected non-empty string")
    return value


def _require_int_list(value: Any, label: str) -> list[int]:
    if not isinstance(value, list):
        raise TypeError(f"{label}: expected list")
    return [_require_int(item, f"{label}[{idx}]") for idx, item in enumerate(value)]


def _require_bool_list(value: Any, label: str) -> list[bool]:
    if not isinstance(value, list):
        raise TypeError(f"{label}: expected list")
    return [_require_bool(item, f"{label}[{idx}]") for idx, item in enumerate(value)]


def _require_str_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{label}: expected list")
    return [_require_str(item, f"{label}[{idx}]") for idx, item in enumerate(value)]


def _require_nested_int_list(value: Any, label: str) -> list[list[int]]:
    if not isinstance(value, list):
        raise TypeError(f"{label}: expected list")
    return [
        _require_int_list(item, f"{label}[{idx}]") for idx, item in enumerate(value)
    ]


def _require_nested_bool_list(value: Any, label: str) -> list[list[bool]]:
    if not isinstance(value, list):
        raise TypeError(f"{label}: expected list")
    return [
        _require_bool_list(item, f"{label}[{idx}]") for idx, item in enumerate(value)
    ]


def _coerce_trace_float(value: Any, label: str) -> float:
    if isinstance(value, str):
        if value == "nan":
            return float("nan")
        if value == "inf":
            return float("inf")
        if value == "-inf":
            return float("-inf")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise TypeError(f"{label}: expected numeric value, got {type(value).__name__}")


def _coerce_optional_trace_float(value: Any, label: str) -> float | None:
    if value is None:
        return None
    return _coerce_trace_float(value, label)


def _require_nested_float_blocks(value: Any, label: str) -> list[list[list[float]]]:
    if not isinstance(value, list):
        raise TypeError(f"{label}: expected list")
    nested_values: list[list[list[float]]] = []
    for residual_idx, residual_blocks in enumerate(value):
        if not isinstance(residual_blocks, list):
            raise TypeError(f"{label}[{residual_idx}]: expected list")
        parsed_blocks = []
        for block_idx, block_values in enumerate(residual_blocks):
            if not isinstance(block_values, list):
                raise TypeError(f"{label}[{residual_idx}][{block_idx}]: expected list")
            parsed_blocks.append(
                [
                    _coerce_trace_float(
                        bound,
                        f"{label}[{residual_idx}][{block_idx}][{bound_idx}]",
                    )
                    for bound_idx, bound in enumerate(block_values)
                ]
            )
        nested_values.append(parsed_blocks)
    return nested_values


def _require_shape(value: Any, label: str) -> tuple[int, ...]:
    shape = tuple(_require_int_list(value, label))
    if not shape:
        raise ValueError(f"{label}: expected non-empty shape")
    for idx, dim in enumerate(shape):
        if dim < 0:
            raise ValueError(f"{label}[{idx}]: expected non-negative dim")
    return shape


def _shape_element_count(shape: tuple[int, ...]) -> int:
    count = 1
    for dim in shape:
        count *= dim
    return count


def _validate_trace_metadata(
    metadata_path: Path,
    metadata: dict[str, Any],
    *,
    expected_iteration: int | None = None,
    expected_run_id: str | None = None,
) -> int:
    schema_version = _require_int(
        _require_key(metadata, "schema_version", str(metadata_path)),
        f"{metadata_path}: schema_version",
    )
    if schema_version != 1:
        raise ValueError(
            f"{metadata_path}: unsupported schema_version {schema_version}"
        )

    run_id = _require_str(
        _require_key(metadata, "run_id", str(metadata_path)),
        f"{metadata_path}: run_id",
    )
    if expected_run_id is not None and run_id != expected_run_id:
        raise ValueError(
            f"{metadata_path}: run_id {run_id!r}, expected {expected_run_id!r}"
        )

    iteration = _require_int(
        _require_key(metadata, "iteration", str(metadata_path)),
        f"{metadata_path}: iteration",
    )
    if iteration < 0:
        raise ValueError(f"{metadata_path}: iteration must be non-negative")
    if expected_iteration is not None and iteration != expected_iteration:
        raise ValueError(
            f"{metadata_path}: iteration {iteration}, " f"expected {expected_iteration}"
        )

    dtype = _require_str(
        _require_key(metadata, "dtype", str(metadata_path)),
        f"{metadata_path}: dtype",
    )
    byte_order = _require_str(
        _require_key(metadata, "byte_order", str(metadata_path)),
        f"{metadata_path}: byte_order",
    )
    if dtype != "float64":
        raise ValueError(f"{metadata_path}: dtype must be float64")
    if byte_order != "little_endian":
        raise ValueError(f"{metadata_path}: byte_order must be little_endian")
    return iteration


@dataclass(frozen=True)
class _ResolvedFloat64Artifact:
    path: Path
    ids: tuple[int, ...] | None
    shape: tuple[int, ...]
    element_count: int


def _artifact_metadata(
    metadata_path: Path,
    metadata: dict[str, Any],
    name: str,
    *,
    required: bool = True,
) -> dict[str, Any] | None:
    artifacts = _require_key(metadata, "artifacts", str(metadata_path))
    if not isinstance(artifacts, dict):
        raise TypeError(f"{metadata_path}: artifacts must be a JSON object")
    if not required and name not in artifacts:
        return None
    artifact = _require_key(artifacts, name, f"{metadata_path}: artifacts")
    if not isinstance(artifact, dict):
        raise TypeError(f"{metadata_path}: artifacts.{name} must be a JSON object")
    return artifact


def _resolve_float64_artifact(
    metadata_path: Path,
    metadata: dict[str, Any],
    name: str,
    *,
    expected_shape: tuple[int, ...] | None = None,
    expected_ids: tuple[int, ...] | None = None,
    required: bool = True,
) -> _ResolvedFloat64Artifact | None:
    artifact = _artifact_metadata(metadata_path, metadata, name, required=required)
    if artifact is None:
        return None

    filename = _require_str(
        _require_key(artifact, "file", f"{metadata_path}: artifacts.{name}"),
        f"{metadata_path}: artifacts.{name}.file",
    )
    relative_path = Path(filename)
    if (
        relative_path.is_absolute()
        or relative_path.name != filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
    ):
        raise ValueError(
            f"{metadata_path}: artifacts.{name}.file must be a bare relative "
            "filename"
        )
    dtype = _require_str(
        _require_key(artifact, "dtype", f"{metadata_path}: artifacts.{name}"),
        f"{metadata_path}: artifacts.{name}.dtype",
    )
    byte_order = _require_str(
        _require_key(artifact, "byte_order", f"{metadata_path}: artifacts.{name}"),
        f"{metadata_path}: artifacts.{name}.byte_order",
    )
    if dtype != "float64":
        raise ValueError(f"{metadata_path}: artifacts.{name}.dtype must be float64")
    if byte_order != "little_endian":
        raise ValueError(
            f"{metadata_path}: artifacts.{name}.byte_order must be " "little_endian"
        )
    shape = _require_shape(
        _require_key(artifact, "shape", f"{metadata_path}: artifacts.{name}"),
        f"{metadata_path}: artifacts.{name}.shape",
    )
    if expected_shape is not None and shape != expected_shape:
        raise ValueError(
            f"{metadata_path}: artifacts.{name}.shape is {shape}, "
            f"expected {expected_shape}"
        )

    ids = None
    if expected_ids is not None or "ids" in artifact:
        ids = tuple(
            _require_int_list(
                _require_key(artifact, "ids", f"{metadata_path}: artifacts.{name}"),
                f"{metadata_path}: artifacts.{name}.ids",
            )
        )
        if expected_ids is not None and ids != expected_ids:
            raise ValueError(
                f"{metadata_path}: artifacts.{name}.ids does not match " "top-level IDs"
            )
        if shape[0] != len(ids):
            raise ValueError(
                f"{metadata_path}: artifacts.{name}.shape has {shape[0]} "
                f"rows, expected {len(ids)} artifact IDs"
            )

    element_count = _shape_element_count(shape)
    path = metadata_path.parent / filename
    metadata_dir = metadata_path.parent.resolve()
    resolved_path = path.resolve(strict=False)
    try:
        resolved_path.relative_to(metadata_dir)
    except ValueError as exc:
        raise ValueError(
            f"{metadata_path}: artifacts.{name}.file escapes metadata directory"
        ) from exc
    expected_size = element_count * 8
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"{path}: byte size {actual_size}, expected {expected_size}")
    return _ResolvedFloat64Artifact(path, ids, shape, element_count)


def _memmap_float64(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    if _shape_element_count(shape) == 0:
        return np.empty(shape, dtype=np.float64)
    return np.memmap(path, dtype="<f8", mode="r", shape=shape)


def _discover_iteration_metadata(directory: Path) -> dict[int, Path]:
    metadata_by_iteration: dict[int, Path] = {}
    if not directory.is_dir():
        return metadata_by_iteration

    for metadata_path in sorted(directory.glob("iter_*.json")):
        metadata = _load_json(metadata_path)
        iteration = _require_int(
            _require_key(metadata, "iteration", str(metadata_path)),
            f"{metadata_path}: iteration",
        )
        expected_name = f"iter_{iteration:06d}.json"
        if metadata_path.name != expected_name:
            raise ValueError(
                f"{metadata_path}: filename does not match metadata "
                f"iteration; expected {expected_name}"
            )
        if iteration in metadata_by_iteration:
            previous = metadata_by_iteration[iteration]
            raise ValueError(
                f"Duplicate trace iteration {iteration}: "
                f"{previous} and {metadata_path}"
            )
        metadata_by_iteration[iteration] = metadata_path
    return metadata_by_iteration


@dataclass(frozen=True)
class GlobalPositioningParameterBlock:
    role: str
    kind: str
    id: int
    size: int
    is_constant: bool | None = None
    lower_bounds: tuple[float, ...] | None = None


@dataclass(frozen=True)
class GlobalPositioningResidualLedgerParameterBlock:
    role: str
    kind: str
    id: int
    size: int


@dataclass(frozen=True)
class GlobalPositioningResidualLedgerLoss:
    bucket: str
    type: str
    scale: float | None
    weight: float | None
    source: str
    observation_count_weight: float | None = None


@dataclass(frozen=True)
class GlobalPositioningResidualLedgerBlock:
    residual_id: str
    event_type: str
    replay_schema_version: int
    parameter_blocks: tuple[GlobalPositioningResidualLedgerParameterBlock, ...]
    loss: GlobalPositioningResidualLedgerLoss
    fixed_parameters_status: str
    fixed_parameters: dict[str, Any]
    attrs: dict[str, Any]


@dataclass(frozen=True)
class GlobalPositioningTraceEvent:
    schema_version: int
    run_id: str
    seq: int
    event_type: str
    stage: str
    iteration: int | None
    timestamp_ns: int
    attrs: dict[str, Any]


@dataclass(frozen=True)
class GlobalPositioningIterationMetric:
    schema_version: int
    run_id: str
    seq: int
    event_type: str
    stage: str
    iteration: int
    timestamp_ns: int
    step_is_successful: bool
    cost: float
    cost_change: float
    gradient_max_norm: float
    step_norm: float
    trust_region_radius: float
    linear_solver_iterations: int
    iteration_time_sec: float
    cumulative_time_sec: float
    attrs: dict[str, Any]
    gradient_norm: float | None = None


@dataclass(frozen=True)
class GlobalPositioningJacobianBlock:
    parameter_block: GlobalPositioningParameterBlock
    offset: int
    residual_dim: int
    values: np.ndarray


@dataclass(frozen=True)
class GlobalPositioningSnapshotArray:
    ids: tuple[int, ...]
    shape: tuple[int, ...]
    values: np.ndarray


@dataclass(frozen=True)
class GlobalPositioningParameterSnapshot:
    metadata_path: Path
    metadata: dict[str, Any]
    iteration: int
    frame_centers: GlobalPositioningSnapshotArray
    points3D: GlobalPositioningSnapshotArray
    scales: GlobalPositioningSnapshotArray
    dmap_scales: GlobalPositioningSnapshotArray | None = None
    cams_in_rig: GlobalPositioningSnapshotArray | None = None


@dataclass(frozen=True)
class GlobalPositioningReplayJacobianBlock:
    parameter_block: GlobalPositioningResidualLedgerParameterBlock
    offset: int
    residual_dim: int
    values: np.ndarray


@dataclass(frozen=True)
class GlobalPositioningReplayResidualBlock:
    residual_id: str
    residual_dim: int
    residual_offset: int
    evaluation_success: bool
    raw_residuals: np.ndarray
    raw_cost: float
    robust_cost: float
    loss_rho: np.ndarray
    parameter_blocks: tuple[GlobalPositioningResidualLedgerParameterBlock, ...]
    jacobian_blocks: tuple[GlobalPositioningReplayJacobianBlock, ...] = ()

    @property
    def loss_rho0(self) -> float:
        return float(self.loss_rho[0])

    @property
    def loss_rho1(self) -> float:
        return float(self.loss_rho[1])

    @property
    def loss_rho2(self) -> float:
        return float(self.loss_rho[2])

    @property
    def loss_derivative_scale(self) -> float:
        return self.loss_rho1


@dataclass(frozen=True)
class GlobalPositioningReplayEvaluation:
    iteration: int
    raw_residuals: np.ndarray
    raw_costs: np.ndarray
    loss_rho_values: np.ndarray
    robust_costs: np.ndarray
    evaluation_success: tuple[bool, ...]
    residual_ids: tuple[str, ...]
    residual_dims: tuple[int, ...]
    residual_offsets: tuple[int, ...]
    parameter_blocks: tuple[
        tuple[GlobalPositioningResidualLedgerParameterBlock, ...], ...
    ]
    raw_jacobians: tuple[tuple[GlobalPositioningReplayJacobianBlock, ...], ...] = ()

    @property
    def has_raw_jacobians(self) -> bool:
        return bool(self.raw_jacobians)

    @property
    def has_loss_rho_values(self) -> bool:
        return True

    def residual(self, residual: int | str) -> GlobalPositioningReplayResidualBlock:
        if isinstance(residual, str):
            try:
                residual = self.residual_ids.index(residual)
            except ValueError as exc:
                raise KeyError(f"Unknown residual id {residual!r}") from exc
        if residual < 0 or residual >= len(self.residual_ids):
            raise IndexError(f"Residual index {residual} out of range")
        offset = self.residual_offsets[residual]
        dim = self.residual_dims[residual]
        jacobian_blocks = self.raw_jacobians[residual] if self.raw_jacobians else ()
        return GlobalPositioningReplayResidualBlock(
            residual_id=self.residual_ids[residual],
            residual_dim=dim,
            residual_offset=offset,
            evaluation_success=self.evaluation_success[residual],
            raw_residuals=self.raw_residuals[offset : offset + dim],
            raw_cost=float(self.raw_costs[residual]),
            robust_cost=float(self.robust_costs[residual]),
            loss_rho=self.loss_rho_values[residual],
            parameter_blocks=self.parameter_blocks[residual],
            jacobian_blocks=jacobian_blocks,
        )


class GlobalPositioningResidualBlock:
    def __init__(self, residual_values: GlobalPositioningResidualValues, index: int):
        self._residual_values = residual_values
        self.index = index

    @property
    def residual_id(self) -> str:
        return self._residual_values.residual_ids[self.index]

    @property
    def residual_dim(self) -> int:
        return self._residual_values.residual_dims[self.index]

    @property
    def residual_offset(self) -> int:
        return self._residual_values.residual_offsets[self.index]

    @property
    def evaluation_success(self) -> bool:
        return self._residual_values.evaluation_success[self.index]

    @property
    def raw_residuals(self) -> np.ndarray:
        begin = self.residual_offset
        end = begin + self.residual_dim
        return self._residual_values.raw_residuals[begin:end]

    @property
    def raw_cost(self) -> float:
        return float(self._residual_values.raw_costs[self.index])

    @property
    def robust_cost(self) -> float:
        return float(self._residual_values.robust_costs[self.index])

    @property
    def loss_rho(self) -> np.ndarray:
        return self._residual_values.loss_rho_values[self.index]

    @property
    def loss_rho0(self) -> float:
        return float(self.loss_rho[0])

    @property
    def loss_rho1(self) -> float:
        return float(self.loss_rho[1])

    @property
    def loss_rho2(self) -> float:
        return float(self.loss_rho[2])

    @property
    def loss_derivative_scale(self) -> float:
        return self.loss_rho1

    @property
    def parameter_blocks(self) -> tuple[GlobalPositioningParameterBlock, ...]:
        if not self._residual_values.has_raw_jacobians:
            return ()
        return tuple(self._residual_values.parameter_blocks[self.index])

    @property
    def jacobian_blocks(self) -> tuple[GlobalPositioningJacobianBlock, ...]:
        if not self._residual_values.has_raw_jacobians:
            return ()
        raw_jacobians = self._residual_values.raw_jacobians
        if raw_jacobians is None:
            return ()

        blocks = []
        for parameter_block, offset in zip(
            self._residual_values.parameter_blocks[self.index],
            self._residual_values.raw_jacobian_offsets[self.index],
            strict=True,
        ):
            end = offset + self.residual_dim * parameter_block.size
            values = raw_jacobians[offset:end].reshape(
                (self.residual_dim, parameter_block.size)
            )
            blocks.append(
                GlobalPositioningJacobianBlock(
                    parameter_block, offset, self.residual_dim, values
                )
            )
        return tuple(blocks)

    def jacobian(self, block: int | str, *, id: int | None = None) -> np.ndarray:
        jacobian_blocks = self.jacobian_blocks
        if isinstance(block, int):
            return jacobian_blocks[block].values
        matches = [
            item
            for item in jacobian_blocks
            if item.parameter_block.role == block
            and (id is None or item.parameter_block.id == id)
        ]
        if len(matches) != 1:
            raise KeyError(
                f"Expected exactly one Jacobian block for role={block!r}, "
                f"id={id!r}; found {len(matches)}"
            )
        return matches[0].values


class GlobalPositioningResidualValues:
    def __init__(
        self,
        metadata_path: Path,
        *,
        expected_iteration: int | None = None,
        expected_run_id: str | None = None,
        expected_residual_ids: tuple[str, ...] | None = None,
    ):
        self.metadata_path = Path(metadata_path)
        self.metadata = _load_json(self.metadata_path)
        self.iteration = _validate_trace_metadata(
            self.metadata_path,
            self.metadata,
            expected_iteration=expected_iteration,
            expected_run_id=expected_run_id,
        )
        self.num_residual_blocks = _require_int(
            _require_key(self.metadata, "num_residual_blocks", str(self.metadata_path)),
            "num_residual_blocks",
        )
        self.total_scalar_residuals = _require_int(
            _require_key(
                self.metadata, "total_scalar_residuals", str(self.metadata_path)
            ),
            "total_scalar_residuals",
        )
        self.residual_ids = _require_str_list(
            _require_key(self.metadata, "residual_ids", str(self.metadata_path)),
            "residual_ids",
        )
        self.residual_dims = _require_int_list(
            _require_key(self.metadata, "residual_dims", str(self.metadata_path)),
            "residual_dims",
        )
        self.residual_offsets = _require_int_list(
            _require_key(self.metadata, "residual_offsets", str(self.metadata_path)),
            "residual_offsets",
        )
        self.evaluation_success = _require_bool_list(
            _require_key(self.metadata, "evaluation_success", str(self.metadata_path)),
            "evaluation_success",
        )
        self._validate_residual_structure()
        if (
            expected_residual_ids is not None
            and tuple(self.residual_ids) != expected_residual_ids
        ):
            raise ValueError(
                "residual_values.residual_ids does not match "
                "residual_blocks.jsonl order"
            )
        self.has_raw_jacobians = _require_bool(
            _require_key(self.metadata, "has_raw_jacobians", str(self.metadata_path)),
            "has_raw_jacobians",
        )
        self._residual_id_to_index = {
            residual_id: idx for idx, residual_id in enumerate(self.residual_ids)
        }

        raw_residuals_artifact = _resolve_float64_artifact(
            self.metadata_path,
            self.metadata,
            "raw_residuals",
            expected_shape=(self.total_scalar_residuals,),
        )
        raw_costs_artifact = _resolve_float64_artifact(
            self.metadata_path,
            self.metadata,
            "raw_costs",
            expected_shape=(self.num_residual_blocks,),
        )
        robust_costs_artifact = _resolve_float64_artifact(
            self.metadata_path,
            self.metadata,
            "robust_costs",
            expected_shape=(self.num_residual_blocks,),
        )
        assert raw_residuals_artifact is not None
        assert raw_costs_artifact is not None
        assert robust_costs_artifact is not None
        self._raw_residuals_artifact = raw_residuals_artifact
        self._raw_costs_artifact = raw_costs_artifact
        self._robust_costs_artifact = robust_costs_artifact
        self._loss_rho_values_artifact = _resolve_float64_artifact(
            self.metadata_path,
            self.metadata,
            "loss_rho_values",
            expected_shape=(self.num_residual_blocks, 3),
            required=False,
        )
        self.has_loss_rho_values = self._loss_rho_values_artifact is not None
        if self.has_loss_rho_values:
            self._validate_loss_rho_contract()

        self.total_jacobian_scalars = 0
        self.parameter_blocks: list[list[GlobalPositioningParameterBlock]] = []
        self.raw_jacobian_offsets: list[list[int]] = []
        self._raw_jacobians_artifact: _ResolvedFloat64Artifact | None = None
        if self.has_raw_jacobians:
            self.total_jacobian_scalars = _require_int(
                _require_key(
                    self.metadata,
                    "total_jacobian_scalars",
                    str(self.metadata_path),
                ),
                "total_jacobian_scalars",
            )
            parameter_block_sizes = _require_nested_int_list(
                _require_key(
                    self.metadata,
                    "parameter_block_sizes",
                    str(self.metadata_path),
                ),
                "parameter_block_sizes",
            )
            self.raw_jacobian_offsets = _require_nested_int_list(
                _require_key(
                    self.metadata,
                    "raw_jacobian_offsets",
                    str(self.metadata_path),
                ),
                "raw_jacobian_offsets",
            )
            parameter_block_descriptors = _require_key(
                self.metadata, "parameter_blocks", str(self.metadata_path)
            )
            parameter_block_is_constant = _require_nested_bool_list(
                _require_key(
                    self.metadata,
                    "parameter_block_is_constant",
                    str(self.metadata_path),
                ),
                "parameter_block_is_constant",
            )
            parameter_block_lower_bounds = _require_nested_float_blocks(
                _require_key(
                    self.metadata,
                    "parameter_block_lower_bounds",
                    str(self.metadata_path),
                ),
                "parameter_block_lower_bounds",
            )
            self._validate_jacobian_contract()
            self.parameter_blocks = self._parse_parameter_blocks(
                parameter_block_descriptors,
                parameter_block_sizes,
                parameter_block_is_constant,
                parameter_block_lower_bounds,
            )
            self._validate_jacobian_structure(
                parameter_block_sizes,
                self.raw_jacobian_offsets,
                parameter_block_is_constant,
                parameter_block_lower_bounds,
                parameter_block_descriptors,
            )
            self._raw_jacobians_artifact = _resolve_float64_artifact(
                self.metadata_path,
                self.metadata,
                "raw_jacobians",
                expected_shape=(self.total_jacobian_scalars,),
            )
            assert self._raw_jacobians_artifact is not None

        self._raw_residuals: np.ndarray | None = None
        self._raw_costs: np.ndarray | None = None
        self._robust_costs: np.ndarray | None = None
        self._loss_rho_values: np.ndarray | None = None
        self._loss_rho_costs_validated = False
        self._raw_jacobians: np.ndarray | None = None

    def _validate_loss_rho_contract(self) -> None:
        loss_rho_layout = _require_str(
            _require_key(self.metadata, "loss_rho_layout", str(self.metadata_path)),
            "loss_rho_layout",
        )
        if loss_rho_layout != "residual_block_major/rho0_rho1_rho2":
            raise ValueError(
                "loss_rho_layout must be 'residual_block_major/rho0_rho1_rho2'"
            )

    def _validate_loss_rho_costs(self) -> None:
        if self._loss_rho_costs_validated:
            return
        loss_rho_values = self._require_loss_rho_values()
        success_mask = np.asarray(self.evaluation_success, dtype=bool)
        if np.any(success_mask):
            robust_costs = np.asarray(self.robust_costs)
            expected_robust_costs = 0.5 * loss_rho_values[:, 0]
            matches = np.isclose(
                robust_costs,
                expected_robust_costs,
                rtol=1e-10,
                atol=1e-12,
                equal_nan=False,
            )
            mismatch_mask = success_mask & ~matches
            if np.any(mismatch_mask):
                mismatch_idx = int(np.flatnonzero(mismatch_mask)[0])
                raise ValueError(
                    "robust_costs must equal 0.5 * loss_rho_values[:, 0] "
                    "for successful residual evaluations; mismatch at "
                    f"residual {mismatch_idx} "
                    f"({self.residual_ids[mismatch_idx]!r})"
                )
        self._loss_rho_costs_validated = True

    def _require_loss_rho_values(self) -> np.ndarray:
        if self._loss_rho_values_artifact is None:
            raise ValueError(
                "loss_rho_values artifact is not present in this trace; "
                "robust loss rho diagnostics require a newer trace"
            )
        if self._loss_rho_values is None:
            self._loss_rho_values = _memmap_float64(
                self._loss_rho_values_artifact.path,
                self._loss_rho_values_artifact.shape,
            )
        return self._loss_rho_values

    def _validate_jacobian_contract(self) -> None:
        raw_jacobian_layout = _require_str(
            _require_key(
                self.metadata,
                "raw_jacobian_layout",
                str(self.metadata_path),
            ),
            "raw_jacobian_layout",
        )
        if (
            raw_jacobian_layout
            != "residual_block_major/parameter_block_major/row_major"
        ):
            raise ValueError(
                "raw_jacobian_layout must be "
                "'residual_block_major/parameter_block_major/row_major'"
            )
        jacobian_domain = _require_str(
            _require_key(self.metadata, "jacobian_domain", str(self.metadata_path)),
            "jacobian_domain",
        )
        if jacobian_domain != "raw_cost_function_ambient_parameters":
            raise ValueError(
                "jacobian_domain must be 'raw_cost_function_ambient_parameters'"
            )
        for field_name in [
            "loss_applied_to_jacobians",
            "manifold_applied_to_jacobians",
            "constant_parameter_blocks_included",
        ]:
            _require_bool(
                _require_key(self.metadata, field_name, str(self.metadata_path)),
                field_name,
            )
        if self.metadata["loss_applied_to_jacobians"]:
            raise ValueError("loss_applied_to_jacobians must be false")
        if self.metadata["manifold_applied_to_jacobians"]:
            raise ValueError("manifold_applied_to_jacobians must be false")
        if not self.metadata["constant_parameter_blocks_included"]:
            raise ValueError("constant_parameter_blocks_included must be true")

    def _validate_residual_structure(self) -> None:
        if self.num_residual_blocks < 0:
            raise ValueError("num_residual_blocks must be non-negative")
        if self.total_scalar_residuals < 0:
            raise ValueError("total_scalar_residuals must be non-negative")
        for name, values in [
            ("residual_ids", self.residual_ids),
            ("residual_dims", self.residual_dims),
            ("residual_offsets", self.residual_offsets),
            ("evaluation_success", self.evaluation_success),
        ]:
            if len(values) != self.num_residual_blocks:
                raise ValueError(
                    f"{name}: length {len(values)}, "
                    f"expected {self.num_residual_blocks}"
                )
        if len(set(self.residual_ids)) != len(self.residual_ids):
            raise ValueError("residual_ids must be unique")

        expected_offset = 0
        for residual_idx, (residual_dim, residual_offset) in enumerate(
            zip(self.residual_dims, self.residual_offsets, strict=True)
        ):
            if residual_dim < 0:
                raise ValueError(f"residual_dims[{residual_idx}] must be non-negative")
            if residual_offset != expected_offset:
                raise ValueError(
                    f"residual_offsets[{residual_idx}] is {residual_offset}, "
                    f"expected {expected_offset}"
                )
            expected_offset += residual_dim
        if expected_offset != self.total_scalar_residuals:
            raise ValueError(
                f"sum(residual_dims) is {expected_offset}, "
                f"expected total_scalar_residuals {self.total_scalar_residuals}"
            )

    def _validate_jacobian_structure(
        self,
        sizes: list[list[int]],
        offsets: list[list[int]],
        is_constant: list[list[bool]],
        lower_bounds: list[list[list[float]]],
        descriptors: Any,
    ) -> None:
        if self.total_jacobian_scalars < 0:
            raise ValueError("total_jacobian_scalars must be non-negative")
        if not isinstance(descriptors, list):
            raise TypeError("parameter_blocks: expected list")
        for name, values in [
            ("parameter_block_sizes", sizes),
            ("raw_jacobian_offsets", offsets),
            ("parameter_block_is_constant", is_constant),
            ("parameter_block_lower_bounds", lower_bounds),
            ("parameter_blocks", descriptors),
        ]:
            if len(values) != self.num_residual_blocks:
                raise ValueError(
                    f"{name}: outer length {len(values)}, "
                    f"expected {self.num_residual_blocks}"
                )

        expected_offset = 0
        for residual_idx in range(self.num_residual_blocks):
            block_count = len(sizes[residual_idx])
            for name, values in [
                ("raw_jacobian_offsets", offsets),
                ("parameter_block_is_constant", is_constant),
                ("parameter_block_lower_bounds", lower_bounds),
                ("parameter_blocks", descriptors),
            ]:
                if len(values[residual_idx]) != block_count:
                    raise ValueError(
                        f"{name}[{residual_idx}]: length "
                        f"{len(values[residual_idx])}, expected {block_count}"
                    )
            for block_idx, block_size in enumerate(sizes[residual_idx]):
                if block_size <= 0:
                    raise ValueError(
                        f"parameter_block_sizes[{residual_idx}][{block_idx}] "
                        "must be positive"
                    )
                if len(lower_bounds[residual_idx][block_idx]) != block_size:
                    raise ValueError(
                        f"parameter_block_lower_bounds[{residual_idx}]"
                        f"[{block_idx}]: length "
                        f"{len(lower_bounds[residual_idx][block_idx])}, "
                        f"expected {block_size}"
                    )
                residual_offset = offsets[residual_idx][block_idx]
                if residual_offset != expected_offset:
                    raise ValueError(
                        f"raw_jacobian_offsets[{residual_idx}][{block_idx}] "
                        f"is {residual_offset}, expected {expected_offset}"
                    )
                expected_offset += self.residual_dims[residual_idx] * block_size
        if expected_offset != self.total_jacobian_scalars:
            raise ValueError(
                f"raw Jacobian scalar count is {expected_offset}, "
                f"expected total_jacobian_scalars {self.total_jacobian_scalars}"
            )

    @staticmethod
    def _parse_parameter_blocks(
        descriptors: Any,
        sizes: list[list[int]],
        is_constant: list[list[bool]],
        lower_bounds: list[list[list[float]]],
    ) -> list[list[GlobalPositioningParameterBlock]]:
        if not isinstance(descriptors, list):
            raise TypeError("parameter_blocks: expected list")
        blocks: list[list[GlobalPositioningParameterBlock]] = []
        for residual_idx, residual_descriptors in enumerate(descriptors):
            if not isinstance(residual_descriptors, list):
                raise TypeError(f"parameter_blocks[{residual_idx}]: expected list")
            residual_blocks = []
            for block_idx, descriptor in enumerate(residual_descriptors):
                if not isinstance(descriptor, dict):
                    raise TypeError(
                        f"parameter_blocks[{residual_idx}][{block_idx}]: "
                        "expected object"
                    )
                role = _require_str(
                    _require_key(descriptor, "role", "parameter_block"), "role"
                )
                kind = _require_str(
                    _require_key(descriptor, "kind", "parameter_block"), "kind"
                )
                block_id = _require_int(
                    _require_key(descriptor, "id", "parameter_block"), "id"
                )
                residual_blocks.append(
                    GlobalPositioningParameterBlock(
                        role=role,
                        kind=kind,
                        id=block_id,
                        size=sizes[residual_idx][block_idx],
                        is_constant=is_constant[residual_idx][block_idx],
                        lower_bounds=tuple(lower_bounds[residual_idx][block_idx]),
                    )
                )
            blocks.append(residual_blocks)
        return blocks

    @property
    def raw_residuals(self) -> np.ndarray:
        if self._raw_residuals is None:
            self._raw_residuals = _memmap_float64(
                self._raw_residuals_artifact.path,
                self._raw_residuals_artifact.shape,
            )
        return self._raw_residuals

    @property
    def raw_costs(self) -> np.ndarray:
        if self._raw_costs is None:
            self._raw_costs = _memmap_float64(
                self._raw_costs_artifact.path,
                self._raw_costs_artifact.shape,
            )
        return self._raw_costs

    @property
    def robust_costs(self) -> np.ndarray:
        if self._robust_costs is None:
            self._robust_costs = _memmap_float64(
                self._robust_costs_artifact.path,
                self._robust_costs_artifact.shape,
            )
        return self._robust_costs

    @property
    def loss_rho_values(self) -> np.ndarray:
        loss_rho_values = self._require_loss_rho_values()
        self._validate_loss_rho_costs()
        return loss_rho_values

    @property
    def raw_jacobians(self) -> np.ndarray | None:
        if self._raw_jacobians_artifact is None:
            return None
        if self._raw_jacobians is None:
            self._raw_jacobians = _memmap_float64(
                self._raw_jacobians_artifact.path,
                self._raw_jacobians_artifact.shape,
            )
        return self._raw_jacobians

    def residual(self, residual: int | str) -> GlobalPositioningResidualBlock:
        if isinstance(residual, str):
            residual = self._residual_id_to_index[residual]
        if residual < 0 or residual >= self.num_residual_blocks:
            raise IndexError(f"Residual index {residual} out of range")
        return GlobalPositioningResidualBlock(self, residual)


def _require_top_level_ids_and_shape(
    metadata_path: Path,
    metadata: dict[str, Any],
    ids_key: str,
    shape_key: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    ids = tuple(
        _require_int_list(
            _require_key(metadata, ids_key, str(metadata_path)),
            f"{metadata_path}: {ids_key}",
        )
    )
    shape = _require_shape(
        _require_key(metadata, shape_key, str(metadata_path)),
        f"{metadata_path}: {shape_key}",
    )
    if shape[0] != len(ids):
        raise ValueError(
            f"{metadata_path}: {shape_key} has {shape[0]} rows, "
            f"expected {len(ids)} {ids_key}"
        )
    return ids, shape


def _optional_top_level_ids_and_shape(
    metadata_path: Path,
    metadata: dict[str, Any],
    ids_key: str,
    shape_key: str,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    if ids_key not in metadata and shape_key not in metadata:
        return None
    return _require_top_level_ids_and_shape(metadata_path, metadata, ids_key, shape_key)


def _normalize_max_points(max_points: int | None) -> int | None:
    if max_points is None or max_points == -1:
        return None
    if not isinstance(max_points, int) or isinstance(max_points, bool):
        raise TypeError(
            f"max_points: expected int, None, or -1; "
            f"got {type(max_points).__name__}"
        )
    if max_points < -1:
        raise ValueError(f"max_points must be >= -1 or None, got {max_points}")
    return max_points


def _load_snapshot_array(
    metadata_path: Path,
    metadata: dict[str, Any],
    name: str,
    *,
    expected_ids: tuple[int, ...] | None = None,
    expected_shape: tuple[int, ...] | None = None,
    max_rows: int | None = None,
    required: bool = True,
) -> GlobalPositioningSnapshotArray | None:
    artifact = _resolve_float64_artifact(
        metadata_path,
        metadata,
        name,
        expected_shape=expected_shape,
        expected_ids=expected_ids,
        required=required,
    )
    if artifact is None:
        return None

    values = _memmap_float64(artifact.path, artifact.shape)
    ids = artifact.ids if artifact.ids is not None else ()
    shape = artifact.shape
    if max_rows is not None:
        row_count = min(max_rows, shape[0])
        values = values[:row_count]
        ids = ids[:row_count]
        shape = (row_count, *shape[1:])

    return GlobalPositioningSnapshotArray(ids=ids, shape=shape, values=values)


def _empty_snapshot_array() -> GlobalPositioningSnapshotArray:
    return GlobalPositioningSnapshotArray(
        ids=(),
        shape=(0,),
        values=np.empty((0,), dtype=np.float64),
    )


class GlobalPositioningParameterSnapshotLoader:
    def __init__(
        self,
        metadata_path: Path,
        *,
        expected_iteration: int | None = None,
        expected_run_id: str | None = None,
        max_points: int | None = None,
    ):
        self.metadata_path = Path(metadata_path)
        self.metadata = _load_json(self.metadata_path)
        self.iteration = _validate_trace_metadata(
            self.metadata_path,
            self.metadata,
            expected_iteration=expected_iteration,
            expected_run_id=expected_run_id,
        )
        self.max_points = _normalize_max_points(max_points)

    def load(self) -> GlobalPositioningParameterSnapshot:
        frame_ids, frame_centers_shape = _require_top_level_ids_and_shape(
            self.metadata_path,
            self.metadata,
            "frame_ids",
            "frame_centers_world_shape",
        )
        point3D_ids, points3D_shape = _require_top_level_ids_and_shape(
            self.metadata_path,
            self.metadata,
            "point3D_ids",
            "points3D_world_shape",
        )
        scales_expected = _optional_top_level_ids_and_shape(
            self.metadata_path,
            self.metadata,
            "bata_scale_ids",
            "bata_scales_shape",
        )

        dmap_expected = _optional_top_level_ids_and_shape(
            self.metadata_path,
            self.metadata,
            "dmap_image_ids",
            "dmap_scales_stored_shape",
        )
        artifacts = _require_key(self.metadata, "artifacts", str(self.metadata_path))
        if not isinstance(artifacts, dict):
            raise TypeError(f"{self.metadata_path}: artifacts must be a JSON object")
        if dmap_expected is not None and "dmap_scales" not in artifacts:
            dmap_ids, dmap_shape = dmap_expected
            if dmap_ids or _shape_element_count(dmap_shape) != 0:
                raise KeyError(
                    f"{self.metadata_path}: artifacts missing optional "
                    "dmap_scales despite non-empty top-level dmap metadata"
                )

        frame_centers = _load_snapshot_array(
            self.metadata_path,
            self.metadata,
            "frame_centers",
            expected_ids=frame_ids,
            expected_shape=frame_centers_shape,
        )
        points3D = _load_snapshot_array(
            self.metadata_path,
            self.metadata,
            "points3D",
            expected_ids=point3D_ids,
            expected_shape=points3D_shape,
            max_rows=self.max_points,
        )
        scales = _empty_snapshot_array()
        if scales_expected is not None:
            scale_ids, scales_shape = scales_expected
            loaded_scales = _load_snapshot_array(
                self.metadata_path,
                self.metadata,
                "scales",
                expected_ids=scale_ids,
                expected_shape=scales_shape,
            )
            assert loaded_scales is not None
            scales = loaded_scales
        assert frame_centers is not None
        assert points3D is not None

        dmap_scales = None
        if "dmap_scales" in artifacts:
            dmap_ids, dmap_shape = dmap_expected or (None, None)
            dmap_scales = _load_snapshot_array(
                self.metadata_path,
                self.metadata,
                "dmap_scales",
                expected_ids=dmap_ids,
                expected_shape=dmap_shape,
                required=False,
            )

        cams_in_rig = _load_snapshot_array(
            self.metadata_path,
            self.metadata,
            "cams_in_rig",
            required=False,
        )

        return GlobalPositioningParameterSnapshot(
            metadata_path=self.metadata_path,
            metadata=self.metadata,
            iteration=self.iteration,
            frame_centers=frame_centers,
            points3D=points3D,
            scales=scales,
            dmap_scales=dmap_scales,
            cams_in_rig=cams_in_rig,
        )


def _parse_ledger_parameter_blocks(
    value: Any, label: str
) -> tuple[GlobalPositioningResidualLedgerParameterBlock, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{label}: expected list")
    blocks = []
    for idx, descriptor in enumerate(value):
        descriptor_label = f"{label}[{idx}]"
        if not isinstance(descriptor, dict):
            raise TypeError(f"{descriptor_label}: expected object")
        role = _require_str(
            _require_key(descriptor, "role", descriptor_label),
            f"{descriptor_label}.role",
        )
        kind = _require_str(
            _require_key(descriptor, "kind", descriptor_label),
            f"{descriptor_label}.kind",
        )
        block_id = _require_int(
            _require_key(descriptor, "id", descriptor_label),
            f"{descriptor_label}.id",
        )
        size = _require_int(
            _require_key(descriptor, "size", descriptor_label),
            f"{descriptor_label}.size",
        )
        if size <= 0:
            raise ValueError(f"{descriptor_label}.size must be positive")
        blocks.append(
            GlobalPositioningResidualLedgerParameterBlock(
                role=role,
                kind=kind,
                id=block_id,
                size=size,
            )
        )
    return tuple(blocks)


def _parse_ledger_loss(value: Any, label: str) -> GlobalPositioningResidualLedgerLoss:
    if not isinstance(value, dict):
        raise TypeError(f"{label}: expected object")
    observation_count_weight = None
    if "observation_count_weight" in value:
        observation_count_weight = _coerce_trace_float(
            value["observation_count_weight"],
            f"{label}.observation_count_weight",
        )
    return GlobalPositioningResidualLedgerLoss(
        bucket=_require_str(_require_key(value, "bucket", label), f"{label}.bucket"),
        type=_require_str(_require_key(value, "type", label), f"{label}.type"),
        scale=_coerce_optional_trace_float(
            _require_key(value, "scale", label), f"{label}.scale"
        ),
        weight=_coerce_optional_trace_float(
            _require_key(value, "weight", label), f"{label}.weight"
        ),
        source=_require_str(_require_key(value, "source", label), f"{label}.source"),
        observation_count_weight=observation_count_weight,
    )


def _parse_residual_ledger_block(
    record: dict[str, Any], idx: int
) -> GlobalPositioningResidualLedgerBlock | None:
    label = f"residual_blocks[{idx}]"
    attrs = _require_key(record, "attrs", label)
    if not isinstance(attrs, dict):
        raise TypeError(f"{label}.attrs must be an object")
    if "replay_schema_version" not in attrs:
        return None
    replay_schema_version = _require_int(
        attrs["replay_schema_version"], f"{label}.attrs.replay_schema_version"
    )
    if replay_schema_version != 1:
        raise ValueError(
            f"{label}.attrs.replay_schema_version: unsupported "
            f"replay_schema_version {replay_schema_version}"
        )

    fixed_parameters_status = _require_str(
        _require_key(attrs, "fixed_parameters_status", f"{label}.attrs"),
        f"{label}.attrs.fixed_parameters_status",
    )
    if fixed_parameters_status != "serialized":
        raise ValueError(
            f"{label}.attrs.fixed_parameters_status must be 'serialized' for "
            f"typed replay loading, got {fixed_parameters_status!r}"
        )
    fixed_parameters = _require_key(attrs, "fixed_parameters", f"{label}.attrs")
    if not isinstance(fixed_parameters, dict):
        raise TypeError(f"{label}.attrs.fixed_parameters must be an object")

    event_type = _require_str(
        _require_key(record, "event_type", label), f"{label}.event_type"
    )
    return GlobalPositioningResidualLedgerBlock(
        residual_id=_require_str(
            _require_key(attrs, "residual_id", f"{label}.attrs"),
            f"{label}.attrs.residual_id",
        ),
        event_type=event_type,
        replay_schema_version=replay_schema_version,
        parameter_blocks=_parse_ledger_parameter_blocks(
            _require_key(attrs, "parameter_blocks", f"{label}.attrs"),
            f"{label}.attrs.parameter_blocks",
        ),
        loss=_parse_ledger_loss(
            _require_key(attrs, "loss", f"{label}.attrs"),
            f"{label}.attrs.loss",
        ),
        fixed_parameters_status=fixed_parameters_status,
        fixed_parameters=fixed_parameters,
        attrs=attrs,
    )


def _parse_trace_event_record(
    record: dict[str, Any],
    *,
    path: Path,
    idx: int,
    expected_run_id: str | None,
) -> GlobalPositioningTraceEvent:
    label = f"{path}:{idx + 1}"
    schema_version = _require_int(
        _require_key(record, "schema_version", label),
        f"{label}.schema_version",
    )
    if schema_version != 1:
        raise ValueError(f"{label}: unsupported schema_version {schema_version}")
    run_id = _require_str(_require_key(record, "run_id", label), f"{label}.run_id")
    if expected_run_id is not None and run_id != expected_run_id:
        raise ValueError(f"{label}: run_id {run_id!r}, expected {expected_run_id!r}")
    seq = _require_int(_require_key(record, "seq", label), f"{label}.seq")
    if seq < 0:
        raise ValueError(f"{label}.seq must be non-negative")
    timestamp_ns = _require_int(
        _require_key(record, "timestamp_ns", label),
        f"{label}.timestamp_ns",
    )
    if timestamp_ns < 0:
        raise ValueError(f"{label}.timestamp_ns must be non-negative")
    attrs = _require_key(record, "attrs", label)
    if not isinstance(attrs, dict):
        raise TypeError(f"{label}.attrs must be an object")
    return GlobalPositioningTraceEvent(
        schema_version=schema_version,
        run_id=run_id,
        seq=seq,
        event_type=_require_str(
            _require_key(record, "event_type", label),
            f"{label}.event_type",
        ),
        stage=_require_str(_require_key(record, "stage", label), f"{label}.stage"),
        iteration=_require_optional_iteration(
            _require_key(record, "iteration", label),
            f"{label}.iteration",
        ),
        timestamp_ns=timestamp_ns,
        attrs=attrs,
    )


def _parse_iteration_metric_record(
    record: dict[str, Any],
    *,
    path: Path,
    idx: int,
    expected_run_id: str | None,
) -> GlobalPositioningIterationMetric:
    event = _parse_trace_event_record(
        record,
        path=path,
        idx=idx,
        expected_run_id=expected_run_id,
    )
    label = f"{path}:{idx + 1}"
    if event.event_type != "ceres_iteration":
        raise ValueError(
            f"{label}.event_type must be 'ceres_iteration', got {event.event_type!r}"
        )
    if event.iteration is None:
        raise ValueError(f"{label}.iteration must not be null")
    attrs = event.attrs
    linear_solver_iterations = _require_int(
        _require_key(attrs, "linear_solver_iterations", f"{label}.attrs"),
        f"{label}.attrs.linear_solver_iterations",
    )
    if linear_solver_iterations < 0:
        raise ValueError(f"{label}.attrs.linear_solver_iterations must be non-negative")
    gradient_norm = None
    if "gradient_norm" in attrs:
        gradient_norm = _coerce_trace_float(
            attrs["gradient_norm"], f"{label}.attrs.gradient_norm"
        )
    return GlobalPositioningIterationMetric(
        schema_version=event.schema_version,
        run_id=event.run_id,
        seq=event.seq,
        event_type=event.event_type,
        stage=event.stage,
        iteration=event.iteration,
        timestamp_ns=event.timestamp_ns,
        step_is_successful=_require_bool(
            _require_key(attrs, "step_is_successful", f"{label}.attrs"),
            f"{label}.attrs.step_is_successful",
        ),
        cost=_coerce_trace_float(
            _require_key(attrs, "cost", f"{label}.attrs"), f"{label}.attrs.cost"
        ),
        cost_change=_coerce_trace_float(
            _require_key(attrs, "cost_change", f"{label}.attrs"),
            f"{label}.attrs.cost_change",
        ),
        gradient_norm=gradient_norm,
        gradient_max_norm=_coerce_trace_float(
            _require_key(attrs, "gradient_max_norm", f"{label}.attrs"),
            f"{label}.attrs.gradient_max_norm",
        ),
        step_norm=_coerce_trace_float(
            _require_key(attrs, "step_norm", f"{label}.attrs"),
            f"{label}.attrs.step_norm",
        ),
        trust_region_radius=_coerce_trace_float(
            _require_key(attrs, "trust_region_radius", f"{label}.attrs"),
            f"{label}.attrs.trust_region_radius",
        ),
        linear_solver_iterations=linear_solver_iterations,
        iteration_time_sec=_coerce_trace_float(
            _require_key(attrs, "iteration_time_sec", f"{label}.attrs"),
            f"{label}.attrs.iteration_time_sec",
        ),
        cumulative_time_sec=_coerce_trace_float(
            _require_key(attrs, "cumulative_time_sec", f"{label}.attrs"),
            f"{label}.attrs.cumulative_time_sec",
        ),
        attrs=attrs,
    )


def _read_optional_trace_events(
    directory: Path,
    filename: str,
    *,
    expected_run_id: str | None,
) -> tuple[GlobalPositioningTraceEvent, ...]:
    path = directory / filename
    if not path.exists():
        return ()
    return tuple(
        _parse_trace_event_record(
            record,
            path=path,
            idx=idx,
            expected_run_id=expected_run_id,
        )
        for idx, record in enumerate(_iter_jsonl(path))
    )


def _read_optional_iteration_metrics(
    directory: Path,
    filename: str,
    *,
    expected_run_id: str | None,
) -> tuple[GlobalPositioningIterationMetric, ...]:
    path = directory / filename
    if not path.exists():
        return ()
    metrics = tuple(
        _parse_iteration_metric_record(
            record,
            path=path,
            idx=idx,
            expected_run_id=expected_run_id,
        )
        for idx, record in enumerate(_iter_jsonl(path))
    )
    seen_iterations: set[int] = set()
    for metric in metrics:
        if metric.iteration in seen_iterations:
            raise ValueError(f"{path}: duplicate iteration metric {metric.iteration}")
        seen_iterations.add(metric.iteration)
    return metrics


def _require_float_vector(value: Any, label: str, size: int) -> np.ndarray:
    if not isinstance(value, list):
        raise TypeError(f"{label}: expected list")
    if len(value) != size:
        raise ValueError(f"{label}: expected length {size}, got {len(value)}")
    return np.asarray(
        [
            _coerce_trace_float(item, f"{label}[{idx}]")
            for idx, item in enumerate(value)
        ],
        dtype=np.float64,
    )


def _require_fixed_vector(
    fixed_parameters: dict[str, Any], key: str, residual_id: str, size: int
) -> np.ndarray:
    return _require_float_vector(
        _require_key(fixed_parameters, key, f"residual {residual_id} fixed_parameters"),
        f"residual {residual_id} fixed_parameters.{key}",
        size,
    )


def _require_fixed_float(
    fixed_parameters: dict[str, Any], key: str, residual_id: str
) -> float:
    return _coerce_trace_float(
        _require_key(fixed_parameters, key, f"residual {residual_id} fixed_parameters"),
        f"residual {residual_id} fixed_parameters.{key}",
    )


def _require_fixed_bool(
    fixed_parameters: dict[str, Any], key: str, residual_id: str
) -> bool:
    return _require_bool(
        _require_key(fixed_parameters, key, f"residual {residual_id} fixed_parameters"),
        f"residual {residual_id} fixed_parameters.{key}",
    )


def _snapshot_lookup(
    array: GlobalPositioningSnapshotArray | None,
    label: str,
) -> dict[int, np.ndarray]:
    if array is None:
        return {}
    if len(array.ids) != array.shape[0]:
        raise ValueError(f"{label}: missing artifact ids for replay")
    return {
        item_id: np.atleast_1d(np.asarray(array.values[idx], dtype=np.float64))
        for idx, item_id in enumerate(array.ids)
    }


def _rotation_matrix_from_quaternion_wxyz(value: np.ndarray, label: str) -> np.ndarray:
    if value.shape != (4,):
        raise ValueError(f"{label}: expected quaternion shape (4,), got {value.shape}")
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"{label}: quaternion norm must be positive and finite")
    w, x, y, z = value / norm
    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def _left_sqrt_information(covariance: np.ndarray, label: str) -> np.ndarray:
    if covariance.shape != (3, 3):
        raise ValueError(f"{label}: expected shape (3, 3), got {covariance.shape}")
    try:
        return np.linalg.cholesky(np.linalg.inv(covariance)).T
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{label}: covariance must be positive definite") from exc


def _loss_rho(loss: GlobalPositioningResidualLedgerLoss, sq_norm: float) -> np.ndarray:
    if sq_norm < 0.0 or not np.isfinite(sq_norm):
        raise ValueError(f"loss input must be finite and non-negative, got {sq_norm}")
    loss_type = loss.type
    scale = 1.0 if loss.scale is None else loss.scale
    weight = 1.0 if loss.weight is None else loss.weight
    if scale <= 0.0:
        raise ValueError(f"loss scale must be positive, got {scale}")
    if weight < 0.0:
        raise ValueError(f"loss weight must be non-negative, got {weight}")

    if loss_type == "trivial":
        rho = np.array([sq_norm, 1.0, 0.0], dtype=np.float64)
    elif loss_type == "huber":
        threshold = scale * scale
        if sq_norm <= threshold:
            rho = np.array([sq_norm, 1.0, 0.0], dtype=np.float64)
        else:
            root = math.sqrt(sq_norm)
            rho = np.array(
                [
                    2.0 * scale * root - threshold,
                    scale / root,
                    -0.5 * scale / (sq_norm * root),
                ],
                dtype=np.float64,
            )
    elif loss_type == "soft_l1":
        z = 1.0 + sq_norm / (scale * scale)
        root = math.sqrt(z)
        rho = np.array(
            [
                2.0 * scale * scale * (root - 1.0),
                1.0 / root,
                -0.5 / (scale * scale * z * root),
            ],
            dtype=np.float64,
        )
    elif loss_type == "cauchy":
        z = 1.0 + sq_norm / (scale * scale)
        rho = np.array(
            [
                scale * scale * math.log(z),
                1.0 / z,
                -1.0 / (scale * scale * z * z),
            ],
            dtype=np.float64,
        )
    else:
        raise ValueError(f"unsupported loss type {loss_type!r}")
    return weight * rho


class _ReplaySnapshotValues:
    def __init__(self, snapshot: GlobalPositioningParameterSnapshot):
        self.by_kind = {
            "frame_center": _snapshot_lookup(snapshot.frame_centers, "frame_centers"),
            "point3D": _snapshot_lookup(snapshot.points3D, "points3D"),
            "bata_scale": _snapshot_lookup(snapshot.scales, "scales"),
            "dmap_scale": _snapshot_lookup(snapshot.dmap_scales, "dmap_scales"),
            "cam_in_rig": _snapshot_lookup(snapshot.cams_in_rig, "cams_in_rig"),
        }

    def value(self, block: GlobalPositioningResidualLedgerParameterBlock) -> np.ndarray:
        if block.kind not in self.by_kind:
            raise ValueError(f"unsupported parameter block kind {block.kind!r}")
        values = self.by_kind[block.kind]
        if block.id not in values:
            raise KeyError(
                "snapshot is missing parameter block "
                f"kind={block.kind!r}, id={block.id}"
            )
        value = np.asarray(values[block.id], dtype=np.float64)
        if value.shape != (block.size,):
            raise ValueError(
                f"parameter block kind={block.kind!r}, id={block.id}: "
                f"expected shape ({block.size},), got {value.shape}"
            )
        return value.copy()


def _select_replay_ledger_blocks(
    ledger_blocks: tuple[GlobalPositioningResidualLedgerBlock, ...],
    residual_ids: str | list[str] | tuple[str, ...] | None,
) -> tuple[GlobalPositioningResidualLedgerBlock, ...]:
    if residual_ids is None:
        return ledger_blocks
    requested_ids = (
        (residual_ids,) if isinstance(residual_ids, str) else tuple(residual_ids)
    )
    if not requested_ids:
        raise ValueError("residual_ids must be non-empty when provided")
    duplicate_ids = sorted(
        {
            residual_id
            for residual_id in requested_ids
            if requested_ids.count(residual_id) > 1
        }
    )
    if duplicate_ids:
        raise ValueError(f"residual_ids contains duplicates: {duplicate_ids}")

    by_id = {block.residual_id: block for block in ledger_blocks}
    missing_ids = [
        residual_id for residual_id in requested_ids if residual_id not in by_id
    ]
    if missing_ids:
        raise KeyError(f"trace is missing replay residual ids: {missing_ids}")
    return tuple(by_id[residual_id] for residual_id in requested_ids)


class GlobalPositioningTraceReplay:
    def __init__(
        self,
        trace: GlobalPositioningTrace,
        *,
        iteration: int,
        compute_jacobians: bool = False,
        residual_ids: str | list[str] | tuple[str, ...] | None = None,
    ):
        self.trace = trace
        self.iteration = iteration
        self.compute_jacobians = compute_jacobians
        ledger_blocks = trace.residual_ledger_blocks
        if not ledger_blocks:
            raise ValueError("trace has no typed residual_ledger_blocks for replay")
        self.ledger_blocks = _select_replay_ledger_blocks(ledger_blocks, residual_ids)
        self.snapshot = trace.snapshot(iteration)
        self.snapshot_values = _ReplaySnapshotValues(self.snapshot)

    def evaluate(self) -> GlobalPositioningReplayEvaluation:
        raw_residual_blocks = []
        raw_costs = []
        robust_costs = []
        loss_rhos = []
        residual_ids = []
        residual_dims = []
        residual_offsets = []
        evaluation_success = []
        parameter_blocks = []
        jacobian_blocks_by_residual = []
        scalar_offset = 0
        jacobian_offset = 0

        for block in self.ledger_blocks:
            residual = self._evaluate_block(block)
            if residual.ndim != 1:
                raise ValueError(
                    f"residual {block.residual_id}: expected vector residual"
                )
            raw_residual_blocks.append(residual)
            raw_cost = 0.5 * float(residual @ residual)
            rho = _loss_rho(block.loss, 2.0 * raw_cost)

            residual_ids.append(block.residual_id)
            residual_dims.append(int(residual.size))
            residual_offsets.append(scalar_offset)
            evaluation_success.append(True)
            raw_costs.append(raw_cost)
            loss_rhos.append(rho)
            robust_costs.append(0.5 * float(rho[0]))
            parameter_blocks.append(block.parameter_blocks)
            scalar_offset += int(residual.size)

            residual_jacobian_blocks = ()
            if self.compute_jacobians:
                finite_diff_blocks = []
                for parameter_block in block.parameter_blocks:
                    jacobian = self._finite_difference_jacobian(block, parameter_block)
                    finite_diff_blocks.append(
                        GlobalPositioningReplayJacobianBlock(
                            parameter_block=parameter_block,
                            offset=jacobian_offset,
                            residual_dim=int(residual.size),
                            values=jacobian,
                        )
                    )
                    jacobian_offset += int(residual.size) * parameter_block.size
                residual_jacobian_blocks = tuple(finite_diff_blocks)
            jacobian_blocks_by_residual.append(residual_jacobian_blocks)

        raw_residuals = (
            np.concatenate(raw_residual_blocks)
            if raw_residual_blocks
            else np.empty((0,), dtype=np.float64)
        )
        return GlobalPositioningReplayEvaluation(
            iteration=self.iteration,
            raw_residuals=raw_residuals,
            raw_costs=np.asarray(raw_costs, dtype=np.float64),
            loss_rho_values=np.asarray(loss_rhos, dtype=np.float64).reshape((-1, 3)),
            robust_costs=np.asarray(robust_costs, dtype=np.float64),
            evaluation_success=tuple(evaluation_success),
            residual_ids=tuple(residual_ids),
            residual_dims=tuple(residual_dims),
            residual_offsets=tuple(residual_offsets),
            parameter_blocks=tuple(parameter_blocks),
            raw_jacobians=(
                tuple(jacobian_blocks_by_residual) if self.compute_jacobians else ()
            ),
        )

    def _parameter_values(
        self,
        block: GlobalPositioningResidualLedgerBlock,
        overrides: dict[tuple[str, int], np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        values = {}
        seen_roles = set()
        overrides = overrides or {}
        for parameter_block in block.parameter_blocks:
            if parameter_block.role in seen_roles:
                raise ValueError(
                    f"residual {block.residual_id}: duplicate parameter role "
                    f"{parameter_block.role!r}"
                )
            seen_roles.add(parameter_block.role)
            key = (parameter_block.kind, parameter_block.id)
            values[parameter_block.role] = (
                np.asarray(overrides[key], dtype=np.float64).copy()
                if key in overrides
                else self.snapshot_values.value(parameter_block)
            )
        return values

    def _evaluate_block(
        self,
        block: GlobalPositioningResidualLedgerBlock,
        overrides: dict[tuple[str, int], np.ndarray] | None = None,
    ) -> np.ndarray:
        residual_type = _require_str(
            _require_key(
                block.attrs,
                "residual_type",
                f"residual {block.residual_id} attrs",
            ),
            f"residual {block.residual_id} attrs.residual_type",
        )
        values = self._parameter_values(block, overrides)
        fixed = block.fixed_parameters

        if residual_type == "bata_ref_frame":
            residual = _require_fixed_vector(
                fixed, "cam_from_point3D_dir", block.residual_id, 3
            ) - values["bata_scale"][0] * (values["point3D"] - values["frame_center"])
            if "keypoint_covariance_world_row_major" in fixed:
                covariance = _require_fixed_vector(
                    fixed,
                    "keypoint_covariance_world_row_major",
                    block.residual_id,
                    9,
                ).reshape((3, 3))
                residual = (
                    _left_sqrt_information(
                        covariance,
                        f"residual {block.residual_id} "
                        "keypoint_covariance_world_row_major",
                    )
                    @ residual
                )
            return residual

        if residual_type == "bata_constant_rig":
            return _require_fixed_vector(
                fixed, "cam_from_point3D_dir", block.residual_id, 3
            ) - values["bata_scale"][0] * (
                values["point3D"]
                - values["frame_center"]
                + _require_fixed_vector(fixed, "cam_from_rig_dir", block.residual_id, 3)
            )

        if residual_type == "bata_variable_rig":
            world_from_rig = _rotation_matrix_from_quaternion_wxyz(
                _require_fixed_vector(
                    fixed, "world_from_rig_rotation_wxyz", block.residual_id, 4
                ),
                f"residual {block.residual_id} world_from_rig_rotation_wxyz",
            )
            cam_from_rig_translation = world_from_rig @ values["cam_in_rig"]
            return _require_fixed_vector(
                fixed, "cam_from_point3D_dir", block.residual_id, 3
            ) - values["bata_scale"][0] * (
                values["point3D"] - values["frame_center"] - cam_from_rig_translation
            )

        if residual_type == "metric_depth":
            return self._evaluate_metric_depth(block, values)

        if residual_type == "scale_prior":
            stddev = _require_fixed_float(
                fixed, "scale_prior_stddev", block.residual_id
            )
            if stddev <= 1e-9:
                raise ValueError(
                    f"residual {block.residual_id}: "
                    "scale_prior_stddev must be positive"
                )
            target = _require_fixed_float(
                fixed, "scale_prior_target", block.residual_id
            )
            return np.asarray(
                [(values["dmap_scale"][0] - target) / stddev], dtype=np.float64
            )

        raise ValueError(f"unsupported residual_type {residual_type!r}")

    def _evaluate_metric_depth(
        self,
        block: GlobalPositioningResidualLedgerBlock,
        values: dict[str, np.ndarray],
    ) -> np.ndarray:
        fixed = block.fixed_parameters
        rotation = _rotation_matrix_from_quaternion_wxyz(
            _require_fixed_vector(fixed, "camera_rotation_wxyz", block.residual_id, 4),
            f"residual {block.residual_id} camera_rotation_wxyz",
        )
        depth_prior = _coerce_trace_float(
            _require_key(
                block.attrs,
                "depth_prior",
                f"residual {block.residual_id} attrs",
            ),
            f"residual {block.residual_id} attrs.depth_prior",
        )
        sigma_depth = _coerce_trace_float(
            _require_key(
                block.attrs,
                "depth_sigma",
                f"residual {block.residual_id} attrs",
            ),
            f"residual {block.residual_id} attrs.depth_sigma",
        )
        if depth_prior <= 0.0:
            raise ValueError(
                f"residual {block.residual_id}: depth_prior must be positive"
            )
        if sigma_depth <= 1e-9:
            raise ValueError(
                f"residual {block.residual_id}: depth_sigma must be positive"
            )

        point_vec_cam = rotation @ (values["point3D"] - values["frame_center"])
        z_est = float(point_vec_cam[2])
        use_log_scale = _require_fixed_bool(
            fixed, "metric_depth_use_log_scale", block.residual_id
        )
        dmap_value = float(values["dmap_scale"][0])
        scale = math.exp(dmap_value) if use_log_scale else dmap_value
        scaled_prior = scale * depth_prior
        scaled_sigma = scale * sigma_depth

        if (
            _require_fixed_bool(
                fixed, "metric_depth_zero_residual_behind", block.residual_id
            )
            and z_est <= 0.0
        ):
            return np.asarray([0.0], dtype=np.float64)

        residual_type = _require_str(
            _require_key(
                fixed,
                "metric_depth_residual_type",
                f"residual {block.residual_id} fixed_parameters",
            ),
            f"residual {block.residual_id} "
            "fixed_parameters.metric_depth_residual_type",
        )
        if residual_type != "linear":
            depth_prior_safe = max(depth_prior, 1e-6)
            sigma_log = sigma_depth / depth_prior_safe
            weight_log = 1.0 / max(1e-6, sigma_log)
            if residual_type == "log_linear":
                threshold = _require_fixed_float(
                    fixed,
                    "metric_depth_log_linear_threshold",
                    block.residual_id,
                )
                if threshold <= 0.0:
                    raise ValueError(
                        f"residual {block.residual_id}: "
                        "metric_depth_log_linear_threshold must be positive"
                    )
                scaled_prior_safe = max(scaled_prior, 1e-6)
                if z_est > threshold:
                    r_depth = math.log(max(z_est, 1e-6) / scaled_prior_safe)
                else:
                    r_depth = (
                        math.log(threshold / scaled_prior_safe)
                        + (z_est - threshold) / threshold
                    )
                weight = weight_log
            elif residual_type == "log":
                if z_est > 0.0:
                    r_depth = math.log(max(z_est, 1e-6) / max(scaled_prior, 1e-6))
                    weight = weight_log
                else:
                    r_depth = z_est - scaled_prior
                    weight = 1.0 / max(1e-6, scaled_sigma)
            else:
                raise ValueError(
                    f"unsupported metric_depth_residual_type {residual_type!r}"
                )
        else:
            r_depth = z_est - scaled_prior
            weight = 1.0 / max(1e-6, scaled_sigma)
        return np.asarray([weight * r_depth], dtype=np.float64)

    def _finite_difference_jacobian(
        self,
        block: GlobalPositioningResidualLedgerBlock,
        parameter_block: GlobalPositioningResidualLedgerParameterBlock,
    ) -> np.ndarray:
        key = (parameter_block.kind, parameter_block.id)
        base_value = self.snapshot_values.value(parameter_block)
        base_residual = self._evaluate_block(block)
        jacobian = np.empty(
            (base_residual.size, parameter_block.size), dtype=np.float64
        )
        for col in range(parameter_block.size):
            step = 1e-6 * max(1.0, abs(float(base_value[col])))
            plus = base_value.copy()
            minus = base_value.copy()
            plus[col] += step
            minus[col] -= step
            plus_residual = self._evaluate_block(block, {key: plus})
            minus_residual = self._evaluate_block(block, {key: minus})
            jacobian[:, col] = (plus_residual - minus_residual) / (2.0 * step)
        return jacobian


class _RawBinaryResidualBlock:
    def __init__(self, residual_values: _RawBinaryResidualValues, index: int):
        self._residual_values = residual_values
        self.index = index

    @property
    def residual_id(self) -> str:
        return self._residual_values.residual_ids[self.index]

    @property
    def residual_dim(self) -> int:
        return self._residual_values.residual_dims[self.index]

    @property
    def residual_offset(self) -> int:
        return self._residual_values.residual_offsets[self.index]

    @property
    def evaluation_success(self) -> bool:
        return self._residual_values.evaluation_success[self.index]

    @property
    def raw_residuals(self) -> np.ndarray:
        begin = self.residual_offset
        end = begin + self.residual_dim
        return self._residual_values.raw_residuals[begin:end]

    @property
    def raw_cost(self) -> float:
        return float(self._residual_values.raw_costs[self.index])

    @property
    def robust_cost(self) -> float:
        return float(self._residual_values.robust_costs[self.index])

    @property
    def loss_rho(self) -> np.ndarray:
        return self._residual_values.loss_rho_values[self.index]

    @property
    def loss_rho0(self) -> float:
        return float(self.loss_rho[0])

    @property
    def loss_rho1(self) -> float:
        return float(self.loss_rho[1])

    @property
    def loss_rho2(self) -> float:
        return float(self.loss_rho[2])

    @property
    def loss_derivative_scale(self) -> float:
        return self.loss_rho1

    @property
    def parameter_blocks(self) -> tuple[GlobalPositioningParameterBlock, ...]:
        if not self._residual_values.has_raw_jacobians:
            return ()
        return tuple(self._residual_values.parameter_blocks[self.index])

    @property
    def jacobian_blocks(self) -> tuple[GlobalPositioningJacobianBlock, ...]:
        if not self._residual_values.has_raw_jacobians:
            return ()
        raw_jacobians = self._residual_values.raw_jacobians
        if raw_jacobians is None:
            return ()

        blocks = []
        for parameter_block, offset in zip(
            self._residual_values.parameter_blocks[self.index],
            self._residual_values.raw_jacobian_offsets[self.index],
            strict=True,
        ):
            end = offset + self.residual_dim * parameter_block.size
            values = raw_jacobians[offset:end].reshape(
                (self.residual_dim, parameter_block.size)
            )
            blocks.append(
                GlobalPositioningJacobianBlock(
                    parameter_block, offset, self.residual_dim, values
                )
            )
        return tuple(blocks)

    def jacobian(self, block: int | str, *, id: int | None = None) -> np.ndarray:
        jacobian_blocks = self.jacobian_blocks
        if isinstance(block, int):
            return jacobian_blocks[block].values
        matches = [
            item
            for item in jacobian_blocks
            if item.parameter_block.role == block
            and (id is None or item.parameter_block.id == id)
        ]
        if len(matches) != 1:
            raise KeyError(
                f"Expected exactly one Jacobian block for role={block!r}, "
                f"id={id!r}; found {len(matches)}"
            )
        return matches[0].values


class _RawBinaryResidualValues:
    def __init__(
        self,
        path: Path,
        *,
        expected_iteration: int | None = None,
        expected_residual_ids: tuple[str, ...] | None = None,
    ):
        self.path = Path(path)
        self.metadata_path = self.path
        self.metadata = {
            "schema_version": 1,
            "storage_format": _RAW_BINARY_STORAGE_FORMAT,
            "artifacts": {
                "residual_values": {
                    "file": self.path.name,
                    "dtype": "float64",
                    "byte_order": "little_endian",
                }
            },
        }
        with self.path.open("rb") as stream:
            _read_magic(stream, _RAW_RESIDUAL_VALUES_MAGIC, str(self.path))
            (version,) = _read_struct(stream, "I", str(self.path))
            if version == 1:
                iteration, num_residuals, total_scalar_residuals, has_loss_rho = (
                    _read_struct(stream, "qQQ?", str(self.path))
                )
                has_raw_jacobians = False
            elif version == 2:
                (
                    iteration,
                    num_residuals,
                    total_scalar_residuals,
                    has_loss_rho,
                    has_raw_jacobians,
                ) = _read_struct(stream, "qQQ??", str(self.path))
            else:
                raise ValueError(
                    f"{self.path}: unsupported residual-values schema version {version}"
                )
            if expected_iteration is not None and iteration != expected_iteration:
                raise ValueError(
                    f"{self.path}: iteration {iteration}, expected {expected_iteration}"
                )
            self.iteration = int(iteration)
            self.num_residual_blocks = int(num_residuals)
            self.total_scalar_residuals = int(total_scalar_residuals)
            self.has_raw_jacobians = bool(has_raw_jacobians)
            self.has_loss_rho_values = bool(has_loss_rho)
            self.residual_ids = []
            self.residual_dims = []
            self.residual_offsets = []
            self.evaluation_success = []
            for idx in range(self.num_residual_blocks):
                label = f"{self.path}: residual_values[{idx}]"
                self.residual_ids.append(
                    _read_binary_string(stream, f"{label}.residual_id")
                )
                residual_dim, residual_offset, success = _read_struct(
                    stream, "IQ?", label
                )
                self.residual_dims.append(int(residual_dim))
                self.residual_offsets.append(int(residual_offset))
                self.evaluation_success.append(bool(success))

            self._validate_residual_structure()
            if (
                expected_residual_ids is not None
                and tuple(self.residual_ids) != expected_residual_ids
            ):
                raise ValueError(
                    "raw binary residual_values.residual_ids does not match residual ledger order"
                )
            self._residual_id_to_index = {
                residual_id: idx for idx, residual_id in enumerate(self.residual_ids)
            }

            raw_residuals = np.frombuffer(
                _read_exact(
                    stream,
                    self.total_scalar_residuals * 8,
                    f"{self.path}: raw_residuals",
                ),
                dtype="<f8",
            ).copy()
            raw_costs = np.frombuffer(
                _read_exact(
                    stream, self.num_residual_blocks * 8, f"{self.path}: raw_costs"
                ),
                dtype="<f8",
            ).copy()
            robust_costs = np.frombuffer(
                _read_exact(
                    stream, self.num_residual_blocks * 8, f"{self.path}: robust_costs"
                ),
                dtype="<f8",
            ).copy()
            if self.has_loss_rho_values:
                loss_rho_values = np.frombuffer(
                    _read_exact(
                        stream,
                        self.num_residual_blocks * 3 * 8,
                        f"{self.path}: loss_rho_values",
                    ),
                    dtype="<f8",
                ).copy()
                self._loss_rho_values = loss_rho_values.reshape(
                    (self.num_residual_blocks, 3)
                )
            else:
                self._loss_rho_values = np.full(
                    (self.num_residual_blocks, 3), np.nan, dtype=np.float64
                )
            self.total_jacobian_scalars = 0
            self.parameter_blocks: list[list[GlobalPositioningParameterBlock]] = []
            self.raw_jacobian_offsets: list[list[int]] = []
            raw_jacobians = None
            if self.has_raw_jacobians:
                (total_jacobian_scalars,) = _read_struct(
                    stream, "Q", f"{self.path}: total_jacobian_scalars"
                )
                self.total_jacobian_scalars = int(total_jacobian_scalars)
                expected_jacobian_offset = 0
                for residual_idx in range(self.num_residual_blocks):
                    label = f"{self.path}: residual_values[{residual_idx}].jacobians"
                    (num_parameter_blocks,) = _read_struct(
                        stream, "I", f"{label}.num_parameter_blocks"
                    )
                    residual_parameter_blocks = []
                    residual_offsets = []
                    for block_idx in range(int(num_parameter_blocks)):
                        block_label = f"{label}.parameter_blocks[{block_idx}]"
                        role = _read_binary_string(stream, f"{block_label}.role")
                        kind = _read_binary_string(stream, f"{block_label}.kind")
                        block_id, block_size, raw_jacobian_offset, is_constant = (
                            _read_struct(stream, "QIQ?", block_label)
                        )
                        block_size = int(block_size)
                        raw_jacobian_offset = int(raw_jacobian_offset)
                        if not role:
                            raise ValueError(f"{block_label}.role must be non-empty")
                        if not kind:
                            raise ValueError(f"{block_label}.kind must be non-empty")
                        if block_size <= 0:
                            raise ValueError(f"{block_label}.size must be positive")
                        if raw_jacobian_offset != expected_jacobian_offset:
                            raise ValueError(
                                f"{block_label}.raw_jacobian_offset is "
                                f"{raw_jacobian_offset}, expected "
                                f"{expected_jacobian_offset}"
                            )
                        lower_bounds = np.frombuffer(
                            _read_exact(
                                stream,
                                block_size * 8,
                                f"{block_label}.lower_bounds",
                            ),
                            dtype="<f8",
                        ).copy()
                        residual_parameter_blocks.append(
                            GlobalPositioningParameterBlock(
                                role=role,
                                kind=kind,
                                id=int(block_id),
                                size=block_size,
                                is_constant=bool(is_constant),
                                lower_bounds=tuple(
                                    float(value) for value in lower_bounds.tolist()
                                ),
                            )
                        )
                        residual_offsets.append(raw_jacobian_offset)
                        expected_jacobian_offset += (
                            self.residual_dims[residual_idx] * block_size
                        )
                    self.parameter_blocks.append(residual_parameter_blocks)
                    self.raw_jacobian_offsets.append(residual_offsets)
                if expected_jacobian_offset != self.total_jacobian_scalars:
                    raise ValueError(
                        f"{self.path}: Jacobian offset sum is "
                        f"{expected_jacobian_offset}, expected "
                        f"total_jacobian_scalars {self.total_jacobian_scalars}"
                    )
                raw_jacobians = np.frombuffer(
                    _read_exact(
                        stream,
                        self.total_jacobian_scalars * 8,
                        f"{self.path}: raw_jacobians",
                    ),
                    dtype="<f8",
                ).copy()
            trailing = stream.read(1)
            if trailing:
                raise ValueError(f"{self.path}: unexpected trailing bytes")

        self._raw_residuals = raw_residuals
        self._raw_costs = raw_costs
        self._robust_costs = robust_costs
        self._raw_jacobians = raw_jacobians
        self.metadata["has_raw_jacobians"] = self.has_raw_jacobians
        if self.has_raw_jacobians:
            self.metadata["total_jacobian_scalars"] = self.total_jacobian_scalars
            self.metadata["parameter_block_sizes"] = [
                [block.size for block in residual_blocks]
                for residual_blocks in self.parameter_blocks
            ]
            self.metadata["raw_jacobian_offsets"] = self.raw_jacobian_offsets
            self.metadata["raw_jacobian_layout"] = (
                "residual_block_major/parameter_block_major/row_major"
            )
            self.metadata["jacobian_domain"] = "raw_cost_function_ambient_parameters"
            self.metadata["loss_applied_to_jacobians"] = False
            self.metadata["manifold_applied_to_jacobians"] = False
            self.metadata["constant_parameter_blocks_included"] = True
        if self.has_loss_rho_values:
            self._validate_loss_rho_costs()

    def _validate_residual_structure(self) -> None:
        expected_offset = 0
        for idx, (residual_id, residual_dim, residual_offset) in enumerate(
            zip(
                self.residual_ids,
                self.residual_dims,
                self.residual_offsets,
                strict=True,
            )
        ):
            if not residual_id:
                raise ValueError(f"{self.path}: residual_ids[{idx}] must be non-empty")
            if residual_dim < 0:
                raise ValueError(
                    f"{self.path}: residual_dims[{idx}] must be non-negative"
                )
            if residual_offset != expected_offset:
                raise ValueError(
                    f"{self.path}: residual_offsets[{idx}] is {residual_offset}, expected {expected_offset}"
                )
            expected_offset += residual_dim
        if len(set(self.residual_ids)) != len(self.residual_ids):
            raise ValueError(f"{self.path}: residual_ids must be unique")
        if expected_offset != self.total_scalar_residuals:
            raise ValueError(
                f"{self.path}: sum(residual_dims) is {expected_offset}, "
                f"expected total_scalar_residuals {self.total_scalar_residuals}"
            )

    def _validate_loss_rho_costs(self) -> None:
        success_mask = np.asarray(self.evaluation_success, dtype=bool)
        if not np.any(success_mask):
            return
        expected_robust_costs = 0.5 * self._loss_rho_values[:, 0]
        matches = np.isclose(
            self._robust_costs,
            expected_robust_costs,
            rtol=1e-10,
            atol=1e-12,
            equal_nan=False,
        )
        mismatch_mask = success_mask & ~matches
        if np.any(mismatch_mask):
            mismatch_idx = int(np.flatnonzero(mismatch_mask)[0])
            raise ValueError(
                "robust_costs must equal 0.5 * loss_rho_values[:, 0] "
                "for successful residual evaluations; mismatch at "
                f"residual {mismatch_idx} "
                f"({self.residual_ids[mismatch_idx]!r})"
            )

    @property
    def raw_residuals(self) -> np.ndarray:
        return self._raw_residuals

    @property
    def raw_costs(self) -> np.ndarray:
        return self._raw_costs

    @property
    def robust_costs(self) -> np.ndarray:
        return self._robust_costs

    @property
    def loss_rho_values(self) -> np.ndarray:
        if not self.has_loss_rho_values:
            raise ValueError(
                "loss_rho_values artifact is not present in this raw binary trace"
            )
        return self._loss_rho_values

    @property
    def raw_jacobians(self) -> np.ndarray | None:
        return self._raw_jacobians

    def residual(self, residual: int | str) -> _RawBinaryResidualBlock:
        if isinstance(residual, str):
            try:
                residual = self._residual_id_to_index[residual]
            except KeyError as exc:
                raise KeyError(f"Unknown residual id {residual!r}") from exc
        if residual < 0 or residual >= self.num_residual_blocks:
            raise IndexError(f"Residual index {residual} out of range")
        return _RawBinaryResidualBlock(self, residual)


def _read_raw_snapshot_array(
    path: Path,
    *,
    expected_name: str,
    max_rows: int | None = None,
) -> GlobalPositioningSnapshotArray:
    with path.open("rb") as stream:
        _read_magic(stream, _RAW_ARRAY_MAGIC, str(path))
        version, rows, cols = _read_struct(stream, "IQQ", str(path))
        if version != 1:
            raise ValueError(f"{path}: unsupported array schema version {version}")
        name = _read_binary_string(stream, f"{path}: name")
        if name != expected_name:
            raise ValueError(f"{path}: array name {name!r}, expected {expected_name!r}")
        ids = np.frombuffer(
            _read_exact(stream, rows * 8, f"{path}: ids"), dtype="<i8"
        ).copy()
        values = np.frombuffer(
            _read_exact(stream, rows * cols * 8, f"{path}: values"), dtype="<f8"
        ).copy()
        trailing = stream.read(1)
        if trailing:
            raise ValueError(f"{path}: unexpected trailing bytes")
    shape = (int(rows), int(cols)) if cols != 1 else (int(rows),)
    values = values.reshape(shape)
    ids_tuple = tuple(int(item) for item in ids.tolist())
    if max_rows is not None:
        row_count = min(max_rows, shape[0])
        values = values[:row_count]
        ids_tuple = ids_tuple[:row_count]
        shape = (row_count, *shape[1:])
    return GlobalPositioningSnapshotArray(ids=ids_tuple, shape=shape, values=values)


class _RawBinaryGlobalPositioningTrace:
    def __init__(self, path: Path, manifest: dict[str, Any]):
        self.path = Path(path)
        self.manifest = manifest
        self._validate_manifest()
        self._iterations = self._parse_iterations()
        self._residual_blocks = self._read_residual_blocks()
        self.residual_blocks = list(self._residual_blocks)
        self.residual_skips: list[dict[str, Any]] = []
        self._residual_ledger_blocks: (
            tuple[GlobalPositioningResidualLedgerBlock, ...] | None
        ) = None

    def _validate_manifest(self) -> None:
        schema_version = _require_int(
            _require_key(
                self.manifest, "schema_version", str(self.path / "manifest.json")
            ),
            "schema_version",
        )
        if schema_version != 1:
            raise ValueError(
                f"{self.path / 'manifest.json'}: unsupported schema_version {schema_version}"
            )
        storage_format = _require_str(
            _require_key(
                self.manifest, "storage_format", str(self.path / "manifest.json")
            ),
            "storage_format",
        )
        if storage_format != _RAW_BINARY_STORAGE_FORMAT:
            raise ValueError(
                f"{self.path / 'manifest.json'}: unsupported storage_format {storage_format!r}"
            )
        byte_order = _require_str(
            _require_key(self.manifest, "byte_order", str(self.path / "manifest.json")),
            "byte_order",
        )
        if byte_order != "little_endian":
            raise ValueError(
                f"{self.path / 'manifest.json'}: byte_order must be little_endian"
            )
        dtype = _require_str(
            _require_key(self.manifest, "dtype", str(self.path / "manifest.json")),
            "dtype",
        )
        if dtype != "float64":
            raise ValueError(f"{self.path / 'manifest.json'}: dtype must be float64")

    def _parse_iterations(self) -> dict[int, dict[str, Any]]:
        iterations = _require_key(
            self.manifest, "iterations", str(self.path / "manifest.json")
        )
        if not isinstance(iterations, list):
            raise TypeError(f"{self.path / 'manifest.json'}: iterations must be a list")
        parsed = {}
        for idx, item in enumerate(iterations):
            if not isinstance(item, dict):
                raise TypeError(
                    f"{self.path / 'manifest.json'}: iterations[{idx}] must be an object"
                )
            iteration = _require_int(
                _require_key(item, "iteration", "iteration"), "iteration"
            )
            if iteration < 0:
                raise ValueError(
                    f"{self.path / 'manifest.json'}: iterations[{idx}].iteration must be non-negative"
                )
            if iteration in parsed:
                raise ValueError(
                    f"{self.path / 'manifest.json'}: duplicate iteration {iteration}"
                )
            directory = _require_str(
                _require_key(item, "directory", "iteration"), "directory"
            )
            relative = Path(directory)
            item = dict(item)
            item["_directory_relative_path"] = relative
            item["_directory_path"] = _resolve_raw_trace_path(
                self.path,
                relative,
                f"{self.path / 'manifest.json'}: iterations[{idx}].directory",
            )
            parsed[iteration] = item
        return parsed

    def _read_residual_blocks(self) -> tuple[dict[str, Any], ...]:
        static = _require_key(self.manifest, "static", str(self.path / "manifest.json"))
        if not isinstance(static, dict):
            raise TypeError(f"{self.path / 'manifest.json'}: static must be an object")
        ledger_name = _require_str(
            _require_key(static, "residual_ledger", "static"), "static.residual_ledger"
        )
        ledger_relative_path = Path(ledger_name)
        ledger_path = _resolve_raw_trace_path(
            self.path,
            ledger_relative_path,
            f"{self.path / 'manifest.json'}: static.residual_ledger",
        )
        with ledger_path.open("rb") as stream:
            _read_magic(stream, _RAW_LEDGER_MAGIC, str(ledger_path))
            version, count = _read_struct(stream, "IQ", str(ledger_path))
            if version != 1:
                raise ValueError(
                    f"{ledger_path}: unsupported residual ledger schema version {version}"
                )
            records = []
            for idx in range(count):
                label = f"{ledger_path}: residual_ledger[{idx}]"
                residual_id = _read_binary_string(stream, f"{label}.residual_id")
                residual_type = _read_binary_string(stream, f"{label}.residual_type")
                loss_bucket = _read_binary_string(stream, f"{label}.loss_bucket")
                frame_id, image_id, point3D_id, is_lc_observation = _read_struct(
                    stream, "qqq?", label
                )
                attrs = _read_binary_json(stream, f"{label}.attrs")
                attrs.update(
                    {
                        "residual_id": residual_id,
                        "residual_type": residual_type,
                        "loss_bucket": loss_bucket,
                        "frame_id": (
                            None if frame_id == _RAW_NONE_ID else int(frame_id)
                        ),
                        "image_id": (
                            None if image_id == _RAW_NONE_ID else int(image_id)
                        ),
                        "point3D_id": (
                            None if point3D_id == _RAW_NONE_ID else int(point3D_id)
                        ),
                        "is_lc_observation": bool(is_lc_observation),
                    }
                )
                records.append(
                    {
                        "event_type": "residual_added",
                        "stage": "problem_build",
                        "attrs": attrs,
                    }
                )
            trailing = stream.read(1)
            if trailing:
                raise ValueError(f"{ledger_path}: unexpected trailing bytes")
        return tuple(records)

    def raw_binary_artifact_paths(self) -> tuple[Path, ...]:
        paths = [_resolve_raw_trace_path(self.path, Path("manifest.json"), "manifest")]
        static = self.manifest.get("static")
        if isinstance(static, dict):
            ledger_name = static.get("residual_ledger")
            if isinstance(ledger_name, str):
                paths.append(
                    _resolve_raw_trace_path(
                        self.path,
                        Path(ledger_name),
                        "static.residual_ledger",
                    )
                )
        for iteration, metadata in sorted(self._iterations.items()):
            directory = metadata["_directory_relative_path"]
            for key in (
                "frame_centers",
                "point_xyz",
                "scales",
                "dmap_scales",
                "cams_in_rig",
                "residual_values",
                "jacobians",
            ):
                filename = metadata.get(key)
                if isinstance(filename, str):
                    paths.append(
                        _resolve_raw_trace_path(
                            self.path,
                            directory / filename,
                            f"iterations[{iteration}].{key}",
                        )
                    )
        return tuple(paths)

    @property
    def status(self) -> str:
        return _require_str(
            _require_key(self.manifest, "status", str(self.path / "manifest.json")),
            "status",
        )

    @property
    def trace_level(self) -> str:
        return _require_str(
            _require_key(
                self.manifest, "trace_level", str(self.path / "manifest.json")
            ),
            "trace_level",
        )

    @property
    def residual_value_iterations(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                iteration
                for iteration, metadata in self._iterations.items()
                if "residual_values" in metadata
            )
        )

    @property
    def snapshot_iterations(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                iteration
                for iteration, metadata in self._iterations.items()
                if "frame_centers" in metadata and "point_xyz" in metadata
            )
        )

    @property
    def events(self) -> tuple[GlobalPositioningTraceEvent, ...]:
        return ()

    @property
    def solver_events(self) -> tuple[GlobalPositioningTraceEvent, ...]:
        return ()

    @property
    def iteration_metrics(self) -> tuple[GlobalPositioningIterationMetric, ...]:
        return ()

    @property
    def iteration_metrics_by_iteration(
        self,
    ) -> dict[int, GlobalPositioningIterationMetric]:
        return {}

    @property
    def residual_ledger_blocks(
        self,
    ) -> tuple[GlobalPositioningResidualLedgerBlock, ...]:
        if self._residual_ledger_blocks is None:
            blocks = []
            for idx, record in enumerate(self.residual_blocks):
                block = _parse_residual_ledger_block(record, idx)
                if block is not None:
                    blocks.append(block)
            self._residual_ledger_blocks = tuple(blocks)
        return self._residual_ledger_blocks

    def _iteration_file(self, iteration: int, key: str) -> Path:
        if iteration not in self._iterations:
            raise KeyError(f"Trace has no iteration {iteration}")
        metadata = self._iterations[iteration]
        if key not in metadata:
            raise KeyError(f"Trace iteration {iteration} has no {key}")
        filename = _require_str(metadata[key], f"iterations[{iteration}].{key}")
        relative = Path(filename)
        if (
            relative.is_absolute()
            or relative.name != filename
            or ".." in relative.parts
        ):
            raise ValueError(f"iterations[{iteration}].{key}: expected bare filename")
        return _resolve_raw_trace_path(
            self.path,
            metadata["_directory_relative_path"] / relative,
            f"iterations[{iteration}].{key}",
        )

    def residual_values(self, iteration: int | None = None) -> _RawBinaryResidualValues:
        if iteration is None:
            iterations = self.residual_value_iterations
            if len(iterations) != 1:
                raise ValueError(
                    f"Trace has {len(iterations)} residual-value iterations; pass iteration explicitly"
                )
            iteration = iterations[0]
        return _RawBinaryResidualValues(
            self._iteration_file(iteration, "residual_values"),
            expected_iteration=iteration,
            expected_residual_ids=self._ledger_residual_ids(),
        )

    def _ledger_residual_ids(self) -> tuple[str, ...] | None:
        if not self.residual_blocks:
            return None
        return tuple(
            _require_str(
                _require_key(
                    _require_key(record, "attrs", f"residual_blocks[{idx}]"),
                    "residual_id",
                    f"residual_blocks[{idx}].attrs",
                ),
                f"residual_blocks[{idx}].attrs.residual_id",
            )
            for idx, record in enumerate(self.residual_blocks)
        )

    def snapshot(
        self, iteration: int, max_points: int | None = None
    ) -> GlobalPositioningParameterSnapshot:
        max_points = _normalize_max_points(max_points)
        metadata = self._iterations[iteration]
        frame_centers = _read_raw_snapshot_array(
            self._iteration_file(iteration, "frame_centers"),
            expected_name="frame_centers",
        )
        points3D = _read_raw_snapshot_array(
            self._iteration_file(iteration, "point_xyz"),
            expected_name="point_xyz",
            max_rows=max_points,
        )
        scales = (
            _read_raw_snapshot_array(
                self._iteration_file(iteration, "scales"), expected_name="scales"
            )
            if "scales" in metadata
            else _empty_snapshot_array()
        )
        dmap_scales = (
            _read_raw_snapshot_array(
                self._iteration_file(iteration, "dmap_scales"),
                expected_name="dmap_scales",
            )
            if "dmap_scales" in metadata
            else None
        )
        cams_in_rig = (
            _read_raw_snapshot_array(
                self._iteration_file(iteration, "cams_in_rig"),
                expected_name="cams_in_rig",
            )
            if "cams_in_rig" in metadata
            else None
        )
        return GlobalPositioningParameterSnapshot(
            metadata_path=self.path / "manifest.json",
            metadata=self.manifest,
            iteration=iteration,
            frame_centers=frame_centers,
            points3D=points3D,
            scales=scales,
            dmap_scales=dmap_scales,
            cams_in_rig=cams_in_rig,
        )

    def replay(
        self,
        iteration: int,
        *,
        compute_jacobians: bool = False,
        residual_ids: str | list[str] | tuple[str, ...] | None = None,
    ) -> GlobalPositioningReplayEvaluation:
        return GlobalPositioningTraceReplay(
            self,
            iteration=iteration,
            compute_jacobians=compute_jacobians,
            residual_ids=residual_ids,
        ).evaluate()


class GlobalPositioningTrace:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.manifest = _load_json(self.path / "manifest.json")
        expected_run_id = self._manifest_run_id()
        self._events = _read_optional_trace_events(
            self.path,
            "events.jsonl",
            expected_run_id=expected_run_id,
        )
        self._iteration_metrics = _read_optional_iteration_metrics(
            self.path,
            "iteration_metrics.jsonl",
            expected_run_id=expected_run_id,
        )
        self._iteration_metrics_by_iteration = {
            metric.iteration: metric for metric in self._iteration_metrics
        }
        self.residual_blocks = self._read_optional_jsonl("residual_blocks.jsonl")
        self.residual_skips = self._read_optional_jsonl("residual_skips.jsonl")
        self._residual_values_by_iteration = _discover_iteration_metadata(
            self.path / "residual_values"
        )
        self._snapshots_by_iteration = _discover_iteration_metadata(
            self.path / "snapshots"
        )
        self._residual_ledger_blocks: (
            tuple[GlobalPositioningResidualLedgerBlock, ...] | None
        ) = None

    @classmethod
    def load(
        cls, path: str | Path
    ) -> GlobalPositioningTrace | _RawBinaryGlobalPositioningTrace:
        path = Path(path)
        manifest = _load_json(path / "manifest.json")
        if manifest.get("storage_format") == _RAW_BINARY_STORAGE_FORMAT:
            return _RawBinaryGlobalPositioningTrace(path, manifest)
        return cls(path)

    def _read_optional_jsonl(self, filename: str) -> list[dict[str, Any]]:
        path = self.path / filename
        if not path.exists():
            return []
        return _iter_jsonl(path)

    def _ledger_residual_ids(self) -> tuple[str, ...] | None:
        if not self.residual_blocks:
            return None
        residual_ids = []
        for idx, record in enumerate(self.residual_blocks):
            attrs = _require_key(record, "attrs", f"residual_blocks[{idx}]")
            if not isinstance(attrs, dict):
                raise TypeError(f"residual_blocks[{idx}].attrs must be an object")
            residual_ids.append(
                _require_str(
                    _require_key(attrs, "residual_id", f"residual_blocks[{idx}].attrs"),
                    f"residual_blocks[{idx}].attrs.residual_id",
                )
            )
        return tuple(residual_ids)

    def _manifest_run_id(self) -> str | None:
        if "run_id" not in self.manifest:
            return None
        return _require_str(self.manifest["run_id"], str(self.path / "manifest.json"))

    @property
    def status(self) -> str:
        return _require_str(
            _require_key(self.manifest, "status", str(self.path / "manifest.json")),
            "status",
        )

    @property
    def trace_level(self) -> str:
        return _require_str(
            _require_key(
                self.manifest, "trace_level", str(self.path / "manifest.json")
            ),
            "trace_level",
        )

    @property
    def residual_value_iterations(self) -> tuple[int, ...]:
        return tuple(sorted(self._residual_values_by_iteration))

    @property
    def snapshot_iterations(self) -> tuple[int, ...]:
        return tuple(sorted(self._snapshots_by_iteration))

    @property
    def events(self) -> tuple[GlobalPositioningTraceEvent, ...]:
        return self._events

    @property
    def solver_events(self) -> tuple[GlobalPositioningTraceEvent, ...]:
        return tuple(event for event in self._events if event.stage == "ceres_solve")

    @property
    def iteration_metrics(
        self,
    ) -> tuple[GlobalPositioningIterationMetric, ...]:
        return self._iteration_metrics

    @property
    def iteration_metrics_by_iteration(
        self,
    ) -> dict[int, GlobalPositioningIterationMetric]:
        return dict(self._iteration_metrics_by_iteration)

    @property
    def residual_ledger_blocks(
        self,
    ) -> tuple[GlobalPositioningResidualLedgerBlock, ...]:
        if self._residual_ledger_blocks is None:
            blocks = []
            for idx, record in enumerate(self.residual_blocks):
                block = _parse_residual_ledger_block(record, idx)
                if block is not None:
                    blocks.append(block)
            self._residual_ledger_blocks = tuple(blocks)
        return self._residual_ledger_blocks

    def residual_values(
        self, iteration: int | None = None
    ) -> GlobalPositioningResidualValues:
        if iteration is None:
            iterations = self.residual_value_iterations
            if len(iterations) != 1:
                raise ValueError(
                    f"Trace has {len(iterations)} residual-value iterations; "
                    "pass iteration explicitly"
                )
            iteration = iterations[0]
        if iteration not in self._residual_values_by_iteration:
            raise KeyError(f"Trace has no residual values for iteration {iteration}")
        return GlobalPositioningResidualValues(
            self._residual_values_by_iteration[iteration],
            expected_iteration=iteration,
            expected_run_id=self._manifest_run_id(),
            expected_residual_ids=self._ledger_residual_ids(),
        )

    def snapshot(
        self, iteration: int, max_points: int | None = None
    ) -> GlobalPositioningParameterSnapshot:
        if iteration not in self._snapshots_by_iteration:
            raise KeyError(f"Trace has no parameter snapshot for iteration {iteration}")
        return GlobalPositioningParameterSnapshotLoader(
            self._snapshots_by_iteration[iteration],
            expected_iteration=iteration,
            expected_run_id=self._manifest_run_id(),
            max_points=max_points,
        ).load()

    def replay(
        self,
        iteration: int,
        *,
        compute_jacobians: bool = False,
        residual_ids: str | list[str] | tuple[str, ...] | None = None,
    ) -> GlobalPositioningReplayEvaluation:
        return GlobalPositioningTraceReplay(
            self,
            iteration=iteration,
            compute_jacobians=compute_jacobians,
            residual_ids=residual_ids,
        ).evaluate()
