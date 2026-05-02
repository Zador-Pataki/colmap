from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number}: expected a JSON object"
                )
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


def _require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label}: expected non-empty string")
    return value


def _require_int_list(value: Any, label: str) -> list[int]:
    if not isinstance(value, list):
        raise TypeError(f"{label}: expected list")
    return [
        _require_int(item, f"{label}[{idx}]") for idx, item in enumerate(value)
    ]


def _require_bool_list(value: Any, label: str) -> list[bool]:
    if not isinstance(value, list):
        raise TypeError(f"{label}: expected list")
    return [
        _require_bool(item, f"{label}[{idx}]") for idx, item in enumerate(value)
    ]


def _require_str_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{label}: expected list")
    return [
        _require_str(item, f"{label}[{idx}]") for idx, item in enumerate(value)
    ]


def _require_nested_int_list(value: Any, label: str) -> list[list[int]]:
    if not isinstance(value, list):
        raise TypeError(f"{label}: expected list")
    return [
        _require_int_list(item, f"{label}[{idx}]")
        for idx, item in enumerate(value)
    ]


def _require_nested_bool_list(value: Any, label: str) -> list[list[bool]]:
    if not isinstance(value, list):
        raise TypeError(f"{label}: expected list")
    return [
        _require_bool_list(item, f"{label}[{idx}]")
        for idx, item in enumerate(value)
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
    raise TypeError(
        f"{label}: expected numeric value, got {type(value).__name__}"
    )


def _require_nested_float_blocks(
    value: Any, label: str
) -> list[list[list[float]]]:
    if not isinstance(value, list):
        raise TypeError(f"{label}: expected list")
    nested_values: list[list[list[float]]] = []
    for residual_idx, residual_blocks in enumerate(value):
        if not isinstance(residual_blocks, list):
            raise TypeError(f"{label}[{residual_idx}]: expected list")
        parsed_blocks = []
        for block_idx, block_values in enumerate(residual_blocks):
            if not isinstance(block_values, list):
                raise TypeError(
                    f"{label}[{residual_idx}][{block_idx}]: expected list"
                )
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
            f"{metadata_path}: iteration {iteration}, "
            f"expected {expected_iteration}"
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
        raise TypeError(
            f"{metadata_path}: artifacts.{name} must be a JSON object"
        )
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
    artifact = _artifact_metadata(
        metadata_path, metadata, name, required=required
    )
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
        _require_key(
            artifact, "byte_order", f"{metadata_path}: artifacts.{name}"
        ),
        f"{metadata_path}: artifacts.{name}.byte_order",
    )
    if dtype != "float64":
        raise ValueError(
            f"{metadata_path}: artifacts.{name}.dtype must be float64"
        )
    if byte_order != "little_endian":
        raise ValueError(
            f"{metadata_path}: artifacts.{name}.byte_order must be "
            "little_endian"
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
                _require_key(
                    artifact, "ids", f"{metadata_path}: artifacts.{name}"
                ),
                f"{metadata_path}: artifacts.{name}.ids",
            )
        )
        if expected_ids is not None and ids != expected_ids:
            raise ValueError(
                f"{metadata_path}: artifacts.{name}.ids does not match "
                "top-level IDs"
            )
        if shape[0] != len(ids):
            raise ValueError(
                f"{metadata_path}: artifacts.{name}.shape has {shape[0]} "
                f"rows, expected {len(ids)} artifact IDs"
            )

    element_count = _shape_element_count(shape)
    path = metadata_path.parent / filename
    expected_size = element_count * 8
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"{path}: byte size {actual_size}, expected {expected_size}"
        )
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


class GlobalPositioningResidualBlock:
    def __init__(
        self, residual_values: GlobalPositioningResidualValues, index: int
    ):
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

    def jacobian(
        self, block: int | str, *, id: int | None = None
    ) -> np.ndarray:
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
            _require_key(
                self.metadata, "num_residual_blocks", str(self.metadata_path)
            ),
            "num_residual_blocks",
        )
        self.total_scalar_residuals = _require_int(
            _require_key(
                self.metadata, "total_scalar_residuals", str(self.metadata_path)
            ),
            "total_scalar_residuals",
        )
        self.residual_ids = _require_str_list(
            _require_key(
                self.metadata, "residual_ids", str(self.metadata_path)
            ),
            "residual_ids",
        )
        self.residual_dims = _require_int_list(
            _require_key(
                self.metadata, "residual_dims", str(self.metadata_path)
            ),
            "residual_dims",
        )
        self.residual_offsets = _require_int_list(
            _require_key(
                self.metadata, "residual_offsets", str(self.metadata_path)
            ),
            "residual_offsets",
        )
        self.evaluation_success = _require_bool_list(
            _require_key(
                self.metadata, "evaluation_success", str(self.metadata_path)
            ),
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
            _require_key(
                self.metadata, "has_raw_jacobians", str(self.metadata_path)
            ),
            "has_raw_jacobians",
        )
        self._residual_id_to_index = {
            residual_id: idx
            for idx, residual_id in enumerate(self.residual_ids)
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
            _require_key(
                self.metadata, "loss_rho_layout", str(self.metadata_path)
            ),
            "loss_rho_layout",
        )
        if loss_rho_layout != "residual_block_major/rho0_rho1_rho2":
            raise ValueError(
                "loss_rho_layout must be "
                "'residual_block_major/rho0_rho1_rho2'"
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
            _require_key(
                self.metadata, "jacobian_domain", str(self.metadata_path)
            ),
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
                _require_key(
                    self.metadata, field_name, str(self.metadata_path)
                ),
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
                raise ValueError(
                    f"residual_dims[{residual_idx}] must be non-negative"
                )
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
                raise TypeError(
                    f"parameter_blocks[{residual_idx}]: expected list"
                )
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
                        lower_bounds=tuple(
                            lower_bounds[residual_idx][block_idx]
                        ),
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
    return _require_top_level_ids_and_shape(
        metadata_path, metadata, ids_key, shape_key
    )


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
        artifacts = _require_key(
            self.metadata, "artifacts", str(self.metadata_path)
        )
        if not isinstance(artifacts, dict):
            raise TypeError(
                f"{self.metadata_path}: artifacts must be a JSON object"
            )
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


class GlobalPositioningTrace:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.manifest = _load_json(self.path / "manifest.json")
        self.residual_blocks = self._read_optional_jsonl(
            "residual_blocks.jsonl"
        )
        self.residual_skips = self._read_optional_jsonl("residual_skips.jsonl")
        self._residual_values_by_iteration = _discover_iteration_metadata(
            self.path / "residual_values"
        )
        self._snapshots_by_iteration = _discover_iteration_metadata(
            self.path / "snapshots"
        )

    @classmethod
    def load(cls, path: str | Path) -> GlobalPositioningTrace:
        return cls(Path(path))

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
                raise TypeError(
                    f"residual_blocks[{idx}].attrs must be an object"
                )
            residual_ids.append(
                _require_str(
                    _require_key(
                        attrs, "residual_id", f"residual_blocks[{idx}].attrs"
                    ),
                    f"residual_blocks[{idx}].attrs.residual_id",
                )
            )
        return tuple(residual_ids)

    def _manifest_run_id(self) -> str | None:
        if "run_id" not in self.manifest:
            return None
        return _require_str(
            self.manifest["run_id"], str(self.path / "manifest.json")
        )

    @property
    def status(self) -> str:
        return _require_str(
            _require_key(
                self.manifest, "status", str(self.path / "manifest.json")
            ),
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
            raise KeyError(
                f"Trace has no residual values for iteration {iteration}"
            )
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
            raise KeyError(
                f"Trace has no parameter snapshot for iteration {iteration}"
            )
        return GlobalPositioningParameterSnapshotLoader(
            self._snapshots_by_iteration[iteration],
            expected_iteration=iteration,
            expected_run_id=self._manifest_run_id(),
            max_points=max_points,
        ).load()
