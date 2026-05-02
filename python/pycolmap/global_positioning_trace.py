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


def _artifact_path(
    metadata_path: Path,
    metadata: dict[str, Any],
    name: str,
    expected_count: int,
) -> Path:
    artifacts = _require_key(metadata, "artifacts", str(metadata_path))
    if not isinstance(artifacts, dict):
        raise TypeError(f"{metadata_path}: artifacts must be a JSON object")
    artifact = _require_key(artifacts, name, f"{metadata_path}: artifacts")
    if not isinstance(artifact, dict):
        raise TypeError(
            f"{metadata_path}: artifacts.{name} must be a JSON object"
        )

    filename = _require_str(
        _require_key(artifact, "file", f"{metadata_path}: artifacts.{name}"),
        "file",
    )
    dtype = _require_key(
        artifact, "dtype", f"{metadata_path}: artifacts.{name}"
    )
    byte_order = _require_key(
        artifact, "byte_order", f"{metadata_path}: artifacts.{name}"
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
    shape = _require_key(
        artifact, "shape", f"{metadata_path}: artifacts.{name}"
    )
    if not isinstance(shape, list) or len(shape) != 1:
        raise TypeError(
            f"{metadata_path}: artifacts.{name}.shape must be [count]"
        )
    shape_count = _require_int(
        shape[0], f"{metadata_path}: artifacts.{name}.shape[0]"
    )
    if shape_count != expected_count:
        raise ValueError(
            f"{metadata_path}: artifacts.{name}.shape[0] is {shape_count}, "
            f"expected {expected_count}"
        )

    path = metadata_path.parent / filename
    expected_size = expected_count * 8
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"{path}: byte size {actual_size}, expected {expected_size}"
        )
    return path


def _memmap_float64(path: Path, count: int) -> np.memmap:
    return np.memmap(path, dtype="<f8", mode="r", shape=(count,))


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
    def __init__(self, metadata_path: Path):
        self.metadata_path = Path(metadata_path)
        self.metadata = _load_json(self.metadata_path)
        self.iteration = _require_int(
            _require_key(self.metadata, "iteration", str(self.metadata_path)),
            "iteration",
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

        self._raw_residuals_path = _artifact_path(
            self.metadata_path,
            self.metadata,
            "raw_residuals",
            self.total_scalar_residuals,
        )
        self._raw_costs_path = _artifact_path(
            self.metadata_path,
            self.metadata,
            "raw_costs",
            self.num_residual_blocks,
        )
        self._robust_costs_path = _artifact_path(
            self.metadata_path,
            self.metadata,
            "robust_costs",
            self.num_residual_blocks,
        )

        self.total_jacobian_scalars = 0
        self.parameter_blocks: list[list[GlobalPositioningParameterBlock]] = []
        self.raw_jacobian_offsets: list[list[int]] = []
        self._raw_jacobians_path: Path | None = None
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
            self.parameter_blocks = self._parse_parameter_blocks(
                parameter_block_descriptors,
                parameter_block_sizes,
                parameter_block_is_constant,
                parameter_block_lower_bounds,
            )
            self._raw_jacobians_path = _artifact_path(
                self.metadata_path,
                self.metadata,
                "raw_jacobians",
                self.total_jacobian_scalars,
            )

        self._raw_residuals: np.memmap | None = None
        self._raw_costs: np.memmap | None = None
        self._robust_costs: np.memmap | None = None
        self._raw_jacobians: np.memmap | None = None

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
    def raw_residuals(self) -> np.memmap:
        if self._raw_residuals is None:
            self._raw_residuals = _memmap_float64(
                self._raw_residuals_path, self.total_scalar_residuals
            )
        return self._raw_residuals

    @property
    def raw_costs(self) -> np.memmap:
        if self._raw_costs is None:
            self._raw_costs = _memmap_float64(
                self._raw_costs_path, self.num_residual_blocks
            )
        return self._raw_costs

    @property
    def robust_costs(self) -> np.memmap:
        if self._robust_costs is None:
            self._robust_costs = _memmap_float64(
                self._robust_costs_path, self.num_residual_blocks
            )
        return self._robust_costs

    @property
    def raw_jacobians(self) -> np.memmap | None:
        if self._raw_jacobians_path is None:
            return None
        if self._raw_jacobians is None:
            self._raw_jacobians = _memmap_float64(
                self._raw_jacobians_path, self.total_jacobian_scalars
            )
        return self._raw_jacobians

    def residual(self, residual: int | str) -> GlobalPositioningResidualBlock:
        if isinstance(residual, str):
            residual = self._residual_id_to_index[residual]
        if residual < 0 or residual >= self.num_residual_blocks:
            raise IndexError(f"Residual index {residual} out of range")
        return GlobalPositioningResidualBlock(self, residual)


class GlobalPositioningTrace:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.manifest = _load_json(self.path / "manifest.json")
        self.residual_blocks = self._read_optional_jsonl(
            "residual_blocks.jsonl"
        )
        self.residual_skips = self._read_optional_jsonl("residual_skips.jsonl")
        self._residual_values_by_iteration: dict[int, Path] = {}
        residual_values_dir = self.path / "residual_values"
        if residual_values_dir.is_dir():
            for metadata_path in sorted(
                residual_values_dir.glob("iter_*.json")
            ):
                metadata = _load_json(metadata_path)
                iteration = _require_int(
                    _require_key(metadata, "iteration", str(metadata_path)),
                    "iteration",
                )
                self._residual_values_by_iteration[iteration] = metadata_path

    @classmethod
    def load(cls, path: str | Path) -> GlobalPositioningTrace:
        return cls(Path(path))

    def _read_optional_jsonl(self, filename: str) -> list[dict[str, Any]]:
        path = self.path / filename
        if not path.exists():
            return []
        return _iter_jsonl(path)

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
            self._residual_values_by_iteration[iteration]
        )
