# Usage:
#   python generate_caspar.py <out_dir> [f32|f64]

import inspect
import sys
from itertools import combinations
from pathlib import Path
from typing import Annotated, get_type_hints

# Must be set before importing symforce.symbolic
precision = sys.argv[2] if len(sys.argv) > 2 else "f32"

if precision not in ("f32", "f64"):
    print(f"ERROR: Unknown precision '{precision}'. Expected f32 or f64.")
    sys.exit(1)

import symforce  # noqa: E402

symforce.set_epsilon_to_number(1e-15 if precision == "f64" else 1e-6)

import symforce.symbolic as sf  # noqa: E402
from symforce import typing as T  # noqa: E402
from symforce.caspar import CasparLibrary  # noqa: E402
from symforce.caspar.code_formulation import ftypes  # noqa: E402
from symforce.caspar import memory as mem  # noqa: E402
from symengine.lib import symengine_wrapper  # noqa: E402


class Log(ftypes.Func):
    def assign_code(self, outs: list[str], args: list[str], dtype) -> str:
        return f"{outs[0]} = {'log' if dtype.is_double() else 'logf'}({args[0]});"


ftypes.EXPR_TO_FUNC.setdefault(symengine_wrapper.log, Log)


# Point and ConstPoint are shared across camera models so that different
# camera types can observe the same 3D points in the same node pool.
class Point(sf.V3):
    pass


class ConstPoint(sf.V3):
    pass


class ConstPixel(sf.V2):
    pass


class ConstLogDepth(sf.V1):
    pass


class ConstInvStd4(sf.V4):
    pass


class ConstInvStd1(sf.V1):
    pass


class ConstRobustLoss(sf.V3):
    pass  # [loss_type, loss_scale, magnitude]


class ConstReprojectionWeightLoss(sf.V7):
    pass  # [sqrt_info00, sqrt_info01, sqrt_info10, sqrt_info11, type, scale, magnitude]


# Calibration node layout:
#
# When both focal_and_distortion/focal and principal_point are tunable, they
# are merged into a single V4 Calib node to save one shared-memory slot per
# block.
# This covers 4 variants: BASE, FIXED_POSE, FIXED_POINT, FIXED_POSE_FIXED_POINT.
#
# When at least one group is fixed, split V2 nodes are used (11 variants).
# Total: 4 merged + 11 split = 15 variants per camera model.
#
# Pose and Calib nodes are camera-model-specific to prevent cross-model
# batching.
# Check that generated kernels stay within the 48 KB shared memory limit if
# adding new parameter groups.


# SimpleRadial: params = [f, cx, cy, k]
class SimpleRadialPose(sf.Pose3):
    pass


class ConstSimpleRadialPose(sf.Pose3):
    pass


class SimpleRadialCalib(sf.V4):
    pass  # [f, k, cx, cy]  (merged)


class ConstSimpleRadialCalib(sf.V4):
    pass


class SimpleRadialPrincipalPoint(sf.V2):
    pass  # [cx, cy]  (split: pp tunable)


class ConstSimpleRadialPrincipalPoint(sf.V2):
    pass


class SimpleRadialFocalAndDistortion(sf.V2):
    pass  # [f, k]    (split: focal tunable)


class ConstSimpleRadialFocalAndDistortion(sf.V2):
    pass


# Pinhole: params = [fx, fy, cx, cy]
class PinholePose(sf.Pose3):
    pass


class ConstPinholePose(sf.Pose3):
    pass


class ConstPinholeRotation(sf.Rot3):
    pass


class PinholeTranslation(sf.V3):
    pass


class PinholeCalib(sf.V4):
    pass  # [fx, fy, cx, cy]  (merged)


class ConstPinholeCalib(sf.V4):
    pass


class PinholePrincipalPoint(sf.V2):
    pass  # [cx, cy]  (split: pp tunable)


class ConstPinholePrincipalPoint(sf.V2):
    pass


class PinholeFocal(sf.V2):
    pass  # [fx, fy]  (split: focal tunable)


class ConstPinholeFocal(sf.V2):
    pass


class DepthScale(sf.V1):
    pass


class ConstDepthScale(sf.V1):
    pass


def _make_variant(
    core_fn, name: str, base_params: list, hints: dict, fixed: dict
):
    new_hints = {}
    for p in base_params:
        if p in fixed:
            new_hints[p] = Annotated[fixed[p], mem.ConstantSequential]
        else:
            new_hints[p] = hints[p]

    tunable_params = [p for p in base_params if p not in fixed]
    const_params = [p for p in base_params if p in fixed]
    ordered = tunable_params + const_params

    # Caspar calls factors as fn(**symbolic_args), so the wrapper accepts both
    # positional and keyword arguments.
    def wrapper(*args, **kwargs):
        merged = {**dict(zip(ordered, args, strict=False)), **kwargs}
        return core_fn(*[merged[p] for p in base_params])

    wrapper.__name__ = name
    wrapper.__annotations__ = {p: new_hints[p] for p in ordered}
    wrapper.__annotations__["return"] = hints.get("return")
    wrapper.__signature__ = inspect.Signature(
        [
            inspect.Parameter(p, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for p in ordered
        ]
    )
    return wrapper


def register_camera_model(
    caslib,
    model_name: str,
    core_fn,
    fixable_params: dict,
    must_fix_one_of: T.Optional[set] = None,
    include_all_fixed: bool = False,
):
    hints = get_type_hints(core_fn, include_extras=True)
    base_params = list(inspect.signature(core_fn).parameters.keys())
    fixable_items = list(fixable_params.items())

    # include_all_fixed extends the range to N for merged-calib models, where
    # the "all-fixed" subset still has a tunable Calib node.
    max_r = len(fixable_items) + (1 if include_all_fixed else 0)
    for r in range(max_r):
        for combo in combinations(fixable_items, r):
            fixed = dict(combo)

            # Skip split variants where both calib groups are tunable, as those
            # are handled by the merged-calib registration.
            if must_fix_one_of and not any(p in fixed for p in must_fix_one_of):
                continue

            if fixed:
                # Suffix order follows fixable_params definition order to
                # match the C++ dispatch naming.
                suffix = "_".join(
                    f"fixed_{p}" for p, _ in fixable_items if p in fixed
                )
                name = f"{model_name}_{suffix}"
            else:
                name = model_name

            caslib.add_factor(
                _make_variant(core_fn, name, base_params, hints, fixed)
            )


def robustify(residual, loss: ConstRobustLoss):
    """Return an equivalent residual vector for the supported Ceres losses."""
    loss_type = loss[0]
    loss_scale = sf.Max(loss[1], sf.epsilon())
    magnitude = sf.Max(loss[2], 0)
    s = residual.squared_norm()
    safe_s = sf.Max(s, sf.epsilon())
    a2 = loss_scale * loss_scale
    rho_trivial = s
    rho_soft_l1 = 2 * a2 * (sf.sqrt(1 + s / a2) - 1)
    rho_cauchy = a2 * sf.log(1 + s / a2)
    rho_huber = sf.Piecewise(
        (s, s <= a2), (2 * loss_scale * sf.sqrt(safe_s) - a2, True)
    )
    rho_nontrivial = sf.Piecewise(
        (rho_soft_l1, loss_type < 1.5),
        (
            sf.Piecewise((rho_cauchy, loss_type < 2.5), (rho_huber, True)),
            True,
        ),
    )
    rho = sf.Piecewise((rho_trivial, loss_type < 0.5), (rho_nontrivial, True))
    scale = sf.Piecewise(
        (sf.sqrt(magnitude), s <= sf.epsilon()),
        (sf.sqrt(magnitude * sf.Max(rho, 0) / safe_s), True),
    )
    return residual * scale


def reprojection_weight_and_robustify(residual: sf.V2, weight_loss: ConstReprojectionWeightLoss):
    weighted = sf.V2(
        [
            weight_loss[0] * residual[0] + weight_loss[1] * residual[1],
            weight_loss[2] * residual[0] + weight_loss[3] * residual[1],
        ]
    )
    return robustify(
        weighted, ConstRobustLoss([weight_loss[4], weight_loss[5], weight_loss[6]])
    )


# --- Camera models ---

# Merged cores define the canonical projection math reused by split variants.


def simple_radial_core(
    pose: T.Annotated[SimpleRadialPose, mem.TunableShared],
    calib: T.Annotated[SimpleRadialCalib, mem.TunableShared],  # [f, k, cx, cy]
    point: T.Annotated[Point, mem.TunableShared],
    pixel: T.Annotated[ConstPixel, mem.ConstantSequential],
) -> sf.V2:
    """Reprojection residual for COLMAP's SIMPLE_RADIAL model (sensor/models.h).

    calib = [f, k, cx, cy]: single focal length, one radial distortion
    coefficient, and principal point.
    """
    cam_T_world = pose
    f, k, cx, cy = calib
    point_cam = cam_T_world * point
    depth = point_cam[2]
    p = sf.V2(point_cam[:2]) / (depth + sf.epsilon() * sf.sign_no_zero(depth))
    r = 1 + k * p.squared_norm()
    return f * r * p + sf.V2([cx, cy]) - pixel


def pinhole_core(
    pose: T.Annotated[PinholePose, mem.TunableShared],
    calib: T.Annotated[PinholeCalib, mem.TunableShared],  # [fx, fy, cx, cy]
    point: T.Annotated[Point, mem.TunableShared],
    pixel: T.Annotated[ConstPixel, mem.ConstantSequential],
    weight_loss: T.Annotated[ConstReprojectionWeightLoss, mem.ConstantSequential],
) -> sf.V2:
    """Reprojection residual for COLMAP's PINHOLE model (sensor/models.h).

    calib = [fx, fy, cx, cy]: two independent focal lengths and principal point,
    no distortion.
    """
    cam_T_world = pose
    fx, fy, cx, cy = calib
    point_cam = cam_T_world * point
    depth = point_cam[2]
    p = sf.V2(point_cam[:2]) / (depth + sf.epsilon() * sf.sign_no_zero(depth))
    residual = sf.V2([fx * p[0] + cx, fy * p[1] + cy]) - pixel
    return reprojection_weight_and_robustify(residual, weight_loss)


def pinhole_fixed_rotation_core(
    rotation: T.Annotated[ConstPinholeRotation, mem.ConstantSequential],
    translation: T.Annotated[PinholeTranslation, mem.TunableShared],
    calib: T.Annotated[PinholeCalib, mem.TunableShared],  # [fx, fy, cx, cy]
    point: T.Annotated[Point, mem.TunableShared],
    pixel: T.Annotated[ConstPixel, mem.ConstantSequential],
    weight_loss: T.Annotated[ConstReprojectionWeightLoss, mem.ConstantSequential],
) -> sf.V2:
    """PINHOLE reprojection residual with fixed rotation and tunable translation."""
    fx, fy, cx, cy = calib
    point_cam = rotation * point + translation
    depth = point_cam[2]
    p = sf.V2(point_cam[:2]) / (depth + sf.epsilon() * sf.sign_no_zero(depth))
    residual = sf.V2([fx * p[0] + cx, fy * p[1] + cy]) - pixel
    return reprojection_weight_and_robustify(residual, weight_loss)


def pinhole_log_depth_core(
    pose: T.Annotated[PinholePose, mem.TunableShared],
    scale: T.Annotated[DepthScale, mem.TunableShared],
    point: T.Annotated[Point, mem.TunableShared],
    log_depth: T.Annotated[ConstLogDepth, mem.ConstantSequential],
    loss: T.Annotated[ConstRobustLoss, mem.ConstantSequential],
) -> sf.V1:
    """MPSFM log-depth residual.

    residual = 0                                        if predicted depth <= 0
             = log((R * point + t).z) - log_depth - scale otherwise
    """
    point_cam = pose * point
    depth = point_cam[2]
    safe_depth = sf.Max(depth, sf.epsilon())
    residual = sf.log(safe_depth) - log_depth[0] - scale[0]
    raw = sf.V1([sf.Piecewise((residual, depth > 0), (0, True))])
    return robustify(raw, loss)


def pinhole_log_depth_fixed_rotation_core(
    rotation: T.Annotated[ConstPinholeRotation, mem.ConstantSequential],
    translation: T.Annotated[PinholeTranslation, mem.TunableShared],
    scale: T.Annotated[DepthScale, mem.TunableShared],
    point: T.Annotated[Point, mem.TunableShared],
    log_depth: T.Annotated[ConstLogDepth, mem.ConstantSequential],
    loss: T.Annotated[ConstRobustLoss, mem.ConstantSequential],
) -> sf.V1:
    """MPSFM log-depth residual with fixed rotation and tunable translation."""
    point_cam = rotation * point + translation
    depth = point_cam[2]
    safe_depth = sf.Max(depth, sf.epsilon())
    residual = sf.log(safe_depth) - log_depth[0] - scale[0]
    raw = sf.V1([sf.Piecewise((residual, depth > 0), (0, True))])
    return robustify(raw, loss)


def pinhole_intrinsics_prior(
    calib: T.Annotated[PinholeCalib, mem.TunableShared],
    prior: T.Annotated[ConstPinholeCalib, mem.ConstantSequential],
    inv_std: T.Annotated[ConstInvStd4, mem.ConstantSequential],
) -> sf.V4:
    """Diagonal Gaussian prior on PINHOLE [fx, fy, cx, cy]."""
    return sf.V4([(calib[i] - prior[i]) * inv_std[i] for i in range(4)])


def pinhole_split_intrinsics_prior(
    focal: T.Annotated[PinholeFocal, mem.TunableShared],
    principal_point: T.Annotated[PinholePrincipalPoint, mem.TunableShared],
    prior: T.Annotated[ConstPinholeCalib, mem.ConstantSequential],
    inv_std: T.Annotated[ConstInvStd4, mem.ConstantSequential],
) -> sf.V4:
    """Split-node PINHOLE intrinsics prior on [fx, fy, cx, cy]."""
    calib = sf.V4([focal[0], focal[1], principal_point[0], principal_point[1]])
    return sf.V4([(calib[i] - prior[i]) * inv_std[i] for i in range(4)])


def pinhole_intrinsics_random_walk(
    prev_calib: T.Annotated[PinholeCalib, mem.TunableShared],
    next_calib: T.Annotated[PinholeCalib, mem.TunableShared],
    inv_std: T.Annotated[ConstInvStd4, mem.ConstantSequential],
) -> sf.V4:
    """Random-walk residual between adjacent PINHOLE [fx, fy, cx, cy] nodes."""
    return sf.V4([(next_calib[i] - prev_calib[i]) * inv_std[i] for i in range(4)])


def pinhole_split_intrinsics_random_walk(
    prev_focal: T.Annotated[PinholeFocal, mem.TunableShared],
    prev_principal_point: T.Annotated[PinholePrincipalPoint, mem.TunableShared],
    next_focal: T.Annotated[PinholeFocal, mem.TunableShared],
    next_principal_point: T.Annotated[PinholePrincipalPoint, mem.TunableShared],
    inv_std: T.Annotated[ConstInvStd4, mem.ConstantSequential],
) -> sf.V4:
    """Split-node random-walk residual between adjacent PINHOLE intrinsics."""
    prev_calib = sf.V4(
        [
            prev_focal[0],
            prev_focal[1],
            prev_principal_point[0],
            prev_principal_point[1],
        ]
    )
    next_calib = sf.V4(
        [
            next_focal[0],
            next_focal[1],
            next_principal_point[0],
            next_principal_point[1],
        ]
    )
    return sf.V4([(next_calib[i] - prev_calib[i]) * inv_std[i] for i in range(4)])


def scale_prior(
    scale: T.Annotated[DepthScale, mem.TunableShared],
    inv_std: T.Annotated[ConstInvStd1, mem.ConstantSequential],
    loss: T.Annotated[ConstRobustLoss, mem.ConstantSequential],
) -> sf.V1:
    """Gaussian prior on MPSFM log-depth scale."""
    return robustify(sf.V1([scale[0] * inv_std[0]]), loss)


# Split cores delegate to merged cores to avoid duplicating projection math.


def simple_radial_split_core(
    pose: T.Annotated[SimpleRadialPose, mem.TunableShared],
    focal_and_distortion: T.Annotated[
        SimpleRadialFocalAndDistortion, mem.TunableShared
    ],
    principal_point: T.Annotated[SimpleRadialPrincipalPoint, mem.TunableShared],
    point: T.Annotated[Point, mem.TunableShared],
    pixel: T.Annotated[ConstPixel, mem.ConstantSequential],
) -> sf.V2:
    """Split-calib variant of simple_radial_core.

    For COLMAP's SIMPLE_RADIAL model. Used when focal/distortion and principal
    point are tuned independently.
    focal_and_distortion = [f, k], principal_point = [cx, cy].
    """
    calib = sf.V4(
        [
            focal_and_distortion[0],
            focal_and_distortion[1],
            principal_point[0],
            principal_point[1],
        ]
    )
    return simple_radial_core(pose, calib, point, pixel)


def pinhole_split_core(
    pose: T.Annotated[PinholePose, mem.TunableShared],
    focal: T.Annotated[PinholeFocal, mem.TunableShared],
    principal_point: T.Annotated[PinholePrincipalPoint, mem.TunableShared],
    point: T.Annotated[Point, mem.TunableShared],
    pixel: T.Annotated[ConstPixel, mem.ConstantSequential],
    weight_loss: T.Annotated[ConstReprojectionWeightLoss, mem.ConstantSequential],
) -> sf.V2:
    """Split-calib variant of pinhole_core for COLMAP's PINHOLE model.

    Used when focal lengths and principal point are tuned independently.
    focal = [fx, fy], principal_point = [cx, cy].
    """
    calib = sf.V4([focal[0], focal[1], principal_point[0], principal_point[1]])
    return pinhole_core(pose, calib, point, pixel, weight_loss)


def pinhole_split_fixed_rotation_core(
    rotation: T.Annotated[ConstPinholeRotation, mem.ConstantSequential],
    translation: T.Annotated[PinholeTranslation, mem.TunableShared],
    focal: T.Annotated[PinholeFocal, mem.TunableShared],
    principal_point: T.Annotated[PinholePrincipalPoint, mem.TunableShared],
    point: T.Annotated[Point, mem.TunableShared],
    pixel: T.Annotated[ConstPixel, mem.ConstantSequential],
    weight_loss: T.Annotated[ConstReprojectionWeightLoss, mem.ConstantSequential],
) -> sf.V2:
    """Split-calib PINHOLE reprojection with fixed rotation and tunable translation."""
    calib = sf.V4([focal[0], focal[1], principal_point[0], principal_point[1]])
    return pinhole_fixed_rotation_core(
        rotation, translation, calib, point, pixel, weight_loss
    )


dtype = mem.DType.DOUBLE if precision == "f64" else mem.DType.FLOAT
caslib = CasparLibrary(name="caspar_lib", dtype=dtype)


# Suffix order defines generated variant names and must match the C++ dispatch
# logic that builds names from BundleAdjustmentOptions.
#
# COLMAP flag mapping:
#   refine_rig_from_world                      -> pose
#   refine_focal_length && refine_extra_params -> focal_and_distortion / focal
#   refine_principal_point                     -> principal_point
#   refine_points3D                            -> point
#
# Limitations:
#   - constant_rig_from_world_rotation is supported for PINHOLE through
#     dedicated fixed-rotation factors with a tunable translation node.
#   - refine_sensor_from_rig not supported (for now) due to high shared memory
#     usage (single camera per rig assumed)
#   - refine_focal_length != refine_extra_params not supported (observations
#     skipped with a warning because the merged focal_and_distortion node
#     cannot be split)

FIXABLE_SIMPLE_RADIAL = {
    "pose": ConstSimpleRadialPose,
    "point": ConstPoint,
}

FIXABLE_PINHOLE = {
    "pose": ConstPinholePose,
    "point": ConstPoint,
}

FIXABLE_PINHOLE_LOG_DEPTH = {
    "pose": ConstPinholePose,
    "scale": ConstDepthScale,
    "point": ConstPoint,
}

FIXABLE_PINHOLE_FIXED_ROTATION = {
    "calib": ConstPinholeCalib,
    "point": ConstPoint,
}

FIXABLE_PINHOLE_LOG_DEPTH_FIXED_ROTATION = {
    "scale": ConstDepthScale,
    "point": ConstPoint,
}

FIXABLE_SIMPLE_RADIAL_SPLIT = {
    "pose": ConstSimpleRadialPose,
    "focal_and_distortion": ConstSimpleRadialFocalAndDistortion,
    "principal_point": ConstSimpleRadialPrincipalPoint,
    "point": ConstPoint,
}

FIXABLE_PINHOLE_SPLIT = {
    "pose": ConstPinholePose,
    "focal": ConstPinholeFocal,
    "principal_point": ConstPinholePrincipalPoint,
    "point": ConstPoint,
}

FIXABLE_PINHOLE_SPLIT_FIXED_ROTATION = {
    "focal": ConstPinholeFocal,
    "principal_point": ConstPinholePrincipalPoint,
    "point": ConstPoint,
}

FIXABLE_PINHOLE_SPLIT_INTRINSICS_PRIOR = {
    "focal": ConstPinholeFocal,
    "principal_point": ConstPinholePrincipalPoint,
}

FIXABLE_PINHOLE_SPLIT_INTRINSICS_RANDOM_WALK = {
    "prev_focal": ConstPinholeFocal,
    "prev_principal_point": ConstPinholePrincipalPoint,
    "next_focal": ConstPinholeFocal,
    "next_principal_point": ConstPinholePrincipalPoint,
}

# Merged: BASE, FIXED_POSE, FIXED_POINT, FIXED_POSE_FIXED_POINT (4 variants).
register_camera_model(
    caslib,
    "simple_radial",
    simple_radial_core,
    FIXABLE_SIMPLE_RADIAL,
    include_all_fixed=True,
)
register_camera_model(
    caslib, "pinhole", pinhole_core, FIXABLE_PINHOLE, include_all_fixed=True
)
register_camera_model(
    caslib,
    "pinhole_log_depth",
    pinhole_log_depth_core,
    FIXABLE_PINHOLE_LOG_DEPTH,
)
register_camera_model(
    caslib,
    "pinhole_fixed_rotation",
    pinhole_fixed_rotation_core,
    FIXABLE_PINHOLE_FIXED_ROTATION,
    include_all_fixed=True,
)
register_camera_model(
    caslib,
    "pinhole_log_depth_fixed_rotation",
    pinhole_log_depth_fixed_rotation_core,
    FIXABLE_PINHOLE_LOG_DEPTH_FIXED_ROTATION,
    include_all_fixed=True,
)
caslib.add_factor(pinhole_intrinsics_prior)
caslib.add_factor(pinhole_intrinsics_random_walk)
caslib.add_factor(scale_prior)

# Split: all variants where at least one of
# {focal_and_distortion, principal_point} is fixed (11 variants per model).
register_camera_model(
    caslib,
    "simple_radial_split",
    simple_radial_split_core,
    FIXABLE_SIMPLE_RADIAL_SPLIT,
    must_fix_one_of={"focal_and_distortion", "principal_point"},
)
register_camera_model(
    caslib,
    "pinhole_split",
    pinhole_split_core,
    FIXABLE_PINHOLE_SPLIT,
    must_fix_one_of={"focal", "principal_point"},
)
register_camera_model(
    caslib,
    "pinhole_split_fixed_rotation",
    pinhole_split_fixed_rotation_core,
    FIXABLE_PINHOLE_SPLIT_FIXED_ROTATION,
    must_fix_one_of={"focal", "principal_point"},
    include_all_fixed=True,
)
register_camera_model(
    caslib,
    "pinhole_split_intrinsics_prior",
    pinhole_split_intrinsics_prior,
    FIXABLE_PINHOLE_SPLIT_INTRINSICS_PRIOR,
    must_fix_one_of={"focal", "principal_point"},
)
register_camera_model(
    caslib,
    "pinhole_split_intrinsics_random_walk",
    pinhole_split_intrinsics_random_walk,
    FIXABLE_PINHOLE_SPLIT_INTRINSICS_RANDOM_WALK,
    must_fix_one_of={
        "prev_focal",
        "prev_principal_point",
        "next_focal",
        "next_principal_point",
    },
)

out_dir = Path(f"{sys.argv[1]}")
print(f"Generating Caspar kernels with precision {precision}: {out_dir}")
caslib.generate(out_dir, use_symlinks=False, python_bindings=False)
