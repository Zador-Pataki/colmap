import pycolmap
import pytest


def test_cost_functions_submodule_exists():
    assert hasattr(pycolmap._core, "cost_functions")


def test_reproj_error_cost_exists():
    assert hasattr(pycolmap._core.cost_functions, "ReprojErrorCost")


def test_sampson_error_cost_exists():
    assert hasattr(pycolmap._core.cost_functions, "SampsonErrorCost")


def test_absolute_pose_prior_cost_exists():
    assert hasattr(pycolmap._core.cost_functions, "AbsolutePosePriorCost")


def test_absolute_pose_position_prior_cost_exists():
    assert hasattr(
        pycolmap._core.cost_functions, "AbsolutePosePositionPriorCost"
    )


def test_relative_pose_prior_cost_exists():
    assert hasattr(pycolmap._core.cost_functions, "RelativePosePriorCost")


def test_point3d_alignment_cost_exists():
    assert hasattr(pycolmap._core.cost_functions, "Point3DAlignmentCost")


# --- Fork extension cost factories ---



@pytest.mark.parametrize(
    "name",
    [
        # BATA variants
        "BATAPairwiseDirectionErrorCost",
        "BATAPairwiseDirectionErrorWithRotationCost",
        "WeightedBATADirectionalErrorCost",
        "WeightedBATADirectionalErrorWithRotationCost",
        "MahalanobisBATADirectionalErrorCost",
        "MahalanobisBATADirectionalErrorWithRotationCost",
        # Range errors
        "PointCameraRangeErrorCost",
        "PointCameraScaledRangeErrorCost",
        "PointCameraDirectScaledRangeErrorCost",
        "PointCameraScaledMetricRangeErrorCost",
        "WeightedPointCameraScaledMetricRangeErrorCost",
        "PointCameraAnisotropicRangeErrorCost",
        # Depth / metric
        "MetricDepthErrorCost",
        "MetricDepthErrorWithRotationCost",
        # Priors & regularization
        "RelativeTranslationErrorCost",
        "RotationPriorErrorCost",
        "ScalePriorErrorCost",
        "LogScalePriorErrorCost",
        "ScaleRegularizationErrorCost",
        "DirectScaleRegularizationErrorCost",
        "GravErrorCost",
        # Focal length
        "FetzerFocalLengthCost",
        "FetzerFocalLengthSameCameraCost",
        "FocalLengthRandomWalkErrorCost",
    ],
)
def test_fork_cost_factory_exists(name):
    assert hasattr(pycolmap._core.cost_functions, name)
    assert callable(getattr(pycolmap._core.cost_functions, name))
