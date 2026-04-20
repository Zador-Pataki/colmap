# Copyright (c), ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Fork-style estimator path: global_positioner, rotation_averaging (SolveCeres),
# iteration_callback, view_graph_calibration, relpose_estimation, track_*,
# bundle_adjustment, optimization_base. §07 lands the options structs;
# implementations follow in §09-§11.

set(FOLDER_NAME "estimators_glomap")

COLMAP_ADD_LIBRARY(
    NAME colmap_estimators_glomap
    SRCS
        # §07 options headers.
        bundle_adjustment_options.h
        global_mapper_options.h
        global_positioner_options.h
        inlier_threshold_options.h
        optimization_base_options.h
        rotation_estimator_options.h
        track_establishment_options.h
        triangulator_options.h
        view_graph_calibrator_options.h
        # §09 rotation averaging + iteration callback.
        iteration_callback.h
        rotation_averaging.h rotation_averaging.cc
        # §10 global positioner.
        global_positioner.h global_positioner.cc
        # §11 non-RA/non-GP estimators — files present in tree but not built
        # yet. Each requires porting additional fork helpers (union_find,
        # two_view_geometry, colmap_converter, ColmapCameraToPoseLibCamera) and
        # reconciling BundleAdjusterOptions→BundleAdjustmentOptions rename +
        # colmap4 API differences (DatabaseCache::Create signature,
        # BundleAdjustmentOptions refine_extra_params). Wired into build at §12
        # when run_* wrappers depend on them.
        # relpose_estimation.h relpose_estimation.cc
        # view_graph_calibration.h view_graph_calibration.cc
        # bundle_adjustment.h bundle_adjustment.cc
        # track_filter.h track_filter.cc
        # track_establishment.h track_establishment.cc
        # track_retriangulation.h track_retriangulation.cc
    PUBLIC_LINK_LIBS
        colmap_glomap
        colmap_estimators_cost_functions
        Ceres::ceres
        Eigen3::Eigen
)

COLMAP_ADD_TEST(
    NAME options_test
    SRCS options_test.cc
    LINK_LIBS colmap_estimators_glomap
)
# §11 non-RA/non-GP estimators (stub — may need per-file surgery to build)
# TODO: several fork references (BundleAdjusterOptions vs BundleAdjustmentOptions
# rename, RelativePoseEstimationOptions poselib dep, colmap_converter dep)
# need reconciliation with §07 options. Commented out until build proven.
#        relpose_estimation.h relpose_estimation.cc
#        view_graph_calibration.h view_graph_calibration.cc
#        bundle_adjustment.h bundle_adjustment.cc
#        track_filter.h track_filter.cc
#        track_establishment.h track_establishment.cc
#        track_retriangulation.h track_retriangulation.cc
