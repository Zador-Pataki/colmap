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
        # §11 non-RA/non-GP estimators (5/6 wired; retriangulation blocked).
        relpose_estimation.h relpose_estimation.cc
        view_graph_calibration.h view_graph_calibration.cc
        bundle_adjustment.h bundle_adjustment.cc
        track_filter.h track_filter.cc
        track_establishment.h track_establishment.cc
        # track_retriangulation needs colmap_converter.h which hits multiple
        # colmap4 API divergences (SetCamFromWorld, PosePrior::IsValid,
        # Database::PairIdToImagePair, std::optional<Rigid3d>) — wiring
        # deferred until those are reconciled.
        # track_retriangulation.h track_retriangulation.cc
    PUBLIC_LINK_LIBS
        colmap_glomap
        colmap_estimators_cost_functions
        PoseLib::PoseLib
        Ceres::ceres
        Eigen3::Eigen
)

COLMAP_ADD_TEST(
    NAME options_test
    SRCS options_test.cc
    LINK_LIBS colmap_estimators_glomap
)
