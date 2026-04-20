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
