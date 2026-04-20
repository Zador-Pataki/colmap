# Copyright (c), ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Fork-style estimator path: global_positioner, rotation_averaging (SolveCeres),
# iteration_callback, view_graph_calibration, relpose_estimation, track_*,
# bundle_adjustment, optimization_base. Populated by sections 09-11.

set(FOLDER_NAME "estimators/glomap")

COLMAP_ADD_LIBRARY(
    NAME colmap_estimators_glomap
    TYPE INTERFACE
    SRCS
)
