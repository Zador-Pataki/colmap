# Copyright (c), ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Parallel fork-style scene layer (namespace colmap::glomap). §04 lands the
# first real source files; §09-§11 (estimators) and §12+ (pipeline) build on
# top of the types defined here.

set(FOLDER_NAME "glomap")

COLMAP_ADD_LIBRARY(
    NAME colmap_glomap
    SRCS
        types.h
        gravity_info.h gravity_info.cc
        image.h image.cc
        image_pair.h image_pair.cc
        track.h track.cc
        view_graph.h view_graph.cc
        camera.h camera.cc
    PUBLIC_LINK_LIBS
        colmap_util
        colmap_geometry
        colmap_sensor
        colmap_scene
        colmap_estimators
        Eigen3::Eigen
)

COLMAP_ADD_TEST(
    NAME view_graph_test
    SRCS view_graph_test.cc
    LINK_LIBS colmap_glomap
)
COLMAP_ADD_TEST(
    NAME image_test
    SRCS image_test.cc
    LINK_LIBS colmap_glomap
)
COLMAP_ADD_TEST(
    NAME image_pair_test
    SRCS image_pair_test.cc
    LINK_LIBS colmap_glomap
)
COLMAP_ADD_TEST(
    NAME track_test
    SRCS track_test.cc
    LINK_LIBS colmap_glomap
)
COLMAP_ADD_TEST(
    NAME camera_test
    SRCS camera_test.cc
    LINK_LIBS colmap_glomap
)
COLMAP_ADD_TEST(
    NAME gravity_info_test
    SRCS gravity_info_test.cc
    LINK_LIBS colmap_glomap
)
