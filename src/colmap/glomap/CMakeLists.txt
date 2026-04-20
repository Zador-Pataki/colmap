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
        # §09 math helpers (l1 solver header-only; rigid3d + tree have .cc)
        math/l1_solver.h
        math/rigid3d.h math/rigid3d.cc
        math/tree.h math/tree.cc
        # §11 additional helpers — union_find is header-only + self-contained.
        # two_view_geometry + colmap_converter also not wired here; they need
        # EPS constant adds and colmap4 std::optional API adaptations
        # respectively. See §11 note in estimators/glomap/CMakeLists.txt.
        math/union_find.h
        math/two_view_geometry.h math/two_view_geometry.cc
        # io/colmap_converter.h io/colmap_converter.cc   # §14 deferred (API divergences)
        # §14 processors (bound as module-level free functions in pycolmap).
        processors/relpose_filter.h processors/relpose_filter.cc
        processors/image_undistorter.h processors/image_undistorter.cc
        processors/image_pair_inliers.h processors/image_pair_inliers.cc
        processors/view_graph_manipulation.h processors/view_graph_manipulation.cc
        processors/reconstruction_normalizer.h processors/reconstruction_normalizer.cc
    PUBLIC_LINK_LIBS
        colmap_util
        colmap_geometry
        colmap_sensor
        colmap_scene
        colmap_estimators
        Eigen3::Eigen
        ${Boost_GRAPH_LIBRARY}
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
