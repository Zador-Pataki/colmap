# Copyright (c), ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

set(FOLDER_NAME "estimators_cost_functions")

COLMAP_ADD_LIBRARY(
    NAME colmap_estimators_cost_functions
    TYPE INTERFACE
    SRCS
        alignment.h
        calibration.h
        depth_prior.h
        manifold.h
        motion_averaging.h
        pose_prior.h
        reprojection_error.h
        sampson_error.h
        utils.h
        weighted_reprojection_error.h
        # §06 glomap-ported cost functors (autodiff-only, Eigen+Ceres-only).
        glomap_helpers.h
        bata_pairwise_direction_error.h
        bata_pairwise_direction_error_with_rotation.h
        direct_scale_regularization_error.h
        fetzer_focal_length_cost.h
        fetzer_focal_length_same_camera_cost.h
        focal_length_random_walk_error.h
        grav_error.h
        log_scale_prior_error.h
        mahalanobis_bata_directional_error.h
        mahalanobis_bata_directional_error_with_rotation.h
        metric_depth_error.h
        metric_depth_error_with_rotation.h
        point_camera_anisotropic_range_error.h
        point_camera_direct_scaled_range_error.h
        point_camera_range_error.h
        point_camera_scaled_metric_range_error.h
        point_camera_scaled_range_error.h
        relative_translation_error.h
        rotation_prior_error.h
        scale_prior_error.h
        scale_regularization_error.h
        weighted_bata_directional_error.h
        weighted_bata_directional_error_with_rotation.h
        weighted_point_camera_scaled_metric_range_error.h
    INTERFACE_LINK_LIBS
        colmap_geometry
        colmap_sensor
        Eigen3::Eigen
        Ceres::ceres
)

COLMAP_ADD_TEST(
    NAME alignment_test
    SRCS alignment_test.cc
    LINK_LIBS colmap_estimators_cost_functions
)
COLMAP_ADD_TEST(
    NAME calibration_test
    SRCS calibration_test.cc
    LINK_LIBS colmap_estimators_cost_functions
)
COLMAP_ADD_TEST(
    NAME utils_test
    SRCS utils_test.cc
    LINK_LIBS colmap_estimators_cost_functions
)
COLMAP_ADD_TEST(
    NAME reprojection_error_test
    SRCS reprojection_error_test.cc
    LINK_LIBS colmap_estimators_cost_functions
)
COLMAP_ADD_TEST(
    NAME sampson_error_test
    SRCS sampson_error_test.cc
    LINK_LIBS colmap_estimators_cost_functions
)
COLMAP_ADD_TEST(
    NAME pose_prior_test
    SRCS pose_prior_test.cc
    LINK_LIBS colmap_estimators_cost_functions
)
COLMAP_ADD_TEST(
    NAME motion_averaging_test
    SRCS motion_averaging_test.cc
    LINK_LIBS colmap_estimators_cost_functions
)
COLMAP_ADD_TEST(
    NAME depth_prior_test
    SRCS depth_prior_test.cc
    LINK_LIBS colmap_estimators_cost_functions
)

# §06 glomap-ported cost functor tests (one smoke test per functor).
COLMAP_ADD_TEST(NAME bata_pairwise_direction_error_test SRCS bata_pairwise_direction_error_test.cc LINK_LIBS colmap_estimators_cost_functions)
COLMAP_ADD_TEST(NAME bata_pairwise_direction_error_with_rotation_test SRCS bata_pairwise_direction_error_with_rotation_test.cc LINK_LIBS colmap_estimators_cost_functions)
COLMAP_ADD_TEST(NAME direct_scale_regularization_error_test SRCS direct_scale_regularization_error_test.cc LINK_LIBS colmap_estimators_cost_functions)
COLMAP_ADD_TEST(NAME fetzer_focal_length_cost_test SRCS fetzer_focal_length_cost_test.cc LINK_LIBS colmap_estimators_cost_functions)
COLMAP_ADD_TEST(NAME fetzer_focal_length_same_camera_cost_test SRCS fetzer_focal_length_same_camera_cost_test.cc LINK_LIBS colmap_estimators_cost_functions)
COLMAP_ADD_TEST(NAME focal_length_random_walk_error_test SRCS focal_length_random_walk_error_test.cc LINK_LIBS colmap_estimators_cost_functions)
COLMAP_ADD_TEST(NAME grav_error_test SRCS grav_error_test.cc LINK_LIBS colmap_estimators_cost_functions)
COLMAP_ADD_TEST(NAME log_scale_prior_error_test SRCS log_scale_prior_error_test.cc LINK_LIBS colmap_estimators_cost_functions)
COLMAP_ADD_TEST(NAME mahalanobis_bata_directional_error_test SRCS mahalanobis_bata_directional_error_test.cc LINK_LIBS colmap_estimators_cost_functions)
COLMAP_ADD_TEST(NAME mahalanobis_bata_directional_error_with_rotation_test SRCS mahalanobis_bata_directional_error_with_rotation_test.cc LINK_LIBS colmap_estimators_cost_functions)
COLMAP_ADD_TEST(NAME metric_depth_error_test SRCS metric_depth_error_test.cc LINK_LIBS colmap_estimators_cost_functions)
COLMAP_ADD_TEST(NAME metric_depth_error_with_rotation_test SRCS metric_depth_error_with_rotation_test.cc LINK_LIBS colmap_estimators_cost_functions)
COLMAP_ADD_TEST(NAME point_camera_anisotropic_range_error_test SRCS point_camera_anisotropic_range_error_test.cc LINK_LIBS colmap_estimators_cost_functions)
COLMAP_ADD_TEST(NAME point_camera_direct_scaled_range_error_test SRCS point_camera_direct_scaled_range_error_test.cc LINK_LIBS colmap_estimators_cost_functions)
COLMAP_ADD_TEST(NAME point_camera_range_error_test SRCS point_camera_range_error_test.cc LINK_LIBS colmap_estimators_cost_functions)
COLMAP_ADD_TEST(NAME point_camera_scaled_metric_range_error_test SRCS point_camera_scaled_metric_range_error_test.cc LINK_LIBS colmap_estimators_cost_functions)
COLMAP_ADD_TEST(NAME point_camera_scaled_range_error_test SRCS point_camera_scaled_range_error_test.cc LINK_LIBS colmap_estimators_cost_functions)
COLMAP_ADD_TEST(NAME relative_translation_error_test SRCS relative_translation_error_test.cc LINK_LIBS colmap_estimators_cost_functions)
COLMAP_ADD_TEST(NAME rotation_prior_error_test SRCS rotation_prior_error_test.cc LINK_LIBS colmap_estimators_cost_functions)
COLMAP_ADD_TEST(NAME scale_prior_error_test SRCS scale_prior_error_test.cc LINK_LIBS colmap_estimators_cost_functions)
COLMAP_ADD_TEST(NAME scale_regularization_error_test SRCS scale_regularization_error_test.cc LINK_LIBS colmap_estimators_cost_functions)
COLMAP_ADD_TEST(NAME weighted_bata_directional_error_test SRCS weighted_bata_directional_error_test.cc LINK_LIBS colmap_estimators_cost_functions)
COLMAP_ADD_TEST(NAME weighted_bata_directional_error_with_rotation_test SRCS weighted_bata_directional_error_with_rotation_test.cc LINK_LIBS colmap_estimators_cost_functions)
COLMAP_ADD_TEST(NAME weighted_point_camera_scaled_metric_range_error_test SRCS weighted_point_camera_scaled_metric_range_error_test.cc LINK_LIBS colmap_estimators_cost_functions)
