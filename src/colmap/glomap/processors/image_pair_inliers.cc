#include "colmap/glomap/processors/image_pair_inliers.h"

#include "colmap/geometry/rigid3.h"
#include "colmap/glomap/math/two_view_geometry.h"
#include "colmap/scene/two_view_geometry.h"

namespace colmap::glomap {

namespace {
constexpr double EPS = 1e-12;
}

struct InlierStats {
  int num_essential_pairs = 0;
  long long total_matches_essential = 0;
  long long failed_epipolar_essential = 0;
  long long failed_cheirality_essential = 0;
  long long failed_angle_check_essential = 0;
  long long failed_epipole_check_essential = 0;
  long long total_inliers_essential = 0;
  int num_fundamental_pairs = 0;
  long long total_matches_fundamental = 0;
  long long failed_epipolar_fundamental = 0;
  long long failed_cheirality_fundamental = 0;
  long long total_inliers_fundamental = 0;
  int num_homography_pairs = 0;
  long long total_matches_homography = 0;
  long long failed_epipolar_homography = 0;
  long long total_inliers_homography = 0;
};

double ImagePairInliers::ScoreError() {
  if (image_pair.config == colmap::TwoViewGeometry::PLANAR ||
      image_pair.config == colmap::TwoViewGeometry::PANORAMIC ||
      image_pair.config == colmap::TwoViewGeometry::PLANAR_OR_PANORAMIC) {
    stats.type = GeometryConfig::HOMOGRAPHY;
    return ScoreErrorHomography();
  } else if (image_pair.config == colmap::TwoViewGeometry::UNCALIBRATED) {
    stats.type = GeometryConfig::FUNDAMENTAL;
    return ScoreErrorFundamental();
  } else if (image_pair.config == colmap::TwoViewGeometry::CALIBRATED) {
    stats.type = GeometryConfig::ESSENTIAL;
    return ScoreErrorEssential();
  }
  return 0;
}

double ImagePairInliers::ScoreErrorEssential() {
  const colmap::Rigid3d& cam2_from_cam1 = image_pair.cam2_from_cam1;
  Eigen::Matrix3d E;
  EssentialFromMotion(cam2_from_cam1, &E);

  Eigen::Vector3d epipole12 = cam2_from_cam1.translation();
  Eigen::Vector3d epipole21 = colmap::Inverse(cam2_from_cam1).translation();

  if (epipole12.norm() > EPS) epipole12.normalize();
  if (epipole21.norm() > EPS) epipole21.normalize();

  if (epipole12[2] < 0) epipole12 = -epipole12;
  if (epipole21[2] < 0) epipole21 = -epipole21;

  if (image_pair.inliers.size() > 0) image_pair.inliers.clear();

  image_t image_id1 = image_pair.image_id1;
  image_t image_id2 = image_pair.image_id2;

  double thres = options.max_epipolar_error_E * 0.5 *
                 (1. / cameras->at(images.at(image_id1).camera_id).Focal() +
                  1. / cameras->at(images.at(image_id2).camera_id).Focal());
  double sq_threshold = thres * thres;
  double score = 0.;
  Eigen::Vector3d pt1, pt2;
  bool d_valid1;
  bool d_valid2;

  double thres_epipole = std::cos(DegToRad(options.min_angle_from_epipole));
  double thres_epipole_nodepth = std::cos(DegToRad(3.));
  double thres_angle = 1 + 1e-6;
  thres_epipole += 1e-6;

  stats.total_matches = image_pair.matches.rows();

  for (size_t k = 0; k < stats.total_matches; ++k) {
    pt1 = images.at(image_id1).features_undist[image_pair.matches(k, 0)];
    pt2 = images.at(image_id2).features_undist[image_pair.matches(k, 1)];
    const auto& img1 = images.at(image_id1);
    const auto& img2 = images.at(image_id2);
    d_valid1 = image_pair.matches(k, 0) < img1.depth_prior_validity.size()
                   ? img1.depth_prior_validity[image_pair.matches(k, 0)]
                   : false;
    d_valid2 = image_pair.matches(k, 1) < img2.depth_prior_validity.size()
                   ? img2.depth_prior_validity[image_pair.matches(k, 1)]
                   : false;
    const double r2 = SampsonError(E, pt1, pt2);

    if (r2 < sq_threshold) {
      bool cheirality = CheckCheirality(cam2_from_cam1, pt1, pt2, 1e-2, 100.);
      double diff_angle =
          pt1.dot(cam2_from_cam1.rotation().inverse() * pt2);
      bool angle_check = (diff_angle < thres_angle);
      double diff_epipole1 = pt1.dot(epipole21);
      double diff_epipole2 = pt2.dot(epipole12);
      bool epipole_check;
      if (d_valid1 && d_valid2) {
        epipole_check =
            (diff_epipole1 < thres_epipole && diff_epipole2 < thres_epipole);
      } else {
        epipole_check = (diff_epipole1 < thres_epipole_nodepth &&
                         diff_epipole2 < thres_epipole_nodepth);
      }
      if (cheirality && angle_check && epipole_check) {
        score += r2;
        image_pair.inliers.push_back(k);
      } else {
        score += sq_threshold;
        if (!cheirality) {
          stats.failed_cheirality_E++;
        } else if (!angle_check) {
          stats.failed_angle_check_E++;
        } else if (!epipole_check) {
          stats.failed_epipole_check_E++;
        }
      }
    } else {
      score += sq_threshold;
      stats.failed_epipolar_E++;
    }
  }

  stats.num_inliers = image_pair.inliers.size();
  return score;
}

double ImagePairInliers::ScoreErrorFundamental() {
  if (image_pair.inliers.size() > 0) image_pair.inliers.clear();

  Eigen::Vector3d epipole = image_pair.F.row(0).cross(image_pair.F.row(2));
  bool status = false;
  for (auto i = 0; i < 3; i++) {
    if ((epipole(i) > EPS) || (epipole(i) < -EPS)) {
      status = true;
      break;
    }
  }
  if (!status) epipole = image_pair.F.row(1).cross(image_pair.F.row(2));

  std::vector<double> signums;
  int positive_count = 0;
  int negative_count = 0;

  image_t image_id1 = image_pair.image_id1;
  image_t image_id2 = image_pair.image_id2;

  double thres = options.max_epipolar_error_F;
  double sq_threshold = thres * thres;

  double score = 0.;
  Eigen::Vector2d pt1, pt2;

  stats.total_matches = image_pair.matches.rows();
  std::vector<int> inliers_pre;
  std::vector<double> errors;
  for (size_t k = 0; k < stats.total_matches; ++k) {
    pt1 = images.at(image_id1).features[image_pair.matches(k, 0)];
    pt2 = images.at(image_id2).features[image_pair.matches(k, 1)];
    const double r2 = SampsonError(image_pair.F, pt1, pt2);

    if (r2 < sq_threshold) {
      signums.push_back(GetOrientationSignum(image_pair.F, epipole, pt1, pt2));
      if (signums.back() > 0) {
        positive_count++;
      } else {
        negative_count++;
      }
      inliers_pre.push_back(k);
      errors.push_back(r2);
    } else {
      score += sq_threshold;
      stats.failed_epipolar_F++;
    }
  }
  bool is_positive = (positive_count > negative_count);
  if (positive_count == negative_count) return 0;

  for (size_t k = 0; k < inliers_pre.size(); k++) {
    bool cheirality = (signums[k] > 0) == is_positive;
    if (!cheirality) {
      score += sq_threshold;
      stats.failed_cheirality_F++;
    } else {
      image_pair.inliers.push_back(inliers_pre[k]);
      score += errors[k];
    }
  }

  stats.num_inliers = image_pair.inliers.size();
  return score;
}

double ImagePairInliers::ScoreErrorHomography() {
  if (image_pair.inliers.size() > 0) image_pair.inliers.clear();

  image_t image_id1 = image_pair.image_id1;
  image_t image_id2 = image_pair.image_id2;

  double thres = options.max_epipolar_error_H;
  double sq_threshold = thres * thres;
  double score = 0.;
  Eigen::Vector2d pt1, pt2;
  stats.total_matches = image_pair.matches.rows();
  for (size_t k = 0; k < stats.total_matches; ++k) {
    pt1 = images.at(image_id1).features[image_pair.matches(k, 0)];
    pt2 = images.at(image_id2).features[image_pair.matches(k, 1)];
    const double r2 = HomographyError(image_pair.H, pt1, pt2);

    if (r2 < sq_threshold) {
      score += r2;
      image_pair.inliers.push_back(k);
    } else {
      score += sq_threshold;
      stats.failed_epipolar_H++;
    }
  }

  stats.num_inliers = image_pair.inliers.size();
  return score;
}

void ImagePairsInlierCount(ViewGraph& view_graph,
                           const std::unordered_map<camera_t, Camera>& cameras,
                           const std::unordered_map<image_t, Image>& images,
                           const InlierThresholdOptions& options,
                           bool clean_inliers) {
  InlierStats stats;
  for (auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (!clean_inliers && image_pair.inliers.size() > 0) continue;
    image_pair.inliers.clear();

    if (!image_pair.is_valid) continue;
    ImagePairInliers inlier_finder(image_pair, images, options, &cameras);
    inlier_finder.ScoreError();

    switch (inlier_finder.stats.type) {
      case GeometryConfig::ESSENTIAL:
        stats.num_essential_pairs++;
        stats.total_matches_essential += inlier_finder.stats.total_matches;
        stats.failed_epipolar_essential +=
            inlier_finder.stats.failed_epipolar_E;
        stats.failed_cheirality_essential +=
            inlier_finder.stats.failed_cheirality_E;
        stats.failed_angle_check_essential +=
            inlier_finder.stats.failed_angle_check_E;
        stats.failed_epipole_check_essential +=
            inlier_finder.stats.failed_epipole_check_E;
        stats.total_inliers_essential += inlier_finder.stats.num_inliers;
        break;
      case GeometryConfig::FUNDAMENTAL:
        stats.num_fundamental_pairs++;
        stats.total_matches_fundamental += inlier_finder.stats.total_matches;
        stats.failed_epipolar_fundamental +=
            inlier_finder.stats.failed_epipolar_F;
        stats.failed_cheirality_fundamental +=
            inlier_finder.stats.failed_cheirality_F;
        stats.total_inliers_fundamental += inlier_finder.stats.num_inliers;
        break;
      case GeometryConfig::HOMOGRAPHY:
        stats.num_homography_pairs++;
        stats.total_matches_homography += inlier_finder.stats.total_matches;
        stats.failed_epipolar_homography +=
            inlier_finder.stats.failed_epipolar_H;
        stats.total_inliers_homography += inlier_finder.stats.num_inliers;
        break;
      default:
        break;
    }
  }

  LOG(INFO) << "Image Pair Inlier Counting Summary:";
  LOG(INFO) << "  E pairs: " << stats.num_essential_pairs
            << ", matches: " << stats.total_matches_essential
            << ", inliers: " << stats.total_inliers_essential;
  LOG(INFO) << "  F pairs: " << stats.num_fundamental_pairs
            << ", matches: " << stats.total_matches_fundamental
            << ", inliers: " << stats.total_inliers_fundamental;
  LOG(INFO) << "  H pairs: " << stats.num_homography_pairs
            << ", matches: " << stats.total_matches_homography
            << ", inliers: " << stats.total_inliers_homography;
}

}  // namespace colmap::glomap
