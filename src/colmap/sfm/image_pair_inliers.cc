#include "colmap/sfm/image_pair_inliers.h"

#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/homography_matrix.h"

namespace colmap {
namespace {
constexpr double EPS = 1e-12;
#ifndef TWO_PI
constexpr double TWO_PI = 2.0 * EIGEN_PI;
#endif

// Single-pair PoseLib-style cheirality with min/max depth bounds.
// Native ``CheckCheirality`` (`colmap/geometry/pose.h`) is batched and
// has no depth bound — different surface — so this stays as a static
// helper colocated with its only caller.
//
// Code from PoseLib by Viktor Larsson.
bool CheckCheirality(const Rigid3d& pose,
                     const Eigen::Vector3d& x1,
                     const Eigen::Vector3d& x2,
                     double min_depth,
                     double max_depth) {
  // This code assumes that x1 and x2 are unit vectors.
  const Eigen::Vector3d Rx1 = pose.rotation() * x1;
  const double a = -Rx1.dot(x2);
  const double b1 = -Rx1.dot(pose.translation());
  const double b2 = x2.dot(pose.translation());
  // Note: we drop the factor 1.0/(1-a*a) since it is always positive.
  const double lambda1 = b1 - a * b2;
  const double lambda2 = -a * b1 + b2;
  min_depth = min_depth * (1 - a * a);
  max_depth = max_depth * (1 - a * a);
  bool status = lambda1 > min_depth && lambda2 > min_depth;
  status = status && (lambda1 < max_depth) && (lambda2 < max_depth);
  return status;
}

// F-cheirality orientation signum (no native counterpart).
// Code from GC-RANSAC by Daniel Barath.
double GetOrientationSignum(const Eigen::Matrix3d& F,
                            const Eigen::Vector3d& epipole,
                            const Eigen::Vector2d& pt1,
                            const Eigen::Vector2d& pt2) {
  double signum1 = F(0, 0) * pt2[0] + F(1, 0) * pt2[1] + F(2, 0);
  double signum2 = epipole(1) - epipole(2) * pt1[1];
  return signum1 * signum2;
}

// Depth-aware Sampson on Vec3 rays (divides by ``z + EPS`` first).
// Native ``ComputeSquaredSampsonError`` is the Vec2 overload — this is
// the depth-aware variant used when ``z`` carries depth meaning (e.g.
// ``Image::features_undist``).
double SampsonError(const Eigen::Matrix3d& E,
                    const Eigen::Vector3d& x1,
                    const Eigen::Vector3d& x2) {
  Eigen::Vector3d Ex1 = E * x1 / (EPS + x1[2]);
  Eigen::Vector3d Etx2 = E.transpose() * x2 / (EPS + x2[2]);
  double C = Ex1.dot(x2);
  double Cx = Ex1.head(2).squaredNorm();
  double Cy = Etx2.head(2).squaredNorm();
  return C * C / (Cx + Cy);
}

}  // namespace

using ViewGraph = colmap::CorrespondenceGraph;
using ImagePair = colmap::CorrespondenceGraph::ImagePair;

namespace {

enum class GeometryConfig {
  UNKNOWN,
  ESSENTIAL,
  FUNDAMENTAL,
  HOMOGRAPHY,
};

struct PerPairStats {
  GeometryConfig type = GeometryConfig::UNKNOWN;
  size_t total_matches = 0;
  size_t num_inliers = 0;
  // Essential matrix
  size_t failed_epipolar_E = 0;
  size_t failed_cheirality_E = 0;
  size_t failed_angle_check_E = 0;
  size_t failed_epipole_check_E = 0;
  // Fundamental matrix
  size_t failed_epipolar_F = 0;
  size_t failed_cheirality_F = 0;
  // Homography
  size_t failed_epipolar_H = 0;
};

class ImagePairInliers {
 public:
  ImagePairInliers(
      ImagePair& image_pair,
      const std::unordered_map<image_t, Image>& images,
      const InlierThresholdOptions& options,
      const std::unordered_map<camera_t, Camera>* cameras = nullptr)
      : image_pair(image_pair),
        images(images),
        cameras(cameras),
        options(options) {}

  // use the sampson error and put the inlier result into the image pair
  double ScoreError(PerPairStats& stats);

 protected:
  // Error for the case of essential matrix
  double ScoreErrorEssential(PerPairStats& stats);

  // Error for the case of fundamental matrix
  double ScoreErrorFundamental(PerPairStats& stats);

  // Error for the case of homography matrix
  double ScoreErrorHomography(PerPairStats& stats);

  ImagePair& image_pair;
  const std::unordered_map<image_t, Image>& images;
  const std::unordered_map<camera_t, Camera>* cameras;
  const InlierThresholdOptions& options;
};


double ImagePairInliers::ScoreError(PerPairStats& stats) {
  // Count inliers base on the type
  if (image_pair.two_view_geometry.config == colmap::TwoViewGeometry::PLANAR ||
      image_pair.two_view_geometry.config == colmap::TwoViewGeometry::PANORAMIC ||
      image_pair.two_view_geometry.config == colmap::TwoViewGeometry::PLANAR_OR_PANORAMIC) {
    stats.type = GeometryConfig::HOMOGRAPHY;
    return ScoreErrorHomography(stats);
  } else if (image_pair.two_view_geometry.config == colmap::TwoViewGeometry::UNCALIBRATED) {
    stats.type = GeometryConfig::FUNDAMENTAL;
    return ScoreErrorFundamental(stats);
  } else if (image_pair.two_view_geometry.config == colmap::TwoViewGeometry::CALIBRATED) {
    stats.type = GeometryConfig::ESSENTIAL;
    return ScoreErrorEssential(stats);
  }
  return 0;
}

double ImagePairInliers::ScoreErrorEssential(PerPairStats& stats) {
  const Rigid3d& cam2_from_cam1 = (*image_pair.two_view_geometry.cam2_from_cam1);
  const Eigen::Matrix3d E = EssentialMatrixFromPose(cam2_from_cam1);

  // eij = camera i on image j
  Eigen::Vector3d epipole12, epipole21;
  epipole12 = cam2_from_cam1.translation();
  epipole21 = Inverse(cam2_from_cam1).translation();

  if (epipole12.norm() > EPS) {  // Avoid normalizing zero vector
    epipole12.normalize();
  }
  if (epipole21.norm() > EPS) {  // Avoid normalizing zero vector
    epipole21.normalize();
  }

  if (epipole12[2] < 0) epipole12 = -epipole12;
  if (epipole21[2] < 0) epipole21 = -epipole21;

  if (image_pair.inliers.size() > 0) {
    image_pair.inliers.clear();
  }

  image_t image_id1 = image_pair.image_id1;
  image_t image_id2 = image_pair.image_id2;

  double thres = options.max_epipolar_error_E;

  // Conver the threshold from pixel space to normalized space
  thres = options.max_epipolar_error_E * 0.5 *
          (1. / cameras->at(images.at(image_id1).CameraId()).MeanFocalLength() +
           1. / cameras->at(images.at(image_id2).CameraId()).MeanFocalLength());

  // Square the threshold for faster computation
  double sq_threshold = thres * thres;
  double score = 0.;
  Eigen::Vector3d pt1, pt2;
  bool d_valid1;
  bool d_valid2;

  // TODO: determine the best threshold for triangulation angle
  // double thres_angle = std::cos(DegToRad(1.));
  double thres_epipole = std::cos(DegToRad(options.min_angle_from_epipole));
  double thres_epipole_nodepth = std::cos(DegToRad(3.));
  double thres_angle = 1;
  thres_angle += 1e-6;
  thres_epipole += 1e-6;

  stats.total_matches = image_pair.matches.rows();

  for (size_t k = 0; k < stats.total_matches; ++k) {
    // Use the undistorted features
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

    // --- GATE 1: Epipolar Error ---
    if (r2 < sq_threshold) {
      // --- GATE 2: Cheirality ---
      bool cheirality = CheckCheirality(cam2_from_cam1, pt1, pt2, 1e-2, 100.);

      // --- GATE 3: Degeneracy (Angle) ---
      double diff_angle = pt1.dot(cam2_from_cam1.rotation().inverse() * pt2);
      bool angle_check = (diff_angle < thres_angle);  // (Currently disabled)

      // --- GATE 4: Degeneracy (Epipole) ---
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
      // --- FINAL CHECK ---
      if (cheirality && angle_check && epipole_check) {
        // This is a true inlier
        score += r2;
        image_pair.inliers.push_back(k);
      } else {
        // It passed epipolar but failed a degeneracy check
        score += sq_threshold;

        // <<< NEW: Find out why it failed >>>
        if (!cheirality) {
          stats.failed_cheirality_E++;
        } else if (!angle_check) {
          stats.failed_angle_check_E++;
        } else if (!epipole_check) {
          stats.failed_epipole_check_E++;
        }
      }
    } else {
      // It failed the epipolar error check
      score += sq_threshold;
      stats.failed_epipolar_E++;
    }
  }

  stats.num_inliers = image_pair.inliers.size();
  return score;
}

double ImagePairInliers::ScoreErrorFundamental(PerPairStats& stats) {
  if (image_pair.inliers.size() > 0) {
    image_pair.inliers.clear();
  }

  Eigen::Vector3d epipole = (*image_pair.two_view_geometry.F).row(0).cross((*image_pair.two_view_geometry.F).row(2));

  bool status = false;
  for (auto i = 0; i < 3; i++) {
    if ((epipole(i) > EPS) || (epipole(i) < -EPS)) {
      status = true;
      break;
    }
  }
  if (!status) {
    epipole = (*image_pair.two_view_geometry.F).row(1).cross((*image_pair.two_view_geometry.F).row(2));
  }

  // First, get the orientation signum for every point
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
    const double r2 = ComputeSquaredSampsonError(
        pt1.homogeneous(), pt2.homogeneous(), *image_pair.two_view_geometry.F);

    if (r2 < sq_threshold) {
      signums.push_back(GetOrientationSignum((*image_pair.two_view_geometry.F), epipole, pt1, pt2));
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

  // If cannot distinguish the signum, the pair should be invalid
  if (positive_count == negative_count) return 0;

  // Then, if the signum is not consistent with the cheirality, discard the
  // point
  for (int k = 0; k < inliers_pre.size(); k++) {
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

double ImagePairInliers::ScoreErrorHomography(PerPairStats& stats) {
  if (image_pair.inliers.size() > 0) {
    image_pair.inliers.clear();
  }

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
    const double r2 = ComputeSquaredHomographyError(
        pt1, pt2, *image_pair.two_view_geometry.H);

    if (r2 < sq_threshold) {
      // TODO: cheirality check for homography. Is that a thing?
      bool cheirality = true;

      if (cheirality) {
        score += r2;
        image_pair.inliers.push_back(k);
      } else {
        score += sq_threshold;
      }
    } else {
      score += sq_threshold;
      stats.failed_epipolar_H++;
    }
  }

  stats.num_inliers = image_pair.inliers.size();
  return score;
}

}  // namespace

void ImagePairsInlierCount(ViewGraph& view_graph,
                           const std::unordered_map<camera_t, Camera>& cameras,
                           const std::unordered_map<image_t, Image>& images,
                           const InlierThresholdOptions& options,
                           bool clean_inliers) {
  for (auto& [pair_id, image_pair] : view_graph.MutableImagePairs()) {
    if (!clean_inliers && image_pair.inliers.size() > 0) continue;
    image_pair.inliers.clear();

    if (image_pair.is_valid == false) continue;
    ImagePairInliers inlier_finder(image_pair, images, options, &cameras);
    PerPairStats pair_stats;
    inlier_finder.ScoreError(pair_stats);
  }
}

}  // namespace colmap
