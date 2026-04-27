#include "colmap/sfm/track_filter_glomap.h"

#include "colmap/math/math.h"

namespace colmap {
namespace glomap_ra {

namespace {
inline Eigen::Vector3d ImageCenter(const Image& image) {
  return image.cam_from_world.rotation().inverse() *
         -image.cam_from_world.translation();
}
}  // namespace


constexpr double EPS = 1e-12;
constexpr double HALF_PI = 3.141592653589793238462643383279502884L / 2;
constexpr double TWO_PI = 2 * 3.141592653589793238462643383279502884L;

using ViewGraph = colmap::CorrespondenceGraph;
using ImagePair = colmap::CorrespondenceGraph::ImagePair;


int TrackFilter::FilterTracksByAngle(
    ViewGraph& view_graph,
    const std::unordered_map<camera_t, Camera>& cameras,
    const std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks,
    double max_angle_error) {
  int counter = 0;
  double thres = std::cos(DegToRad(max_angle_error));
  double thres_uncalib = std::cos(DegToRad(max_angle_error * 2));
  for (auto& [track_id, track] : tracks) {
    std::vector<Observation> observation_new;
    for (auto& [image_id, feature_id] : track.observations) {
      const Image& image = images.at(image_id);
      // const Camera& camera = image.camera;
      const Eigen::Vector3d& feature_undist =
          image.features_undist.at(feature_id);
      Eigen::Vector3d pt_calc = image.cam_from_world * track.xyz;
      if (pt_calc(2) < EPS) continue;

      pt_calc = pt_calc.normalized();
      double thres_cam = (cameras.at(image.CameraId()).has_prior_focal_length)
                             ? thres
                             : thres_uncalib;

      if (pt_calc.dot(feature_undist) > thres_cam) {
        observation_new.emplace_back(std::make_pair(image_id, feature_id));
      }
    }
    if (observation_new.size() != track.observations.size()) {
      counter++;
      track.observations = observation_new;
    }
  }
  LOG(INFO) << "Filtered " << counter << " / " << tracks.size()
            << " tracks by angle error";
  return counter;
}

int TrackFilter::FilterTrackTriangulationAngle(
    ViewGraph& view_graph,
    const std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks,
    double min_angle) {
  int counter = 0;
  double thres = std::cos(DegToRad(min_angle));
  for (auto& [track_id, track] : tracks) {
    std::vector<Observation> observation_new;
    std::vector<Eigen::Vector3d> pts_calc;
    pts_calc.reserve(track.observations.size());
    for (auto& [image_id, feature_id] : track.observations) {
      const Image& image = images.at(image_id);
      Eigen::Vector3d pt_calc = (track.xyz - ImageCenter(image)).normalized();
      pts_calc.emplace_back(pt_calc);
    }
    bool status = false;
    for (int i = 0; i < track.observations.size(); i++) {
      for (int j = i + 1; j < track.observations.size(); j++) {
        if (pts_calc[i].dot(pts_calc[j]) < thres) {
          status = true;
          break;
        }
      }
    }

    // If the triangulation angle is too small, just remove it
    if (!status) {
      counter++;
      track.observations.clear();
    }
  }
  LOG(INFO) << "Filtered " << counter << " / " << tracks.size()
            << " tracks by too small triangulation angle";
  return counter;
}

}  // namespace glomap_ra
}  // namespace colmap