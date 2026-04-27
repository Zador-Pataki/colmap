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
    std::unordered_map<track_t, Point3D>& tracks,
    double max_angle_error) {
  int counter = 0;
  double thres = std::cos(DegToRad(max_angle_error));
  double thres_uncalib = std::cos(DegToRad(max_angle_error * 2));
  for (auto& [track_id, point3D] : tracks) {
    std::vector<TrackElement> elements_new;
    for (const auto& el : point3D.track.Elements()) {
      const Image& image = images.at(el.image_id);
      const Eigen::Vector3d& feature_undist =
          image.features_undist.at(el.point2D_idx);
      Eigen::Vector3d pt_calc = image.cam_from_world * point3D.xyz;
      if (pt_calc(2) < EPS) continue;

      pt_calc = pt_calc.normalized();
      double thres_cam = (cameras.at(image.CameraId()).has_prior_focal_length)
                             ? thres
                             : thres_uncalib;

      if (pt_calc.dot(feature_undist) > thres_cam) {
        elements_new.emplace_back(el);
      }
    }
    if (elements_new.size() != point3D.track.Length()) {
      counter++;
      point3D.track.SetElements(std::move(elements_new));
    }
  }
  LOG(INFO) << "Filtered " << counter << " / " << tracks.size()
            << " tracks by angle error";
  return counter;
}

int TrackFilter::FilterTrackTriangulationAngle(
    ViewGraph& view_graph,
    const std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Point3D>& tracks,
    double min_angle) {
  int counter = 0;
  double thres = std::cos(DegToRad(min_angle));
  for (auto& [track_id, point3D] : tracks) {
    std::vector<Eigen::Vector3d> pts_calc;
    pts_calc.reserve(point3D.track.Length());
    for (const auto& el : point3D.track.Elements()) {
      const Image& image = images.at(el.image_id);
      Eigen::Vector3d pt_calc =
          (point3D.xyz - ImageCenter(image)).normalized();
      pts_calc.emplace_back(pt_calc);
    }
    bool status = false;
    for (size_t i = 0; i < pts_calc.size(); i++) {
      for (size_t j = i + 1; j < pts_calc.size(); j++) {
        if (pts_calc[i].dot(pts_calc[j]) < thres) {
          status = true;
          break;
        }
      }
    }

    // If the triangulation angle is too small, just remove it
    if (!status) {
      counter++;
      point3D.track.Elements().clear();
    }
  }
  LOG(INFO) << "Filtered " << counter << " / " << tracks.size()
            << " tracks by too small triangulation angle";
  return counter;
}

}  // namespace glomap_ra
}  // namespace colmap