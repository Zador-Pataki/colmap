#include "colmap/sfm/track_filter_glomap.h"

#include "colmap/geometry/triangulation.h"
#include "colmap/math/math.h"

namespace colmap {
namespace sfm_ext {

constexpr double EPS = 1e-12;

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
  // Mirrors ObservationManager::FindPoints3DWithSmallTriangulationAngle but
  // operates on the dict-of-tracks state model used by the glomap RA pipeline
  // instead of a Reconstruction. Shares the underlying angle math via
  // CalculateTriangulationAngle for consistency with native colmap.
  int counter = 0;
  const double min_angle_rad = DegToRad(min_angle);
  std::unordered_map<image_t, Eigen::Vector3d> proj_centers;
  for (auto& [track_id, point3D] : tracks) {
    bool keep_point = false;
    const auto& elements = point3D.track.Elements();
    for (size_t i1 = 0; i1 < elements.size() && !keep_point; ++i1) {
      const image_t image_id1 = elements[i1].image_id;
      auto it1 = proj_centers.find(image_id1);
      if (it1 == proj_centers.end()) {
        // Direct read of the RA-shadow ``cam_from_world`` field instead of
        // Image::ProjectionCenter(), which goes through frame_ptr_ that the
        // glomap RA pipeline does not populate. This is the standard idiom
        // throughout this file (see FilterTracksByAngle above).
        const Rigid3d& cfw = images.at(image_id1).cam_from_world;
        it1 = proj_centers
                  .emplace(image_id1,
                           cfw.rotation().inverse() * -cfw.translation())
                  .first;
      }
      const Eigen::Vector3d& proj_center1 = it1->second;
      for (size_t i2 = 0; i2 < i1; ++i2) {
        const Eigen::Vector3d& proj_center2 =
            proj_centers.at(elements[i2].image_id);
        const double tri_angle = CalculateTriangulationAngle(
            proj_center1, proj_center2, point3D.xyz);
        if (tri_angle >= min_angle_rad) {
          keep_point = true;
          break;
        }
      }
    }

    // If the triangulation angle is too small, just remove it
    if (!keep_point) {
      counter++;
      point3D.track.SetElements({});
    }
  }
  LOG(INFO) << "Filtered " << counter << " / " << tracks.size()
            << " tracks by too small triangulation angle";
  return counter;
}

}  // namespace sfm_ext
}  // namespace colmap