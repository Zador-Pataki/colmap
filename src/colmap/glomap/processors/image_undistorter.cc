#include "colmap/glomap/processors/image_undistorter.h"

#include <colmap/util/threading.h>

namespace colmap::glomap {

void UndistortImages(std::unordered_map<camera_t, Camera>& cameras,
                     std::unordered_map<image_t, Image>& images,
                     bool clean_points) {
  std::vector<image_t> image_ids;
  for (auto& [image_id, image] : images) {
    const size_t num_points = image.features.size();
    if (image.features_undist.size() == num_points && !clean_points) continue;
    image_ids.push_back(image_id);
  }

  colmap::ThreadPool thread_pool(colmap::ThreadPool::kMaxNumThreads);

  LOG(INFO) << "Undistorting images..";
  const int num_images = image_ids.size();
  for (int image_idx = 0; image_idx < num_images; image_idx++) {
    Image& image = images[image_ids[image_idx]];
    const int num_points = image.features.size();
    if (image.features_undist.size() == static_cast<size_t>(num_points) &&
        !clean_points)
      continue;

    const Camera& camera = cameras[image.camera_id];

    thread_pool.AddTask([&image, &camera, num_points]() {
      image.features_undist.clear();
      image.features_undist.reserve(num_points);
      for (int i = 0; i < num_points; i++) {
        // colmap4: CamFromImg returns std::optional<Eigen::Vector2d>.
        std::optional<Eigen::Vector2d> pt_calc =
            camera.camera.CamFromImg(image.features[i]);
        if (pt_calc.has_value()) {
          image.features_undist.emplace_back(
              pt_calc->homogeneous().normalized());
        } else {
          image.features_undist.emplace_back(Eigen::Vector3d::Zero());
        }
      }
    });
  }

  thread_pool.Wait();
  LOG(INFO) << "Image undistortion done";
}

}  // namespace colmap::glomap
