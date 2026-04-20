#include "colmap/glomap/image.h"

#include "pycolmap/glomap/conversions.h"
#include "pycolmap/glomap/types.h"
#include "pycolmap/helpers.h"

#include <cmath>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <vector>

#include <colmap/geometry/rigid3.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
using namespace pybind11::literals;

using colmap::Rigid3d;
using colmap::glomap::camera_t;
using colmap::glomap::CameraMap;
using colmap::glomap::Image;
using colmap::glomap::ImageMap;
using colmap::glomap::image_t;
using colmap::glomap::py_helpers::EigenToStdVectorBool;
using colmap::glomap::py_helpers::EigenToStdVectorDouble;
using colmap::glomap::py_helpers::StdVectorBoolToEigen;

using EigenBoolArray = Eigen::Array<bool, Eigen::Dynamic, 1>;

static std::shared_ptr<Image> MakeImage(image_t image_id,
                                        camera_t camera_id,
                                        std::string file_name) {
  return std::make_shared<Image>(image_id, camera_id, std::move(file_name));
}

void BindGlomapImage(py::module& m) {
  py::classh<Image> PyImage(m, "Image");
  PyImage.def(py::init<>())
      .def(py::init(&MakeImage),
           "image_id"_a = image_t{static_cast<image_t>(-1)},
           "camera_id"_a = camera_t{static_cast<camera_t>(-1)},
           "file_name"_a = "")
      // --- Basic properties ---
      .def_property_readonly("image_id",
                             [](const Image& self) { return self.image_id; })
      .def_property(
          "camera_id",
          [](const Image& self) { return self.camera_id; },
          [](Image& self, camera_t v) { self.camera_id = v; })
      .def_property_readonly("file_name",
                             [](const Image& self) { return self.file_name; })
      .def_property(
          "cam_from_world",
          [](const Image& self) { return self.cam_from_world; },
          [](Image& self, const Rigid3d& value) {
            self.cam_from_world = value;
          })
      // --- Feature properties ---
      .def_property(
          "features",
          [](const Image& self) { return py::cast(self.features); },
          [](Image& self, const std::vector<Eigen::Vector2d>& value) {
            self.features = value;
          },
          "Distorted feature points in pixels (numpy Nx2).")
      .def_property(
          "features_undist",
          [](const Image& self) { return py::cast(self.features_undist); },
          [](Image& self, const std::vector<Eigen::Vector3d>& value) {
            self.features_undist = value;
          },
          "Normalized feature rays (numpy Nx3).")
      // --- Depth priors ---
      .def_property(
          "depth_priors",
          [](const Image& self) -> Eigen::VectorXd {
            return Eigen::Map<const Eigen::VectorXd>(self.depth_priors.data(),
                                                     self.depth_priors.size());
          },
          [](Image& self, const Eigen::VectorXd& value) {
            self.depth_priors = EigenToStdVectorDouble(value);
          },
          "Estimated depth values (numpy N,).")
      .def_property(
          "depth_prior_stddevs",
          [](const Image& self) -> Eigen::VectorXd {
            return Eigen::Map<const Eigen::VectorXd>(
                self.depth_prior_stddevs.data(),
                self.depth_prior_stddevs.size());
          },
          [](Image& self, const Eigen::VectorXd& value) {
            self.depth_prior_stddevs = EigenToStdVectorDouble(value);
          },
          "Uncertainties (std dev) for depth priors (numpy N,).")
      .def_property(
          "depth_prior_validity",
          [](const Image& self) -> EigenBoolArray {
            return StdVectorBoolToEigen(self.depth_prior_validity);
          },
          [](Image& self, const EigenBoolArray& value) {
            self.depth_prior_validity = EigenToStdVectorBool(value);
          },
          "Validity flags for depth priors (numpy N, dtype=bool).")
      // --- Inlier / outlier / anchor / excluded flags (std::vector<bool>) ---
      .def_property(
          "is_inlier",
          [](const Image& self) -> EigenBoolArray {
            return StdVectorBoolToEigen(self.is_inlier);
          },
          [](Image& self, const EigenBoolArray& value) {
            self.is_inlier = EigenToStdVectorBool(value);
          },
          "Per-feature inlier flag for second GP. If true, use trivial loss.")
      .def_property(
          "is_depth_outlier",
          [](const Image& self) -> EigenBoolArray {
            return StdVectorBoolToEigen(self.is_depth_outlier);
          },
          [](Image& self, const EigenBoolArray& value) {
            self.is_depth_outlier = EigenToStdVectorBool(value);
          },
          "Per-feature MDRP depth outlier flag. If true, use robust loss.")
      .def_property(
          "is_track_anchor",
          [](const Image& self) -> EigenBoolArray {
            return StdVectorBoolToEigen(self.is_track_anchor);
          },
          [](Image& self, const EigenBoolArray& value) {
            self.is_track_anchor = EigenToStdVectorBool(value);
          },
          "Per-feature track-anchor flag.")
      .def_property(
          "is_excluded",
          [](const Image& self) -> EigenBoolArray {
            return StdVectorBoolToEigen(self.is_excluded);
          },
          [](Image& self, const EigenBoolArray& value) {
            self.is_excluded = EigenToStdVectorBool(value);
          },
          "Per-feature hard-exclusion flag.")
      // --- Angular uncertainties ---
      .def_property(
          "angular_stddevs",
          [](const Image& self) { return py::cast(self.angular_stddevs); },
          [](Image& self, const std::vector<Eigen::Vector2d>& value) {
            self.angular_stddevs = value;
          },
          "Angular uncertainties as (sigma_x, sigma_y) per feature (Nx2).")
      .def_property(
          "angular_cholesky_xy",
          [](const Image& self) { return py::cast(self.angular_cholesky_xy); },
          [](Image& self, const std::vector<Eigen::Vector3d>& value) {
            self.angular_cholesky_xy = value;
          },
          "Cholesky factor (L00, L10, L11) for XY precision matrix.")
      .def_property(
          "angular_stddevs_z",
          [](const Image& self) -> Eigen::VectorXd {
            return Eigen::Map<const Eigen::VectorXd>(
                self.angular_stddevs_z.data(), self.angular_stddevs_z.size());
          },
          [](Image& self, const Eigen::VectorXd& value) {
            self.angular_stddevs_z = EigenToStdVectorDouble(value);
          },
          "Z-component stddev per feature.")
      // --- Per-image scale ---
      .def_property(
          "log_scale",
          [](const Image& self) { return self.log_scale; },
          [](Image& self, double v) { self.log_scale = v; })
      .def_property(
          "log_scale_stddev",
          [](const Image& self) { return self.log_scale_stddev; },
          [](Image& self, double v) {
            if (v < 0)
              throw std::invalid_argument("stddev must be non-negative");
            self.log_scale_stddev = v;
          })
      // --- Gravity ---
      .def_readwrite("gravity_info", &Image::gravity_info)
      .def_readwrite("gravity_sigma", &Image::gravity_sigma)
      // --- Other flags ---
      .def_readonly("is_registered", &Image::is_registered)
      .def_readwrite("cluster_id", &Image::cluster_id)
      // --- Batch update (depth priors) ---
      .def(
          "update_depth_priors",
          [](Image& self,
             std::optional<Eigen::VectorXd> depth_priors,
             std::optional<Eigen::VectorXd> depth_prior_stddevs,
             std::optional<EigenBoolArray> depth_prior_validity,
             std::optional<double> log_scale,
             std::optional<double> log_scale_stddev) {
            if (depth_priors) {
              self.depth_priors = EigenToStdVectorDouble(*depth_priors);
            }
            if (depth_prior_stddevs) {
              self.depth_prior_stddevs =
                  EigenToStdVectorDouble(*depth_prior_stddevs);
            }
            if (depth_prior_validity) {
              self.depth_prior_validity =
                  EigenToStdVectorBool(*depth_prior_validity);
            }
            if (log_scale) self.log_scale = *log_scale;
            if (log_scale_stddev) {
              if (*log_scale_stddev < 0)
                throw std::invalid_argument("stddev must be non-negative");
              self.log_scale_stddev = *log_scale_stddev;
            }
          },
          py::arg("depth_priors") = py::none(),
          py::arg("depth_prior_stddevs") = py::none(),
          py::arg("depth_prior_validity") = py::none(),
          py::arg("log_scale") = py::none(),
          py::arg("log_scale_stddev") = py::none(),
          "Batch update depth-prior attributes in a single call.")
      // --- Batch update (angular) ---
      .def(
          "update_angular",
          [](Image& self,
             std::optional<std::vector<Eigen::Vector3d>> angular_cholesky_xy,
             std::optional<Eigen::VectorXd> angular_stddevs_z,
             std::optional<std::vector<Eigen::Vector2d>> angular_stddevs) {
            if (angular_cholesky_xy) {
              self.angular_cholesky_xy = *angular_cholesky_xy;
            }
            if (angular_stddevs_z) {
              self.angular_stddevs_z =
                  EigenToStdVectorDouble(*angular_stddevs_z);
            }
            if (angular_stddevs) self.angular_stddevs = *angular_stddevs;
          },
          py::arg("angular_cholesky_xy") = py::none(),
          py::arg("angular_stddevs_z") = py::none(),
          py::arg("angular_stddevs") = py::none(),
          "Batch update angular covariance attributes.")
      // --- Repr ---
      .def("__repr__", [](const Image& self) {
        std::ostringstream ss;
        ss.precision(3);
        ss << "Image(image_id=" << self.image_id
           << ", file_name=" << self.file_name
           << ", camera_id=" << self.camera_id
           << ", num_features=" << self.features.size();
        if (!self.depth_priors.empty()) {
          const double scale = std::exp(self.log_scale);
          ss << ", scale=" << scale;
          const size_t valid_depth_count =
              std::accumulate(self.depth_prior_validity.begin(),
                              self.depth_prior_validity.end(),
                              static_cast<size_t>(0));
          ss << ", num_valid_depths=" << valid_depth_count << "/"
             << self.depth_priors.size();
        }
        ss << ")";
        return ss.str();
      });
  MakeDataclass(PyImage);

  py::bind_map<ImageMap>(m, "MapImageIdToImage");
  py::bind_map<CameraMap>(m, "MapCameraIdToCamera");

  // Module-level batch extractor. Avoids per-image pybind overhead when mpsfm
  // iterates thousands of images.
  m.def(
      "extract_all_image_data",
      [](const ImageMap& images) {
        py::dict result;
        for (const auto& [imid, im] : images) {
          const size_t n_feat = im.features.size();

          py::array_t<double> features({(py::ssize_t)n_feat, (py::ssize_t)2});
          auto feat_mut = features.mutable_unchecked<2>();
          for (size_t i = 0; i < n_feat; ++i) {
            feat_mut(i, 0) = im.features[i][0];
            feat_mut(i, 1) = im.features[i][1];
          }

          py::array_t<double> depth_priors(im.depth_priors.size());
          auto dp_mut = depth_priors.mutable_unchecked<1>();
          for (size_t i = 0; i < im.depth_priors.size(); ++i) {
            dp_mut(i) = im.depth_priors[i];
          }

          py::array_t<bool> dpv(im.depth_prior_validity.size());
          auto dpv_mut = dpv.mutable_unchecked<1>();
          for (size_t i = 0; i < im.depth_prior_validity.size(); ++i) {
            dpv_mut(i) = im.depth_prior_validity[i];
          }

          py::dict img_dict;
          img_dict["features"] = std::move(features);
          img_dict["depth_priors"] = std::move(depth_priors);
          img_dict["depth_prior_validity"] = std::move(dpv);
          img_dict["camera_id"] = im.camera_id;
          img_dict["file_name"] = py::str(im.file_name);

          result[py::int_(imid)] = std::move(img_dict);
        }
        return result;
      },
      py::arg("images"),
      "Batch extract image data (features, depth_priors, "
      "depth_prior_validity, camera_id, file_name) for all images. "
      "Returns dict[image_id -> dict].");
}
