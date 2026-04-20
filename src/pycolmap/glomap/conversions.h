#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace colmap::glomap::py_helpers {

// Convert pybind11 numpy bool array / Eigen bool array to std::vector<bool>.
// std::vector<bool> is bit-packed — cannot use pybind11's default `stl.h`
// round-trip without this helper on the read-side.
inline std::vector<bool> EigenToStdVectorBool(
    const Eigen::Ref<const Eigen::Array<bool, Eigen::Dynamic, 1>>& arr) {
  std::vector<bool> out;
  out.reserve(arr.size());
  for (Eigen::Index i = 0; i < arr.size(); ++i) {
    out.push_back(arr[i]);
  }
  return out;
}

// Convert Eigen::VectorXd to std::vector<double> via element copy.
inline std::vector<double> EigenToStdVectorDouble(
    const Eigen::Ref<const Eigen::VectorXd>& vec) {
  return std::vector<double>(vec.data(), vec.data() + vec.size());
}

// Convert std::vector<bool> to Eigen::Array<bool,Dynamic,1> (safe copy).
inline Eigen::Array<bool, Eigen::Dynamic, 1> StdVectorBoolToEigen(
    const std::vector<bool>& v) {
  Eigen::Array<bool, Eigen::Dynamic, 1> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) out[i] = v[i];
  return out;
}

}  // namespace colmap::glomap::py_helpers
