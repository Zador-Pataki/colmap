// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
// (BSD-3-Clause license, see LICENSE)

#pragma once

#include <memory>
#include <string>

#include <ceres/loss_function.h>

namespace colmap {

// Configuration for a per-channel Ceres loss function.
// `name` dispatches Create() to the matching ceres::LossFunction subclass.
// `scale` is the robustifier parameter (Huber threshold, Cauchy scale, etc.).
// `weight` multiplies via a ScaledLoss wrapper when != 1.0.
struct LossFunctionConfig {
  std::string name = "trivial";  // "trivial" | "huber" | "cauchy" | "arctan" | "softlone"
  double scale = 1.0;
  double weight = 1.0;

  LossFunctionConfig() = default;
  LossFunctionConfig(const std::string& n, double s, double w)
      : name(n), scale(s), weight(w) {}

  // Returns a shared_ptr-owned loss function. Pass .get() to Ceres with
  // DO_NOT_TAKE_OWNERSHIP to avoid double-free.
  std::shared_ptr<ceres::LossFunction> Create() const;
};

}  // namespace colmap
