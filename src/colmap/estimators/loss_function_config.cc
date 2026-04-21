// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
// (BSD-3-Clause license, see LICENSE)

#include "colmap/estimators/loss_function_config.h"

#include "colmap/util/logging.h"

namespace colmap {

std::shared_ptr<ceres::LossFunction> LossFunctionConfig::Create() const {
  ceres::LossFunction* raw = nullptr;
  if (name == "trivial") {
    raw = new ceres::TrivialLoss();
  } else if (name == "huber") {
    raw = new ceres::HuberLoss(scale);
  } else if (name == "cauchy") {
    raw = new ceres::CauchyLoss(scale);
  } else if (name == "arctan") {
    raw = new ceres::ArctanLoss(scale);
  } else if (name == "softlone") {
    raw = new ceres::SoftLOneLoss(scale);
  } else {
    LOG(FATAL) << "Unknown loss function name: " << name;
  }

  if (std::abs(weight - 1.0) < 1e-9) {
    return std::shared_ptr<ceres::LossFunction>(raw);
  }
  return std::shared_ptr<ceres::LossFunction>(
      new ceres::ScaledLoss(raw, weight, ceres::TAKE_OWNERSHIP));
}

}  // namespace colmap
