// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include <ceres/loss_function.h>

namespace colmap {

// Typed (name, scale, weight) triple describing a Ceres loss function. Allows
// downstream optimizers (e.g. GlobalPositioner with metric-depth + LC routing)
// to expose 10+ different loss buckets via configurable dict-typed properties
// from Python instead of hardcoded scales. ``weight != 1`` wraps the inner
// loss in ``ceres::ScaledLoss``.
struct LossFunctionConfig {
  std::string name = "trivial";  // {trivial, huber, cauchy, arctan, softlone}
  double scale = 1.0;
  double weight = 1.0;
};

// Materialize a Ceres ``LossFunction`` from a config. Returns ``nullptr`` if
// ``weight == 0`` (no residual contribution). Throws ``std::invalid_argument``
// for unknown ``name``.
inline std::shared_ptr<ceres::LossFunction> CreateLossFromConfig(
    const LossFunctionConfig& cfg) {
  if (cfg.weight == 0.0) {
    return nullptr;
  }

  ceres::LossFunction* inner = nullptr;
  if (cfg.name == "trivial") {
    // ``ceres::TrivialLoss`` ignores ``scale``; emit anyway for symmetry.
    inner = new ceres::TrivialLoss();
  } else if (cfg.name == "huber") {
    inner = new ceres::HuberLoss(cfg.scale);
  } else if (cfg.name == "cauchy") {
    inner = new ceres::CauchyLoss(cfg.scale);
  } else if (cfg.name == "arctan") {
    inner = new ceres::ArctanLoss(cfg.scale);
  } else if (cfg.name == "softlone" || cfg.name == "soft_l_one") {
    inner = new ceres::SoftLOneLoss(cfg.scale);
  } else {
    throw std::invalid_argument(
        "Unknown LossFunctionConfig.name: '" + cfg.name +
        "'. Expected one of {trivial, huber, cauchy, arctan, softlone}.");
  }

  if (cfg.weight == 1.0) {
    return std::shared_ptr<ceres::LossFunction>(inner);
  }
  return std::make_shared<ceres::ScaledLoss>(
      inner, cfg.weight, ceres::TAKE_OWNERSHIP);
}

}  // namespace colmap
