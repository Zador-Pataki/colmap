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

#include "colmap/estimators/bundle_adjustment_ceres.h"

#include <memory>

#include <ceres/loss_function.h>

namespace colmap {

// Light (type, scale, weight) triple over native ``LossFunctionType``.
// Used by GlobalPositioner's 10-bucket per-observation loss cascade.
// The pycolmap binding accepts ``{name: str, scale, weight}`` dicts and
// maps the string name to the enum at the boundary.
struct LossConfig {
  CeresBundleAdjustmentOptions::LossFunctionType type =
      CeresBundleAdjustmentOptions::LossFunctionType::TRIVIAL;
  double scale = 1.0;
  double weight = 1.0;

  // Materialize the Ceres loss function described by this config.
  // Wraps in ``ScaledLoss(weight)`` when weight != 1.
  std::shared_ptr<ceres::LossFunction> CreateLossFunction() const {
    auto loss = colmap::CreateLossFunction(type, scale);
    if (weight != 1.0) {
      loss.reset(new ceres::ScaledLoss(
          loss.release(), weight, ceres::TAKE_OWNERSHIP));
    }
    return std::shared_ptr<ceres::LossFunction>(loss.release());
  }
};

}  // namespace colmap
