#include "kernel_pinhole_log_depth_fixed_rotation_fixed_scale_fixed_point_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeLogDepthFixedRotationFixedScaleFixedPointJtjnjtrDirectKernel(
        double* translation_njtr,
        unsigned int translation_njtr_num_alloc,
        SharedIndex* translation_njtr_indices,
        double* translation_jac,
        unsigned int translation_jac_num_alloc,
        double* const out_translation_njtr,
        unsigned int out_translation_njtr_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex translation_njtr_indices_loc[1024];
  translation_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? translation_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
}

void PinholeLogDepthFixedRotationFixedScaleFixedPointJtjnjtrDirect(
    double* translation_njtr,
    unsigned int translation_njtr_num_alloc,
    SharedIndex* translation_njtr_indices,
    double* translation_jac,
    unsigned int translation_jac_num_alloc,
    double* const out_translation_njtr,
    unsigned int out_translation_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeLogDepthFixedRotationFixedScaleFixedPointJtjnjtrDirectKernel<<<
      n_blocks,
      1024>>>(translation_njtr,
              translation_njtr_num_alloc,
              translation_njtr_indices,
              translation_jac,
              translation_jac_num_alloc,
              out_translation_njtr,
              out_translation_njtr_num_alloc,
              problem_size);
}

}  // namespace caspar