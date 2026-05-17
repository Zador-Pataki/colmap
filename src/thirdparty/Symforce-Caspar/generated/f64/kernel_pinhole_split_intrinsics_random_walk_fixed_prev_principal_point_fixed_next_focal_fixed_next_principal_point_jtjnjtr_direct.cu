#include "kernel_pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_focal_fixed_next_principal_point_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalFixedNextPrincipalPointJtjnjtrDirectKernel(
        double* prev_focal_njtr,
        unsigned int prev_focal_njtr_num_alloc,
        SharedIndex* prev_focal_njtr_indices,
        double* prev_focal_jac,
        unsigned int prev_focal_jac_num_alloc,
        double* const out_prev_focal_njtr,
        unsigned int out_prev_focal_njtr_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex prev_focal_njtr_indices_loc[1024];
  prev_focal_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? prev_focal_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
}

void PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalFixedNextPrincipalPointJtjnjtrDirect(
    double* prev_focal_njtr,
    unsigned int prev_focal_njtr_num_alloc,
    SharedIndex* prev_focal_njtr_indices,
    double* prev_focal_jac,
    unsigned int prev_focal_jac_num_alloc,
    double* const out_prev_focal_njtr,
    unsigned int out_prev_focal_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalFixedNextPrincipalPointJtjnjtrDirectKernel<<<
      n_blocks,
      1024>>>(prev_focal_njtr,
              prev_focal_njtr_num_alloc,
              prev_focal_njtr_indices,
              prev_focal_jac,
              prev_focal_jac_num_alloc,
              out_prev_focal_njtr,
              out_prev_focal_njtr_num_alloc,
              problem_size);
}

}  // namespace caspar