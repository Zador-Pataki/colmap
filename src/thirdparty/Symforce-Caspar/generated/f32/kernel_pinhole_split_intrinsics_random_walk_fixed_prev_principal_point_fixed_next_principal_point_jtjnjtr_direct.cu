#include "kernel_pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_principal_point_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextPrincipalPointJtjnjtrDirectKernel(
        float* prev_focal_njtr,
        unsigned int prev_focal_njtr_num_alloc,
        SharedIndex* prev_focal_njtr_indices,
        float* prev_focal_jac,
        unsigned int prev_focal_jac_num_alloc,
        float* next_focal_njtr,
        unsigned int next_focal_njtr_num_alloc,
        SharedIndex* next_focal_njtr_indices,
        float* next_focal_jac,
        unsigned int next_focal_jac_num_alloc,
        float* const out_prev_focal_njtr,
        unsigned int out_prev_focal_njtr_num_alloc,
        float* const out_next_focal_njtr,
        unsigned int out_next_focal_njtr_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ SharedIndex prev_focal_njtr_indices_loc[1024];
  prev_focal_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? prev_focal_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex next_focal_njtr_indices_loc[1024];
  next_focal_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? next_focal_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5;
  LoadShared<2, float, float>(next_focal_njtr,
                              0 * next_focal_njtr_num_alloc,
                              next_focal_njtr_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared,
                       next_focal_njtr_indices_loc[threadIdx.x].target,
                       r0,
                       r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(prev_focal_jac,
                                         0 * prev_focal_jac_num_alloc,
                                         global_thread_idx,
                                         r2,
                                         r3);
    ReadIdx2<1024, float, float, float2>(next_focal_jac,
                                         0 * next_focal_jac_num_alloc,
                                         global_thread_idx,
                                         r4,
                                         r5);
    r4 = r2 * r4;
    r0 = r0 * r4;
    r5 = r3 * r5;
    r1 = r1 * r5;
    WriteSum2<float, float>((float*)inout_shared, r0, r1);
  };
  FlushSumShared<2, float>(out_prev_focal_njtr,
                           0 * out_prev_focal_njtr_num_alloc,
                           prev_focal_njtr_indices_loc,
                           (float*)inout_shared);
  LoadShared<2, float, float>(prev_focal_njtr,
                              0 * prev_focal_njtr_num_alloc,
                              prev_focal_njtr_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared,
                       prev_focal_njtr_indices_loc[threadIdx.x].target,
                       r1,
                       r0);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r4 = r1 * r4;
    r5 = r0 * r5;
    WriteSum2<float, float>((float*)inout_shared, r4, r5);
  };
  FlushSumShared<2, float>(out_next_focal_njtr,
                           0 * out_next_focal_njtr_num_alloc,
                           next_focal_njtr_indices_loc,
                           (float*)inout_shared);
}

void PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextPrincipalPointJtjnjtrDirect(
    float* prev_focal_njtr,
    unsigned int prev_focal_njtr_num_alloc,
    SharedIndex* prev_focal_njtr_indices,
    float* prev_focal_jac,
    unsigned int prev_focal_jac_num_alloc,
    float* next_focal_njtr,
    unsigned int next_focal_njtr_num_alloc,
    SharedIndex* next_focal_njtr_indices,
    float* next_focal_jac,
    unsigned int next_focal_jac_num_alloc,
    float* const out_prev_focal_njtr,
    unsigned int out_prev_focal_njtr_num_alloc,
    float* const out_next_focal_njtr,
    unsigned int out_next_focal_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextPrincipalPointJtjnjtrDirectKernel<<<
      n_blocks,
      1024>>>(prev_focal_njtr,
              prev_focal_njtr_num_alloc,
              prev_focal_njtr_indices,
              prev_focal_jac,
              prev_focal_jac_num_alloc,
              next_focal_njtr,
              next_focal_njtr_num_alloc,
              next_focal_njtr_indices,
              next_focal_jac,
              next_focal_jac_num_alloc,
              out_prev_focal_njtr,
              out_prev_focal_njtr_num_alloc,
              out_next_focal_njtr,
              out_next_focal_njtr_num_alloc,
              problem_size);
}

}  // namespace caspar