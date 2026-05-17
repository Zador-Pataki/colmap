#include "kernel_pinhole_intrinsics_random_walk_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeIntrinsicsRandomWalkJtjnjtrDirectKernel(
        float* prev_calib_njtr,
        unsigned int prev_calib_njtr_num_alloc,
        SharedIndex* prev_calib_njtr_indices,
        float* prev_calib_jac,
        unsigned int prev_calib_jac_num_alloc,
        float* next_calib_njtr,
        unsigned int next_calib_njtr_num_alloc,
        SharedIndex* next_calib_njtr_indices,
        float* next_calib_jac,
        unsigned int next_calib_jac_num_alloc,
        float* const out_prev_calib_njtr,
        unsigned int out_prev_calib_njtr_num_alloc,
        float* const out_next_calib_njtr,
        unsigned int out_next_calib_njtr_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex prev_calib_njtr_indices_loc[1024];
  prev_calib_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? prev_calib_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex next_calib_njtr_indices_loc[1024];
  next_calib_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? next_calib_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11;
  LoadShared<4, float, float>(next_calib_njtr,
                              0 * next_calib_njtr_num_alloc,
                              next_calib_njtr_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       next_calib_njtr_indices_loc[threadIdx.x].target,
                       r0,
                       r1,
                       r2,
                       r3);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(prev_calib_jac,
                                         0 * prev_calib_jac_num_alloc,
                                         global_thread_idx,
                                         r4,
                                         r5,
                                         r6,
                                         r7);
    ReadIdx4<1024, float, float, float4>(next_calib_jac,
                                         0 * next_calib_jac_num_alloc,
                                         global_thread_idx,
                                         r8,
                                         r9,
                                         r10,
                                         r11);
    r8 = r4 * r8;
    r0 = r0 * r8;
    r9 = r5 * r9;
    r1 = r1 * r9;
    r10 = r6 * r10;
    r2 = r2 * r10;
    r11 = r7 * r11;
    r3 = r3 * r11;
    WriteSum4<float, float>((float*)inout_shared, r0, r1, r2, r3);
  };
  FlushSumShared<4, float>(out_prev_calib_njtr,
                           0 * out_prev_calib_njtr_num_alloc,
                           prev_calib_njtr_indices_loc,
                           (float*)inout_shared);
  LoadShared<4, float, float>(prev_calib_njtr,
                              0 * prev_calib_njtr_num_alloc,
                              prev_calib_njtr_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       prev_calib_njtr_indices_loc[threadIdx.x].target,
                       r3,
                       r2,
                       r1,
                       r0);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r8 = r3 * r8;
    r9 = r2 * r9;
    r10 = r1 * r10;
    r11 = r0 * r11;
    WriteSum4<float, float>((float*)inout_shared, r8, r9, r10, r11);
  };
  FlushSumShared<4, float>(out_next_calib_njtr,
                           0 * out_next_calib_njtr_num_alloc,
                           next_calib_njtr_indices_loc,
                           (float*)inout_shared);
}

void PinholeIntrinsicsRandomWalkJtjnjtrDirect(
    float* prev_calib_njtr,
    unsigned int prev_calib_njtr_num_alloc,
    SharedIndex* prev_calib_njtr_indices,
    float* prev_calib_jac,
    unsigned int prev_calib_jac_num_alloc,
    float* next_calib_njtr,
    unsigned int next_calib_njtr_num_alloc,
    SharedIndex* next_calib_njtr_indices,
    float* next_calib_jac,
    unsigned int next_calib_jac_num_alloc,
    float* const out_prev_calib_njtr,
    unsigned int out_prev_calib_njtr_num_alloc,
    float* const out_next_calib_njtr,
    unsigned int out_next_calib_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeIntrinsicsRandomWalkJtjnjtrDirectKernel<<<n_blocks, 1024>>>(
      prev_calib_njtr,
      prev_calib_njtr_num_alloc,
      prev_calib_njtr_indices,
      prev_calib_jac,
      prev_calib_jac_num_alloc,
      next_calib_njtr,
      next_calib_njtr_num_alloc,
      next_calib_njtr_indices,
      next_calib_jac,
      next_calib_jac_num_alloc,
      out_prev_calib_njtr,
      out_prev_calib_njtr_num_alloc,
      out_next_calib_njtr,
      out_next_calib_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar