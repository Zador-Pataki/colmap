#include "kernel_pinhole_intrinsics_random_walk_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeIntrinsicsRandomWalkScoreKernel(float* prev_calib,
                                           unsigned int prev_calib_num_alloc,
                                           SharedIndex* prev_calib_indices,
                                           float* next_calib,
                                           unsigned int next_calib_num_alloc,
                                           SharedIndex* next_calib_indices,
                                           float* inv_std,
                                           unsigned int inv_std_num_alloc,
                                           float* const out_rTr,
                                           size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex prev_calib_indices_loc[1024];
  prev_calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? prev_calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex next_calib_indices_loc[1024];
  next_calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? next_calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11;
  LoadShared<4, float, float>(next_calib,
                              0 * next_calib_num_alloc,
                              next_calib_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       next_calib_indices_loc[threadIdx.x].target,
                       r0,
                       r1,
                       r2,
                       r3);
  };
  __syncthreads();
  LoadShared<4, float, float>(prev_calib,
                              0 * prev_calib_num_alloc,
                              prev_calib_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       prev_calib_indices_loc[threadIdx.x].target,
                       r4,
                       r5,
                       r6,
                       r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r8 = -1.00000000000000000e+00;
    r7 = fmaf(r7, r8, r3);
    r7 = r7 * r7;
    ReadIdx4<1024, float, float, float4>(
        inv_std, 0 * inv_std_num_alloc, global_thread_idx, r3, r9, r10, r11);
    r11 = r11 * r11;
    r5 = fmaf(r5, r8, r1);
    r5 = r5 * r5;
    r9 = r9 * r9;
    r9 = fmaf(r5, r9, r7 * r11);
    r4 = fmaf(r4, r8, r0);
    r4 = r4 * r4;
    r3 = r3 * r3;
    r8 = fmaf(r6, r8, r2);
    r8 = r8 * r8;
    r10 = r10 * r10;
    r9 = fmaf(r4, r3, r9);
    r9 = fmaf(r8, r10, r9);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r9);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void PinholeIntrinsicsRandomWalkScore(float* prev_calib,
                                      unsigned int prev_calib_num_alloc,
                                      SharedIndex* prev_calib_indices,
                                      float* next_calib,
                                      unsigned int next_calib_num_alloc,
                                      SharedIndex* next_calib_indices,
                                      float* inv_std,
                                      unsigned int inv_std_num_alloc,
                                      float* const out_rTr,
                                      size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeIntrinsicsRandomWalkScoreKernel<<<n_blocks, 1024>>>(
      prev_calib,
      prev_calib_num_alloc,
      prev_calib_indices,
      next_calib,
      next_calib_num_alloc,
      next_calib_indices,
      inv_std,
      inv_std_num_alloc,
      out_rTr,
      problem_size);
}

}  // namespace caspar