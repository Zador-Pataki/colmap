#include "kernel_pinhole_intrinsics_prior_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeIntrinsicsPriorScoreKernel(float* calib,
                                      unsigned int calib_num_alloc,
                                      SharedIndex* calib_indices,
                                      float* prior,
                                      unsigned int prior_num_alloc,
                                      float* inv_std,
                                      unsigned int inv_std_num_alloc,
                                      float* const out_rTr,
                                      size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex calib_indices_loc[1024];
  calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(
        inv_std, 0 * inv_std_num_alloc, global_thread_idx, r0, r1, r2, r3);
    r2 = r2 * r2;
  };
  LoadShared<4, float, float>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       calib_indices_loc[threadIdx.x].target,
                       r4,
                       r5,
                       r6,
                       r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(
        prior, 0 * prior_num_alloc, global_thread_idx, r8, r9, r10, r11);
    r12 = -1.00000000000000000e+00;
    r10 = fmaf(r10, r12, r6);
    r10 = r10 * r10;
    r3 = r3 * r3;
    r11 = fmaf(r11, r12, r7);
    r11 = r11 * r11;
    r11 = fmaf(r3, r11, r2 * r10);
    r0 = r0 * r0;
    r8 = fmaf(r8, r12, r4);
    r8 = r8 * r8;
    r1 = r1 * r1;
    r12 = fmaf(r9, r12, r5);
    r12 = r12 * r12;
    r11 = fmaf(r0, r8, r11);
    r11 = fmaf(r1, r12, r11);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r11);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void PinholeIntrinsicsPriorScore(float* calib,
                                 unsigned int calib_num_alloc,
                                 SharedIndex* calib_indices,
                                 float* prior,
                                 unsigned int prior_num_alloc,
                                 float* inv_std,
                                 unsigned int inv_std_num_alloc,
                                 float* const out_rTr,
                                 size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeIntrinsicsPriorScoreKernel<<<n_blocks, 1024>>>(calib,
                                                        calib_num_alloc,
                                                        calib_indices,
                                                        prior,
                                                        prior_num_alloc,
                                                        inv_std,
                                                        inv_std_num_alloc,
                                                        out_rTr,
                                                        problem_size);
}

}  // namespace caspar