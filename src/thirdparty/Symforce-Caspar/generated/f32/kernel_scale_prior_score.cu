#include "kernel_scale_prior_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ScalePriorScoreKernel(float* scale,
                          unsigned int scale_num_alloc,
                          SharedIndex* scale_indices,
                          float* inv_std,
                          unsigned int inv_std_num_alloc,
                          float* loss,
                          unsigned int loss_num_alloc,
                          float* const out_rTr,
                          size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  __shared__ SharedIndex scale_indices_loc[1024];
  scale_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? scale_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  LoadShared<1, float, float>(
      scale, 0 * scale_num_alloc, scale_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>(
        (float*)inout_shared, scale_indices_loc[threadIdx.x].target, r0);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r0 = r0 * r0;
    ReadIdx1<1024, float, float, float>(
        inv_std, 0 * inv_std_num_alloc, global_thread_idx, r1);
    r1 = r1 * r1;
    r1 = r0 * r1;
    r0 = 9.99999999999999955e-07;
    ReadIdx3<1024, float, float, float4>(
        loss, 0 * loss_num_alloc, global_thread_idx, r2, r3, r4);
    r5 = 0.00000000000000000e+00;
    r4 = fmaxf(r4, r5);
    r6 = sqrtf(r4);
    r7 = 5.00000000000000000e-01;
    r3 = fmaxf(r3, r0);
    r8 = r3 * r3;
    r9 = 2.00000000000000000e+00;
    r10 = r9 * r3;
    r11 = fmaxf(r0, r1);
    r12 = sqrtf(r11);
    r13 = -1.00000000000000000e+00;
    r10 = fmaf(r13, r8, r12 * r10);
    r10 = r1 <= r8 ? r1 : r10;
    r12 = 2.50000000000000000e+00;
    r14 = 1.00000000000000000e+00;
    r15 = 1.0 / r8;
    r15 = fmaf(r15, r1, r14);
    r14 = logf(r15);
    r14 = r14 * r8;
    r10 = r2 < r12 ? r14 : r10;
    r14 = 1.50000000000000000e+00;
    r15 = sqrtf(r15);
    r15 = r13 + r15;
    r15 = r9 * r15;
    r15 = r15 * r8;
    r10 = r2 < r14 ? r15 : r10;
    r10 = r2 < r7 ? r1 : r10;
    r10 = fmaxf(r5, r10);
    r10 = r4 * r10;
    r11 = 1.0 / r11;
    r10 = r10 * r11;
    r10 = sqrtf(r10);
    r10 = r1 <= r0 ? r6 : r10;
    r10 = r10 * r10;
    r10 = r1 * r10;
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r10);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void ScalePriorScore(float* scale,
                     unsigned int scale_num_alloc,
                     SharedIndex* scale_indices,
                     float* inv_std,
                     unsigned int inv_std_num_alloc,
                     float* loss,
                     unsigned int loss_num_alloc,
                     float* const out_rTr,
                     size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ScalePriorScoreKernel<<<n_blocks, 1024>>>(scale,
                                            scale_num_alloc,
                                            scale_indices,
                                            inv_std,
                                            inv_std_num_alloc,
                                            loss,
                                            loss_num_alloc,
                                            out_rTr,
                                            problem_size);
}

}  // namespace caspar