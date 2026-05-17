#include "kernel_scale_prior_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ScalePriorResJacKernel(float* scale,
                           unsigned int scale_num_alloc,
                           SharedIndex* scale_indices,
                           float* inv_std,
                           unsigned int inv_std_num_alloc,
                           float* loss,
                           unsigned int loss_num_alloc,
                           float* out_res,
                           unsigned int out_res_num_alloc,
                           float* const out_scale_njtr,
                           unsigned int out_scale_njtr_num_alloc,
                           float* const out_scale_precond_diag,
                           unsigned int out_scale_precond_diag_num_alloc,
                           float* const out_scale_precond_tril,
                           unsigned int out_scale_precond_tril_num_alloc,
                           size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  __shared__ SharedIndex scale_indices_loc[1024];
  scale_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? scale_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28;

  if (global_thread_idx < problem_size) {
    r0 = 9.99999999999999955e-07;
    ReadIdx3<1024, float, float, float4>(
        loss, 0 * loss_num_alloc, global_thread_idx, r1, r2, r3);
    r4 = 0.00000000000000000e+00;
    r3 = fmaxf(r3, r4);
    r5 = sqrtf(r3);
  };
  LoadShared<1, float, float>(
      scale, 0 * scale_num_alloc, scale_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>(
        (float*)inout_shared, scale_indices_loc[threadIdx.x].target, r6);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx1<1024, float, float, float>(
        inv_std, 0 * inv_std_num_alloc, global_thread_idx, r7);
    r8 = r6 * r7;
    r9 = r7 * r8;
    r10 = r6 * r9;
    r11 = 5.00000000000000000e-01;
    r2 = fmaxf(r2, r0);
    r12 = r2 * r2;
    r13 = 2.00000000000000000e+00;
    r14 = r13 * r2;
    r15 = fmaxf(r0, r10);
    r16 = sqrtf(r15);
    r17 = -1.00000000000000000e+00;
    r14 = fmaf(r17, r12, r16 * r14);
    r14 = r10 <= r12 ? r10 : r14;
    r16 = 2.50000000000000000e+00;
    r18 = 1.00000000000000000e+00;
    r19 = 1.0 / r12;
    r20 = r6 * r19;
    r20 = fmaf(r9, r20, r18);
    r21 = logf(r20);
    r21 = r21 * r12;
    r14 = r1 < r16 ? r21 : r14;
    r21 = 1.50000000000000000e+00;
    r22 = sqrtf(r20);
    r22 = r17 + r22;
    r22 = r13 * r22;
    r22 = r22 * r12;
    r14 = r1 < r21 ? r22 : r14;
    r14 = r1 < r11 ? r10 : r14;
    r22 = fmaxf(r4, r14);
    r23 = 1.0 / r15;
    r23 = r3 * r23;
    r24 = r22 * r23;
    r25 = sqrtf(r24);
    r25 = r10 <= r0 ? r5 : r25;
    r5 = r25 * r8;
    WriteIdx1<1024, float, float, float>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r5);
    r5 = r17 * r25;
    r26 = r3 * r17;
    r27 = -9.99999999999999955e-07;
    r27 = r27 + r10;
    r27 = copysign(1.0, r27);
    r27 = r18 + r27;
    r28 = r15 * r15;
    r28 = 1.0 / r28;
    r26 = r26 * r22;
    r26 = r26 * r27;
    r26 = r26 * r28;
    r28 = r6 * r13;
    r22 = r7 * r7;
    r28 = r28 * r22;
    r27 = r2 * r27;
    r15 = rsqrtf(r15);
    r27 = r27 * r15;
    r27 = r27 * r9;
    r27 = r10 <= r12 ? r28 : r27;
    r15 = 1.0 / r20;
    r15 = r15 * r28;
    r27 = r1 < r16 ? r15 : r27;
    r20 = rsqrtf(r20);
    r20 = r20 * r28;
    r27 = r1 < r21 ? r20 : r27;
    r27 = r1 < r11 ? r28 : r27;
    r28 = r11 * r27;
    r14 = copysign(1.0, r14);
    r14 = r18 + r14;
    r23 = r14 * r23;
    r28 = fmaf(r23, r28, r9 * r26);
    r28 = r11 * r28;
    r24 = rsqrtf(r24);
    r28 = r28 * r24;
    r28 = r10 <= r0 ? r4 : r28;
    r28 = fmaf(r28, r8, r7 * r25);
    r25 = 2.50000000000000000e-01;
    r12 = r10 <= r12 ? r4 : r4;
    r12 = r1 < r16 ? r4 : r12;
    r12 = r1 < r21 ? r4 : r12;
    r12 = r1 < r11 ? r4 : r12;
    r12 = r25 * r12;
    r12 = r12 * r24;
    r12 = r12 * r23;
    r12 = r10 <= r0 ? r4 : r12;
    r4 = r17 * r12;
    r28 = fmaf(r8, r4, r28);
    r5 = r5 * r28;
    r5 = r5 * r8;
    WriteSum1<float, float>((float*)inout_shared, r5);
  };
  FlushSumShared<1, float>(out_scale_njtr,
                           0 * out_scale_njtr_num_alloc,
                           scale_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r28 = r28 * r28;
    WriteSum1<float, float>((float*)inout_shared, r28);
  };
  FlushSumShared<1, float>(out_scale_precond_diag,
                           0 * out_scale_precond_diag_num_alloc,
                           scale_indices_loc,
                           (float*)inout_shared);
}

void ScalePriorResJac(float* scale,
                      unsigned int scale_num_alloc,
                      SharedIndex* scale_indices,
                      float* inv_std,
                      unsigned int inv_std_num_alloc,
                      float* loss,
                      unsigned int loss_num_alloc,
                      float* out_res,
                      unsigned int out_res_num_alloc,
                      float* const out_scale_njtr,
                      unsigned int out_scale_njtr_num_alloc,
                      float* const out_scale_precond_diag,
                      unsigned int out_scale_precond_diag_num_alloc,
                      float* const out_scale_precond_tril,
                      unsigned int out_scale_precond_tril_num_alloc,
                      size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ScalePriorResJacKernel<<<n_blocks, 1024>>>(scale,
                                             scale_num_alloc,
                                             scale_indices,
                                             inv_std,
                                             inv_std_num_alloc,
                                             loss,
                                             loss_num_alloc,
                                             out_res,
                                             out_res_num_alloc,
                                             out_scale_njtr,
                                             out_scale_njtr_num_alloc,
                                             out_scale_precond_diag,
                                             out_scale_precond_diag_num_alloc,
                                             out_scale_precond_tril,
                                             out_scale_precond_tril_num_alloc,
                                             problem_size);
}

}  // namespace caspar