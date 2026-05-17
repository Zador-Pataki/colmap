#include "kernel_scale_prior_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ScalePriorResJacFirstKernel(float* scale,
                                unsigned int scale_num_alloc,
                                SharedIndex* scale_indices,
                                float* inv_std,
                                unsigned int inv_std_num_alloc,
                                float* loss,
                                unsigned int loss_num_alloc,
                                float* out_res,
                                unsigned int out_res_num_alloc,
                                float* const out_rTr,
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

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26;

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
    r7 = r6 * r6;
    ReadIdx1<1024, float, float, float>(
        inv_std, 0 * inv_std_num_alloc, global_thread_idx, r8);
    r9 = r8 * r8;
    r9 = r7 * r9;
    r7 = 5.00000000000000000e-01;
    r2 = fmaxf(r2, r0);
    r10 = r2 * r2;
    r11 = 2.00000000000000000e+00;
    r12 = r11 * r2;
    r13 = fmaxf(r0, r9);
    r14 = sqrtf(r13);
    r15 = -1.00000000000000000e+00;
    r12 = fmaf(r15, r10, r14 * r12);
    r12 = r9 <= r10 ? r9 : r12;
    r14 = 2.50000000000000000e+00;
    r16 = 1.00000000000000000e+00;
    r17 = 1.0 / r10;
    r17 = fmaf(r17, r9, r16);
    r18 = logf(r17);
    r18 = r18 * r10;
    r12 = r1 < r14 ? r18 : r12;
    r18 = 1.50000000000000000e+00;
    r19 = sqrtf(r17);
    r19 = r15 + r19;
    r19 = r11 * r19;
    r19 = r19 * r10;
    r12 = r1 < r18 ? r19 : r12;
    r12 = r1 < r7 ? r9 : r12;
    r19 = fmaxf(r4, r12);
    r20 = 1.0 / r13;
    r20 = r3 * r20;
    r21 = r19 * r20;
    r22 = sqrtf(r21);
    r22 = r9 <= r0 ? r5 : r22;
    r6 = r6 * r8;
    r5 = r22 * r6;
    WriteIdx1<1024, float, float, float>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r5);
    r5 = r22 * r22;
    r5 = r9 * r5;
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r5);
  if (global_thread_idx < problem_size) {
    r5 = r15 * r22;
    r23 = r3 * r15;
    r24 = -9.99999999999999955e-07;
    r24 = r9 + r24;
    r24 = copysign(1.0, r24);
    r24 = r16 + r24;
    r25 = r13 * r13;
    r25 = 1.0 / r25;
    r26 = r8 * r6;
    r23 = r23 * r19;
    r23 = r23 * r24;
    r23 = r23 * r25;
    r25 = r11 * r26;
    r24 = r2 * r24;
    r13 = rsqrtf(r13);
    r24 = r24 * r13;
    r24 = r24 * r26;
    r24 = r9 <= r10 ? r25 : r24;
    r13 = 1.0 / r17;
    r13 = r11 * r13;
    r13 = r13 * r26;
    r24 = r1 < r14 ? r13 : r24;
    r17 = rsqrtf(r17);
    r17 = r11 * r17;
    r17 = r17 * r26;
    r24 = r1 < r18 ? r17 : r24;
    r24 = r1 < r7 ? r25 : r24;
    r25 = r7 * r24;
    r12 = copysign(1.0, r12);
    r12 = r16 + r12;
    r20 = r12 * r20;
    r25 = fmaf(r20, r25, r26 * r23);
    r25 = r7 * r25;
    r21 = rsqrtf(r21);
    r25 = r25 * r21;
    r25 = r9 <= r0 ? r4 : r25;
    r25 = fmaf(r25, r6, r8 * r22);
    r22 = 2.50000000000000000e-01;
    r10 = r9 <= r10 ? r4 : r4;
    r10 = r1 < r14 ? r4 : r10;
    r10 = r1 < r18 ? r4 : r10;
    r10 = r1 < r7 ? r4 : r10;
    r10 = r22 * r10;
    r10 = r10 * r21;
    r10 = r10 * r20;
    r10 = r9 <= r0 ? r4 : r10;
    r4 = r15 * r10;
    r25 = fmaf(r6, r4, r25);
    r5 = r5 * r25;
    r5 = r5 * r6;
    WriteSum1<float, float>((float*)inout_shared, r5);
  };
  FlushSumShared<1, float>(out_scale_njtr,
                           0 * out_scale_njtr_num_alloc,
                           scale_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = r25 * r25;
    WriteSum1<float, float>((float*)inout_shared, r25);
  };
  FlushSumShared<1, float>(out_scale_precond_diag,
                           0 * out_scale_precond_diag_num_alloc,
                           scale_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void ScalePriorResJacFirst(float* scale,
                           unsigned int scale_num_alloc,
                           SharedIndex* scale_indices,
                           float* inv_std,
                           unsigned int inv_std_num_alloc,
                           float* loss,
                           unsigned int loss_num_alloc,
                           float* out_res,
                           unsigned int out_res_num_alloc,
                           float* const out_rTr,
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
  ScalePriorResJacFirstKernel<<<n_blocks, 1024>>>(
      scale,
      scale_num_alloc,
      scale_indices,
      inv_std,
      inv_std_num_alloc,
      loss,
      loss_num_alloc,
      out_res,
      out_res_num_alloc,
      out_rTr,
      out_scale_njtr,
      out_scale_njtr_num_alloc,
      out_scale_precond_diag,
      out_scale_precond_diag_num_alloc,
      out_scale_precond_tril,
      out_scale_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar