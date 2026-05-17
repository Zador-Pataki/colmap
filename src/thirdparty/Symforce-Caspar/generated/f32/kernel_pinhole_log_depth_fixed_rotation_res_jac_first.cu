#include "kernel_pinhole_log_depth_fixed_rotation_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeLogDepthFixedRotationResJacFirstKernel(
        float* rotation,
        unsigned int rotation_num_alloc,
        float* translation,
        unsigned int translation_num_alloc,
        SharedIndex* translation_indices,
        float* scale,
        unsigned int scale_num_alloc,
        SharedIndex* scale_indices,
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* log_depth,
        unsigned int log_depth_num_alloc,
        float* loss,
        unsigned int loss_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_rTr,
        float* out_translation_jac,
        unsigned int out_translation_jac_num_alloc,
        float* const out_translation_njtr,
        unsigned int out_translation_njtr_num_alloc,
        float* const out_translation_precond_diag,
        unsigned int out_translation_precond_diag_num_alloc,
        float* const out_translation_precond_tril,
        unsigned int out_translation_precond_tril_num_alloc,
        float* out_scale_jac,
        unsigned int out_scale_jac_num_alloc,
        float* const out_scale_njtr,
        unsigned int out_scale_njtr_num_alloc,
        float* const out_scale_precond_diag,
        unsigned int out_scale_precond_diag_num_alloc,
        float* const out_scale_precond_tril,
        unsigned int out_scale_precond_tril_num_alloc,
        float* out_point_jac,
        unsigned int out_point_jac_num_alloc,
        float* const out_point_njtr,
        unsigned int out_point_njtr_num_alloc,
        float* const out_point_precond_diag,
        unsigned int out_point_precond_diag_num_alloc,
        float* const out_point_precond_tril,
        unsigned int out_point_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex translation_indices_loc[1024];
  translation_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? translation_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex scale_indices_loc[1024];
  scale_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? scale_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37;

  if (global_thread_idx < problem_size) {
    r0 = 0.00000000000000000e+00;
  };
  LoadShared<3, float, float>(translation,
                              0 * translation_num_alloc,
                              translation_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       translation_indices_loc[threadIdx.x].target,
                       r1,
                       r2,
                       r3);
  };
  __syncthreads();
  LoadShared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_indices_loc[threadIdx.x].target,
                       r2,
                       r1,
                       r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r5 = 1.00000000000000000e+00;
    r6 = -2.00000000000000000e+00;
    ReadIdx4<1024, float, float, float4>(
        rotation, 0 * rotation_num_alloc, global_thread_idx, r7, r8, r9, r10);
    r11 = r8 * r8;
    r11 = fmaf(r6, r11, r5);
    r12 = r7 * r7;
    r11 = fmaf(r6, r12, r11);
    r4 = fmaf(r4, r11, r3);
    r3 = r7 * r9;
    r12 = 2.00000000000000000e+00;
    r13 = r8 * r10;
    r13 = fmaf(r6, r13, r12 * r3);
    r3 = r7 * r10;
    r6 = r8 * r9;
    r6 = fmaf(r12, r6, r12 * r3);
    r4 = fmaf(r2, r13, r4);
    r4 = fmaf(r1, r6, r4);
    r1 = 9.99999999999999955e-07;
    r2 = fmaxf(r1, r4);
    r3 = logf(r2);
  };
  LoadShared<1, float, float>(
      scale, 0 * scale_num_alloc, scale_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>(
        (float*)inout_shared, scale_indices_loc[threadIdx.x].target, r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r15 = -1.00000000000000000e+00;
    r14 = fmaf(r14, r15, r3);
    ReadIdx1<1024, float, float, float>(
        log_depth, 0 * log_depth_num_alloc, global_thread_idx, r3);
    r14 = fmaf(r3, r15, r14);
    r14 = r0 < r4 ? r14 : r0;
    ReadIdx3<1024, float, float, float4>(
        loss, 0 * loss_num_alloc, global_thread_idx, r3, r16, r17);
    r17 = fmaxf(r17, r0);
    r18 = sqrtf(r17);
    r19 = r14 * r14;
    r20 = fmaxf(r1, r19);
    r21 = 1.0 / r20;
    r22 = 5.00000000000000000e-01;
    r16 = fmaxf(r16, r1);
    r23 = r16 * r16;
    r24 = r15 * r16;
    r25 = r12 * r16;
    r26 = sqrtf(r20);
    r25 = fmaf(r26, r25, r16 * r24);
    r25 = r19 <= r23 ? r19 : r25;
    r24 = 2.50000000000000000e+00;
    r26 = r16 * r16;
    r27 = r14 * r14;
    r28 = 1.0 / r23;
    r27 = fmaf(r28, r27, r5);
    r28 = logf(r27);
    r26 = r26 * r28;
    r25 = r3 < r24 ? r26 : r25;
    r26 = 1.50000000000000000e+00;
    r28 = r12 * r16;
    r29 = sqrtf(r27);
    r29 = r15 + r29;
    r28 = r28 * r16;
    r28 = r28 * r29;
    r25 = r3 < r26 ? r28 : r25;
    r25 = r3 < r22 ? r19 : r25;
    r28 = fmaxf(r0, r25);
    r28 = r17 * r28;
    r29 = r21 * r28;
    r30 = sqrtf(r29);
    r30 = r19 <= r1 ? r18 : r30;
    r18 = r14 * r30;
    WriteIdx1<1024, float, float, float>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r18);
    r18 = r14 * r14;
    r31 = r30 * r30;
    r18 = r18 * r31;
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r18);
  if (global_thread_idx < problem_size) {
    r2 = 1.0 / r2;
    r2 = r22 * r2;
    r18 = -9.99999999999999955e-07;
    r31 = r18 + r4;
    r31 = copysign(1.0, r31);
    r31 = r5 + r31;
    r2 = r2 * r31;
    r31 = r0 < r4 ? r2 : r0;
    r32 = r12 * r14;
    r33 = r31 * r32;
    r34 = r14 * r16;
    r18 = r18 + r19;
    r18 = copysign(1.0, r18);
    r18 = r5 + r18;
    r35 = rsqrtf(r20);
    r34 = r34 * r18;
    r34 = r34 * r35;
    r35 = r31 * r34;
    r35 = r19 <= r23 ? r33 : r35;
    r36 = 1.0 / r27;
    r37 = r31 * r36;
    r37 = r37 * r32;
    r35 = r3 < r24 ? r37 : r35;
    r27 = rsqrtf(r27);
    r27 = r27 * r32;
    r37 = r31 * r27;
    r35 = r3 < r26 ? r37 : r35;
    r35 = r3 < r22 ? r33 : r35;
    r17 = r17 * r22;
    r25 = copysign(1.0, r25);
    r25 = r5 + r25;
    r17 = r17 * r25;
    r17 = r17 * r21;
    r21 = r15 * r14;
    r20 = r20 * r20;
    r20 = 1.0 / r20;
    r21 = r21 * r18;
    r21 = r21 * r20;
    r21 = r21 * r28;
    r35 = fmaf(r31, r21, r35 * r17);
    r35 = r22 * r35;
    r29 = rsqrtf(r29);
    r35 = r35 * r29;
    r35 = r19 <= r1 ? r0 : r35;
    r35 = fmaf(r14, r35, r30 * r31);
    r31 = r15 * r14;
    r28 = r0 < r4 ? r0 : r0;
    r20 = r28 * r32;
    r18 = r28 * r34;
    r18 = r19 <= r23 ? r20 : r18;
    r25 = r28 * r36;
    r25 = r25 * r32;
    r18 = r3 < r24 ? r25 : r18;
    r25 = r28 * r27;
    r18 = r3 < r26 ? r25 : r18;
    r18 = r3 < r22 ? r20 : r18;
    r18 = fmaf(r28, r21, r18 * r17);
    r18 = r22 * r18;
    r18 = r18 * r29;
    r18 = r19 <= r1 ? r0 : r18;
    r20 = r15 * r30;
    r20 = fmaf(r28, r20, r18 * r31);
    r35 = r35 + r20;
    WriteIdx1<1024, float, float, float>(out_translation_jac,
                                         0 * out_translation_jac_num_alloc,
                                         global_thread_idx,
                                         r35);
    r31 = r15 * r14;
    r31 = r31 * r30;
    r31 = r31 * r35;
    WriteSum3<float, float>((float*)inout_shared, r0, r0, r31);
  };
  FlushSumShared<3, float>(out_translation_njtr,
                           0 * out_translation_njtr_num_alloc,
                           translation_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = r35 * r35;
    WriteSum3<float, float>((float*)inout_shared, r0, r0, r35);
  };
  FlushSumShared<3, float>(out_translation_precond_diag,
                           0 * out_translation_precond_diag_num_alloc,
                           translation_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = r0 < r4 ? r15 : r0;
    r31 = r35 * r32;
    r28 = r35 * r34;
    r28 = r19 <= r23 ? r31 : r28;
    r18 = r35 * r36;
    r18 = r18 * r32;
    r28 = r3 < r24 ? r18 : r28;
    r18 = r35 * r27;
    r28 = r3 < r26 ? r18 : r28;
    r28 = r3 < r22 ? r31 : r28;
    r28 = fmaf(r35, r21, r28 * r17);
    r28 = r22 * r28;
    r28 = r28 * r29;
    r28 = r19 <= r1 ? r0 : r28;
    r35 = fmaf(r30, r35, r14 * r28);
    r35 = r35 + r20;
    WriteIdx1<1024, float, float, float>(
        out_scale_jac, 0 * out_scale_jac_num_alloc, global_thread_idx, r35);
    r28 = r15 * r14;
    r28 = r28 * r30;
    r28 = r28 * r35;
    WriteSum1<float, float>((float*)inout_shared, r28);
  };
  FlushSumShared<1, float>(out_scale_njtr,
                           0 * out_scale_njtr_num_alloc,
                           scale_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = r35 * r35;
    WriteSum1<float, float>((float*)inout_shared, r35);
  };
  FlushSumShared<1, float>(out_scale_precond_diag,
                           0 * out_scale_precond_diag_num_alloc,
                           scale_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = r13 * r2;
    r13 = r0 < r4 ? r13 : r0;
    r35 = r13 * r32;
    r28 = r13 * r34;
    r28 = r19 <= r23 ? r35 : r28;
    r31 = r13 * r36;
    r31 = r31 * r32;
    r28 = r3 < r24 ? r31 : r28;
    r31 = r13 * r27;
    r28 = r3 < r26 ? r31 : r28;
    r28 = r3 < r22 ? r35 : r28;
    r28 = fmaf(r13, r21, r28 * r17);
    r28 = r22 * r28;
    r28 = r28 * r29;
    r28 = r19 <= r1 ? r0 : r28;
    r13 = fmaf(r30, r13, r14 * r28);
    r13 = r13 + r20;
    r6 = r6 * r2;
    r6 = r0 < r4 ? r6 : r0;
    r28 = r6 * r32;
    r35 = r6 * r34;
    r35 = r19 <= r23 ? r28 : r35;
    r31 = r6 * r36;
    r31 = r31 * r32;
    r35 = r3 < r24 ? r31 : r35;
    r31 = r6 * r27;
    r35 = r3 < r26 ? r31 : r35;
    r35 = r3 < r22 ? r28 : r35;
    r35 = fmaf(r6, r21, r35 * r17);
    r35 = r22 * r35;
    r35 = r35 * r29;
    r35 = r19 <= r1 ? r0 : r35;
    r35 = fmaf(r14, r35, r30 * r6);
    r35 = r35 + r20;
    r2 = r11 * r2;
    r2 = r0 < r4 ? r2 : r0;
    r4 = r2 * r32;
    r34 = r2 * r34;
    r34 = r19 <= r23 ? r4 : r34;
    r36 = r2 * r36;
    r36 = r36 * r32;
    r34 = r3 < r24 ? r36 : r34;
    r27 = r2 * r27;
    r34 = r3 < r26 ? r27 : r34;
    r34 = r3 < r22 ? r4 : r34;
    r21 = fmaf(r2, r21, r34 * r17);
    r21 = r22 * r21;
    r21 = r21 * r29;
    r21 = r19 <= r1 ? r0 : r21;
    r21 = fmaf(r14, r21, r30 * r2);
    r21 = r21 + r20;
    WriteIdx3<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r13,
                                          r35,
                                          r21);
    r20 = r15 * r14;
    r20 = r20 * r30;
    r20 = r20 * r13;
    r2 = r15 * r14;
    r2 = r2 * r30;
    r2 = r2 * r35;
    r0 = r15 * r14;
    r0 = r0 * r30;
    r0 = r0 * r21;
    WriteSum3<float, float>((float*)inout_shared, r20, r2, r0);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = r13 * r13;
    r2 = r35 * r35;
    r20 = r21 * r21;
    WriteSum3<float, float>((float*)inout_shared, r0, r2, r20);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r20 = r13 * r35;
    r13 = r13 * r21;
    r21 = r35 * r21;
    WriteSum3<float, float>((float*)inout_shared, r20, r13, r21);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void PinholeLogDepthFixedRotationResJacFirst(
    float* rotation,
    unsigned int rotation_num_alloc,
    float* translation,
    unsigned int translation_num_alloc,
    SharedIndex* translation_indices,
    float* scale,
    unsigned int scale_num_alloc,
    SharedIndex* scale_indices,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* log_depth,
    unsigned int log_depth_num_alloc,
    float* loss,
    unsigned int loss_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
    float* out_translation_jac,
    unsigned int out_translation_jac_num_alloc,
    float* const out_translation_njtr,
    unsigned int out_translation_njtr_num_alloc,
    float* const out_translation_precond_diag,
    unsigned int out_translation_precond_diag_num_alloc,
    float* const out_translation_precond_tril,
    unsigned int out_translation_precond_tril_num_alloc,
    float* out_scale_jac,
    unsigned int out_scale_jac_num_alloc,
    float* const out_scale_njtr,
    unsigned int out_scale_njtr_num_alloc,
    float* const out_scale_precond_diag,
    unsigned int out_scale_precond_diag_num_alloc,
    float* const out_scale_precond_tril,
    unsigned int out_scale_precond_tril_num_alloc,
    float* out_point_jac,
    unsigned int out_point_jac_num_alloc,
    float* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    float* const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    float* const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeLogDepthFixedRotationResJacFirstKernel<<<n_blocks, 1024>>>(
      rotation,
      rotation_num_alloc,
      translation,
      translation_num_alloc,
      translation_indices,
      scale,
      scale_num_alloc,
      scale_indices,
      point,
      point_num_alloc,
      point_indices,
      log_depth,
      log_depth_num_alloc,
      loss,
      loss_num_alloc,
      out_res,
      out_res_num_alloc,
      out_rTr,
      out_translation_jac,
      out_translation_jac_num_alloc,
      out_translation_njtr,
      out_translation_njtr_num_alloc,
      out_translation_precond_diag,
      out_translation_precond_diag_num_alloc,
      out_translation_precond_tril,
      out_translation_precond_tril_num_alloc,
      out_scale_jac,
      out_scale_jac_num_alloc,
      out_scale_njtr,
      out_scale_njtr_num_alloc,
      out_scale_precond_diag,
      out_scale_precond_diag_num_alloc,
      out_scale_precond_tril,
      out_scale_precond_tril_num_alloc,
      out_point_jac,
      out_point_jac_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      out_point_precond_diag,
      out_point_precond_diag_num_alloc,
      out_point_precond_tril,
      out_point_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar