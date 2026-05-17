#include "kernel_pinhole_log_depth_fixed_rotation_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeLogDepthFixedRotationFixedPointResJacFirstKernel(
        float* rotation,
        unsigned int rotation_num_alloc,
        float* translation,
        unsigned int translation_num_alloc,
        SharedIndex* translation_indices,
        float* scale,
        unsigned int scale_num_alloc,
        SharedIndex* scale_indices,
        float* log_depth,
        unsigned int log_depth_num_alloc,
        float* loss,
        unsigned int loss_num_alloc,
        float* point,
        unsigned int point_num_alloc,
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

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32;

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
  if (global_thread_idx < problem_size) {
    ReadIdx3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r2, r1, r4);
    r5 = 1.00000000000000000e+00;
    r6 = -2.00000000000000000e+00;
    ReadIdx4<1024, float, float, float4>(
        rotation, 0 * rotation_num_alloc, global_thread_idx, r7, r8, r9, r10);
    r11 = r8 * r8;
    r11 = fmaf(r6, r11, r5);
    r12 = r7 * r7;
    r11 = fmaf(r6, r12, r11);
    r11 = fmaf(r4, r11, r3);
    r4 = r7 * r9;
    r3 = 2.00000000000000000e+00;
    r12 = r8 * r10;
    r12 = fmaf(r6, r12, r3 * r4);
    r4 = r7 * r10;
    r6 = r8 * r9;
    r6 = fmaf(r3, r6, r3 * r4);
    r11 = fmaf(r2, r12, r11);
    r11 = fmaf(r1, r6, r11);
    r6 = 9.99999999999999955e-07;
    r1 = fmaxf(r6, r11);
    r12 = logf(r1);
  };
  LoadShared<1, float, float>(
      scale, 0 * scale_num_alloc, scale_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>(
        (float*)inout_shared, scale_indices_loc[threadIdx.x].target, r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r4 = -1.00000000000000000e+00;
    r2 = fmaf(r2, r4, r12);
    ReadIdx1<1024, float, float, float>(
        log_depth, 0 * log_depth_num_alloc, global_thread_idx, r12);
    r2 = fmaf(r12, r4, r2);
    r2 = r0 < r11 ? r2 : r0;
    ReadIdx3<1024, float, float, float4>(
        loss, 0 * loss_num_alloc, global_thread_idx, r12, r13, r14);
    r14 = fmaxf(r14, r0);
    r15 = sqrtf(r14);
    r16 = r2 * r2;
    r17 = fmaxf(r6, r16);
    r18 = 1.0 / r17;
    r19 = 5.00000000000000000e-01;
    r13 = fmaxf(r13, r6);
    r20 = r13 * r13;
    r21 = r3 * r13;
    r22 = sqrtf(r17);
    r21 = fmaf(r22, r21, r4 * r20);
    r21 = r16 <= r20 ? r16 : r21;
    r22 = 2.50000000000000000e+00;
    r23 = r2 * r2;
    r24 = 1.0 / r20;
    r23 = fmaf(r24, r23, r5);
    r24 = logf(r23);
    r24 = r24 * r20;
    r21 = r12 < r22 ? r24 : r21;
    r24 = 1.50000000000000000e+00;
    r25 = sqrtf(r23);
    r25 = r4 + r25;
    r25 = r3 * r25;
    r25 = r25 * r20;
    r21 = r12 < r24 ? r25 : r21;
    r21 = r12 < r19 ? r16 : r21;
    r25 = fmaxf(r0, r21);
    r25 = r14 * r25;
    r26 = r18 * r25;
    r27 = sqrtf(r26);
    r27 = r16 <= r6 ? r15 : r27;
    r15 = r2 * r27;
    WriteIdx1<1024, float, float, float>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r15);
    r15 = r2 * r2;
    r28 = r27 * r27;
    r15 = r15 * r28;
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r15);
  if (global_thread_idx < problem_size) {
    r15 = -9.99999999999999955e-07;
    r28 = r15 + r11;
    r28 = copysign(1.0, r28);
    r28 = r5 + r28;
    r28 = r19 * r28;
    r1 = 1.0 / r1;
    r28 = r28 * r1;
    r28 = r0 < r11 ? r28 : r0;
    r1 = r3 * r2;
    r1 = r1 * r28;
    r29 = r2 * r13;
    r15 = r15 + r16;
    r15 = copysign(1.0, r15);
    r15 = r5 + r15;
    r30 = rsqrtf(r17);
    r29 = r29 * r28;
    r29 = r29 * r15;
    r29 = r29 * r30;
    r29 = r16 <= r20 ? r1 : r29;
    r31 = 1.0 / r23;
    r32 = r31 * r1;
    r29 = r12 < r22 ? r32 : r29;
    r23 = rsqrtf(r23);
    r32 = r23 * r1;
    r29 = r12 < r24 ? r32 : r29;
    r29 = r12 < r19 ? r1 : r29;
    r14 = r14 * r19;
    r21 = copysign(1.0, r21);
    r21 = r5 + r21;
    r14 = r14 * r21;
    r14 = r14 * r18;
    r18 = r4 * r2;
    r17 = r17 * r17;
    r17 = 1.0 / r17;
    r18 = r18 * r15;
    r18 = r18 * r17;
    r18 = r18 * r25;
    r29 = fmaf(r28, r18, r29 * r14);
    r29 = r19 * r29;
    r26 = rsqrtf(r26);
    r29 = r29 * r26;
    r29 = r16 <= r6 ? r0 : r29;
    r29 = fmaf(r2, r29, r27 * r28);
    r28 = r4 * r2;
    r25 = r0 < r11 ? r0 : r0;
    r17 = r3 * r2;
    r21 = r25 * r17;
    r5 = r2 * r13;
    r5 = r5 * r25;
    r5 = r5 * r15;
    r5 = r5 * r30;
    r5 = r16 <= r20 ? r21 : r5;
    r1 = r25 * r31;
    r1 = r1 * r17;
    r5 = r12 < r22 ? r1 : r5;
    r1 = r25 * r23;
    r1 = r1 * r17;
    r5 = r12 < r24 ? r1 : r5;
    r5 = r12 < r19 ? r21 : r5;
    r5 = fmaf(r25, r18, r5 * r14);
    r5 = r19 * r5;
    r5 = r5 * r26;
    r5 = r16 <= r6 ? r0 : r5;
    r21 = r4 * r27;
    r21 = fmaf(r25, r21, r5 * r28);
    r29 = r29 + r21;
    WriteIdx1<1024, float, float, float>(out_translation_jac,
                                         0 * out_translation_jac_num_alloc,
                                         global_thread_idx,
                                         r29);
    r28 = r4 * r2;
    r28 = r28 * r27;
    r28 = r28 * r29;
    WriteSum3<float, float>((float*)inout_shared, r0, r0, r28);
  };
  FlushSumShared<3, float>(out_translation_njtr,
                           0 * out_translation_njtr_num_alloc,
                           translation_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r29 = r29 * r29;
    WriteSum3<float, float>((float*)inout_shared, r0, r0, r29);
  };
  FlushSumShared<3, float>(out_translation_precond_diag,
                           0 * out_translation_precond_diag_num_alloc,
                           translation_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = r0 < r11 ? r4 : r0;
    r29 = r11 * r17;
    r28 = r2 * r13;
    r28 = r28 * r15;
    r28 = r28 * r11;
    r28 = r28 * r30;
    r28 = r16 <= r20 ? r29 : r28;
    r31 = r11 * r31;
    r31 = r31 * r17;
    r28 = r12 < r22 ? r31 : r28;
    r23 = r11 * r23;
    r23 = r23 * r17;
    r28 = r12 < r24 ? r23 : r28;
    r28 = r12 < r19 ? r29 : r28;
    r18 = fmaf(r11, r18, r28 * r14);
    r18 = r19 * r18;
    r18 = r18 * r26;
    r18 = r16 <= r6 ? r0 : r18;
    r11 = fmaf(r27, r11, r2 * r18);
    r11 = r11 + r21;
    WriteIdx1<1024, float, float, float>(
        out_scale_jac, 0 * out_scale_jac_num_alloc, global_thread_idx, r11);
    r21 = r4 * r2;
    r21 = r21 * r27;
    r21 = r21 * r11;
    WriteSum1<float, float>((float*)inout_shared, r21);
  };
  FlushSumShared<1, float>(out_scale_njtr,
                           0 * out_scale_njtr_num_alloc,
                           scale_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = r11 * r11;
    WriteSum1<float, float>((float*)inout_shared, r11);
  };
  FlushSumShared<1, float>(out_scale_precond_diag,
                           0 * out_scale_precond_diag_num_alloc,
                           scale_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void PinholeLogDepthFixedRotationFixedPointResJacFirst(
    float* rotation,
    unsigned int rotation_num_alloc,
    float* translation,
    unsigned int translation_num_alloc,
    SharedIndex* translation_indices,
    float* scale,
    unsigned int scale_num_alloc,
    SharedIndex* scale_indices,
    float* log_depth,
    unsigned int log_depth_num_alloc,
    float* loss,
    unsigned int loss_num_alloc,
    float* point,
    unsigned int point_num_alloc,
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
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeLogDepthFixedRotationFixedPointResJacFirstKernel<<<n_blocks, 1024>>>(
      rotation,
      rotation_num_alloc,
      translation,
      translation_num_alloc,
      translation_indices,
      scale,
      scale_num_alloc,
      scale_indices,
      log_depth,
      log_depth_num_alloc,
      loss,
      loss_num_alloc,
      point,
      point_num_alloc,
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
      problem_size);
}

}  // namespace caspar