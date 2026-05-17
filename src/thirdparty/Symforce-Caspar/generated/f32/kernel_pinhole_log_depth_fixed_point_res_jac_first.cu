#include "kernel_pinhole_log_depth_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeLogDepthFixedPointResJacFirstKernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
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
        float* out_pose_jac,
        unsigned int out_pose_jac_num_alloc,
        float* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        float* const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        float* const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
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

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex scale_indices_loc[1024];
  scale_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? scale_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43;

  if (global_thread_idx < problem_size) {
    r0 = 0.00000000000000000e+00;
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>(
        (float*)inout_shared, pose_indices_loc[threadIdx.x].target, r1, r2, r3);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r2, r1, r4);
  };
  LoadShared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r5,
                       r6,
                       r7,
                       r8);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r9 = r6 * r7;
    r10 = 2.00000000000000000e+00;
    r11 = r5 * r8;
    r11 = fmaf(r10, r11, r10 * r9);
    r3 = fmaf(r1, r11, r3);
    r9 = r5 * r7;
    r12 = -2.00000000000000000e+00;
    r13 = r6 * r12;
    r9 = fmaf(r8, r13, r10 * r9);
    r14 = 1.00000000000000000e+00;
    r15 = fmaf(r6, r13, r14);
    r16 = r5 * r5;
    r15 = fmaf(r12, r16, r15);
    r3 = fmaf(r2, r9, r3);
    r3 = fmaf(r4, r15, r3);
  };
  LoadShared<1, float, float>(
      scale, 0 * scale_num_alloc, scale_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>(
        (float*)inout_shared, scale_indices_loc[threadIdx.x].target, r15);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r17 = -1.00000000000000000e+00;
    ReadIdx1<1024, float, float, float>(
        log_depth, 0 * log_depth_num_alloc, global_thread_idx, r18);
    r18 = fmaf(r18, r17, r15 * r17);
    r15 = 9.99999999999999955e-07;
    r19 = fmaxf(r3, r15);
    r20 = logf(r19);
    r18 = r18 + r20;
    r18 = r0 < r3 ? r18 : r0;
    ReadIdx3<1024, float, float, float4>(
        loss, 0 * loss_num_alloc, global_thread_idx, r20, r21, r22);
    r22 = fmaxf(r22, r0);
    r23 = sqrtf(r22);
    r24 = r18 * r18;
    r25 = fmaxf(r15, r24);
    r26 = 1.0 / r25;
    r27 = 5.00000000000000000e-01;
    r21 = fmaxf(r21, r15);
    r28 = r21 * r21;
    r29 = r17 * r21;
    r30 = r10 * r21;
    r31 = sqrtf(r25);
    r30 = fmaf(r31, r30, r21 * r29);
    r30 = r24 <= r28 ? r24 : r30;
    r29 = 2.50000000000000000e+00;
    r31 = r21 * r21;
    r32 = r18 * r18;
    r33 = 1.0 / r28;
    r32 = fmaf(r33, r32, r14);
    r33 = logf(r32);
    r31 = r31 * r33;
    r30 = r20 < r29 ? r31 : r30;
    r31 = 1.50000000000000000e+00;
    r33 = r10 * r21;
    r34 = sqrtf(r32);
    r34 = r17 + r34;
    r33 = r33 * r21;
    r33 = r33 * r34;
    r30 = r20 < r31 ? r33 : r30;
    r30 = r20 < r27 ? r24 : r30;
    r33 = fmaxf(r0, r30);
    r33 = r22 * r33;
    r34 = r26 * r33;
    r35 = sqrtf(r34);
    r35 = r24 <= r15 ? r23 : r35;
    r23 = r18 * r35;
    WriteIdx1<1024, float, float, float>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r23);
    r23 = r18 * r18;
    r36 = r35 * r35;
    r23 = r23 * r36;
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r23);
  if (global_thread_idx < problem_size) {
    r23 = r5 * r8;
    r13 = fmaf(r7, r13, r12 * r23);
    r23 = r8 * r8;
    r36 = r7 * r7;
    r37 = r23 + r36;
    r38 = r6 * r6;
    r37 = fmaf(r17, r38, r37);
    r37 = fmaf(r17, r16, r37);
    r37 = fmaf(r1, r37, r4 * r13);
    r13 = -9.99999999999999955e-07;
    r38 = r13 + r3;
    r38 = copysign(1.0, r38);
    r38 = r14 + r38;
    r38 = r27 * r38;
    r19 = 1.0 / r19;
    r38 = r38 * r19;
    r37 = r37 * r38;
    r37 = r0 < r3 ? r37 : r0;
    r19 = r10 * r18;
    r39 = r37 * r19;
    r40 = r18 * r21;
    r13 = r13 + r24;
    r13 = copysign(1.0, r13);
    r13 = r14 + r13;
    r41 = rsqrtf(r25);
    r40 = r40 * r13;
    r40 = r40 * r41;
    r41 = r37 * r40;
    r41 = r24 <= r28 ? r39 : r41;
    r42 = 1.0 / r32;
    r43 = r37 * r42;
    r43 = r43 * r19;
    r41 = r20 < r29 ? r43 : r41;
    r32 = rsqrtf(r32);
    r32 = r32 * r19;
    r43 = r37 * r32;
    r41 = r20 < r31 ? r43 : r41;
    r41 = r20 < r27 ? r39 : r41;
    r22 = r22 * r27;
    r30 = copysign(1.0, r30);
    r30 = r14 + r30;
    r22 = r22 * r30;
    r22 = r22 * r26;
    r26 = r17 * r18;
    r25 = r25 * r25;
    r25 = 1.0 / r25;
    r26 = r26 * r13;
    r26 = r26 * r25;
    r26 = r26 * r33;
    r41 = fmaf(r37, r26, r41 * r22);
    r41 = r27 * r41;
    r34 = rsqrtf(r34);
    r41 = r41 * r34;
    r41 = r24 <= r15 ? r0 : r41;
    r37 = fmaf(r35, r37, r18 * r41);
    r41 = r17 * r18;
    r33 = r0 < r3 ? r0 : r0;
    r25 = r33 * r19;
    r13 = r33 * r40;
    r13 = r24 <= r28 ? r25 : r13;
    r30 = r33 * r42;
    r30 = r30 * r19;
    r13 = r20 < r29 ? r30 : r13;
    r30 = r33 * r32;
    r13 = r20 < r31 ? r30 : r13;
    r13 = r20 < r27 ? r25 : r13;
    r13 = fmaf(r13, r22, r33 * r26);
    r13 = r27 * r13;
    r13 = r13 * r34;
    r13 = r24 <= r15 ? r0 : r13;
    r25 = r17 * r35;
    r25 = fmaf(r33, r25, r13 * r41);
    r37 = r37 + r25;
    r16 = fmaf(r6, r6, r16);
    r16 = fmaf(r17, r23, r16);
    r16 = fmaf(r17, r36, r16);
    r16 = fmaf(r2, r16, r4 * r9);
    r16 = r16 * r38;
    r16 = r0 < r3 ? r16 : r0;
    r9 = r16 * r19;
    r4 = r16 * r40;
    r4 = r24 <= r28 ? r9 : r4;
    r36 = r16 * r42;
    r36 = r36 * r19;
    r4 = r20 < r29 ? r36 : r4;
    r36 = r16 * r32;
    r4 = r20 < r31 ? r36 : r4;
    r4 = r20 < r27 ? r9 : r4;
    r4 = fmaf(r16, r26, r4 * r22);
    r4 = r27 * r4;
    r4 = r4 * r34;
    r4 = r24 <= r15 ? r0 : r4;
    r4 = fmaf(r18, r4, r35 * r16);
    r4 = r4 + r25;
    r16 = r5 * r7;
    r9 = r6 * r8;
    r9 = fmaf(r10, r9, r12 * r16);
    r9 = fmaf(r1, r9, r2 * r11);
    r9 = r9 * r38;
    r9 = r0 < r3 ? r9 : r0;
    r1 = r9 * r19;
    r11 = r9 * r40;
    r11 = r24 <= r28 ? r1 : r11;
    r2 = r9 * r42;
    r2 = r2 * r19;
    r11 = r20 < r29 ? r2 : r11;
    r2 = r9 * r32;
    r11 = r20 < r31 ? r2 : r11;
    r11 = r20 < r27 ? r1 : r11;
    r11 = fmaf(r11, r22, r9 * r26);
    r11 = r27 * r11;
    r11 = r11 * r34;
    r11 = r24 <= r15 ? r0 : r11;
    r11 = fmaf(r18, r11, r35 * r9);
    r11 = r11 + r25;
    r38 = r0 < r3 ? r38 : r0;
    r9 = r38 * r19;
    r1 = r38 * r40;
    r1 = r24 <= r28 ? r9 : r1;
    r2 = r38 * r42;
    r2 = r2 * r19;
    r1 = r20 < r29 ? r2 : r1;
    r2 = r38 * r32;
    r1 = r20 < r31 ? r2 : r1;
    r1 = r20 < r27 ? r9 : r1;
    r1 = fmaf(r1, r22, r38 * r26);
    r1 = r27 * r1;
    r1 = r1 * r34;
    r1 = r24 <= r15 ? r0 : r1;
    r38 = fmaf(r35, r38, r18 * r1);
    r38 = r38 + r25;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r37,
                                          r4,
                                          r11,
                                          r38);
    r1 = r17 * r18;
    r1 = r1 * r35;
    r1 = r1 * r37;
    r9 = r17 * r18;
    r9 = r9 * r35;
    r9 = r9 * r4;
    r2 = r17 * r18;
    r2 = r2 * r35;
    r2 = r2 * r11;
    WriteSum4<float, float>((float*)inout_shared, r1, r9, r2, r0);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r2 = r17 * r18;
    r2 = r2 * r35;
    r2 = r2 * r38;
    WriteSum2<float, float>((float*)inout_shared, r0, r2);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r2 = r37 * r37;
    r9 = r4 * r4;
    r1 = r11 * r11;
    WriteSum4<float, float>((float*)inout_shared, r2, r9, r1, r0);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r38 * r38;
    WriteSum2<float, float>((float*)inout_shared, r0, r1);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r37 * r4;
    r9 = r37 * r11;
    WriteSum4<float, float>((float*)inout_shared, r1, r9, r0, r0);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = r37 * r38;
    r9 = r4 * r11;
    WriteSum4<float, float>((float*)inout_shared, r37, r9, r0, r0);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r4 = r4 * r38;
    r38 = r11 * r38;
    WriteSum4<float, float>((float*)inout_shared, r4, r0, r0, r38);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r3 = r0 < r3 ? r17 : r0;
    r38 = r3 * r19;
    r40 = r3 * r40;
    r40 = r24 <= r28 ? r38 : r40;
    r42 = r3 * r42;
    r42 = r42 * r19;
    r40 = r20 < r29 ? r42 : r40;
    r32 = r3 * r32;
    r40 = r20 < r31 ? r32 : r40;
    r40 = r20 < r27 ? r38 : r40;
    r22 = fmaf(r40, r22, r3 * r26);
    r22 = r27 * r22;
    r22 = r22 * r34;
    r22 = r24 <= r15 ? r0 : r22;
    r3 = fmaf(r35, r3, r18 * r22);
    r3 = r3 + r25;
    WriteIdx1<1024, float, float, float>(
        out_scale_jac, 0 * out_scale_jac_num_alloc, global_thread_idx, r3);
    r25 = r17 * r18;
    r25 = r25 * r35;
    r25 = r25 * r3;
    WriteSum1<float, float>((float*)inout_shared, r25);
  };
  FlushSumShared<1, float>(out_scale_njtr,
                           0 * out_scale_njtr_num_alloc,
                           scale_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r3 = r3 * r3;
    WriteSum1<float, float>((float*)inout_shared, r3);
  };
  FlushSumShared<1, float>(out_scale_precond_diag,
                           0 * out_scale_precond_diag_num_alloc,
                           scale_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void PinholeLogDepthFixedPointResJacFirst(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
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
    float* out_pose_jac,
    unsigned int out_pose_jac_num_alloc,
    float* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    float* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
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
  PinholeLogDepthFixedPointResJacFirstKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
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
      out_pose_jac,
      out_pose_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_pose_precond_diag,
      out_pose_precond_diag_num_alloc,
      out_pose_precond_tril,
      out_pose_precond_tril_num_alloc,
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