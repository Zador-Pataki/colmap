#include "kernel_pinhole_log_depth_fixed_scale_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeLogDepthFixedScaleFixedPointResJacFirstKernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* log_depth,
        unsigned int log_depth_num_alloc,
        float* loss,
        unsigned int loss_num_alloc,
        float* scale,
        unsigned int scale_num_alloc,
        float* point,
        unsigned int point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_rTr,
        float* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        float* const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        float* const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44;

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
    ReadIdx1<1024, float, float, float>(
        scale, 0 * scale_num_alloc, global_thread_idx, r15);
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
    r23 = r17 * r18;
    r36 = r5 * r8;
    r13 = fmaf(r7, r13, r12 * r36);
    r36 = r8 * r8;
    r37 = r7 * r7;
    r38 = r36 + r37;
    r39 = r6 * r6;
    r38 = fmaf(r17, r39, r38);
    r38 = fmaf(r17, r16, r38);
    r38 = fmaf(r1, r38, r4 * r13);
    r13 = -9.99999999999999955e-07;
    r39 = r13 + r3;
    r39 = copysign(1.0, r39);
    r39 = r14 + r39;
    r39 = r27 * r39;
    r19 = 1.0 / r19;
    r39 = r39 * r19;
    r38 = r38 * r39;
    r38 = r0 < r3 ? r38 : r0;
    r19 = r10 * r18;
    r40 = r38 * r19;
    r41 = r18 * r21;
    r13 = r13 + r24;
    r13 = copysign(1.0, r13);
    r13 = r14 + r13;
    r42 = rsqrtf(r25);
    r41 = r41 * r13;
    r41 = r41 * r42;
    r42 = r38 * r41;
    r42 = r24 <= r28 ? r40 : r42;
    r43 = 1.0 / r32;
    r44 = r38 * r43;
    r44 = r44 * r19;
    r42 = r20 < r29 ? r44 : r42;
    r32 = rsqrtf(r32);
    r32 = r32 * r19;
    r44 = r38 * r32;
    r42 = r20 < r31 ? r44 : r42;
    r42 = r20 < r27 ? r40 : r42;
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
    r42 = fmaf(r38, r26, r42 * r22);
    r42 = r27 * r42;
    r34 = rsqrtf(r34);
    r42 = r42 * r34;
    r42 = r24 <= r15 ? r0 : r42;
    r38 = fmaf(r35, r38, r18 * r42);
    r42 = r17 * r18;
    r33 = r0 < r3 ? r0 : r0;
    r25 = r33 * r19;
    r13 = r33 * r41;
    r13 = r24 <= r28 ? r25 : r13;
    r30 = r33 * r43;
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
    r25 = fmaf(r33, r25, r13 * r42);
    r38 = r38 + r25;
    r23 = r23 * r35;
    r23 = r23 * r38;
    r42 = r17 * r18;
    r16 = fmaf(r6, r6, r16);
    r16 = fmaf(r17, r36, r16);
    r16 = fmaf(r17, r37, r16);
    r16 = fmaf(r2, r16, r4 * r9);
    r16 = r16 * r39;
    r16 = r0 < r3 ? r16 : r0;
    r9 = r16 * r19;
    r4 = r16 * r41;
    r4 = r24 <= r28 ? r9 : r4;
    r37 = r16 * r43;
    r37 = r37 * r19;
    r4 = r20 < r29 ? r37 : r4;
    r37 = r16 * r32;
    r4 = r20 < r31 ? r37 : r4;
    r4 = r20 < r27 ? r9 : r4;
    r4 = fmaf(r16, r26, r4 * r22);
    r4 = r27 * r4;
    r4 = r4 * r34;
    r4 = r24 <= r15 ? r0 : r4;
    r4 = fmaf(r18, r4, r35 * r16);
    r4 = r4 + r25;
    r42 = r42 * r35;
    r42 = r42 * r4;
    r16 = r17 * r18;
    r9 = r5 * r7;
    r37 = r6 * r8;
    r37 = fmaf(r10, r37, r12 * r9);
    r37 = fmaf(r1, r37, r2 * r11);
    r37 = r37 * r39;
    r37 = r0 < r3 ? r37 : r0;
    r1 = r37 * r19;
    r11 = r37 * r41;
    r11 = r24 <= r28 ? r1 : r11;
    r2 = r37 * r43;
    r2 = r2 * r19;
    r11 = r20 < r29 ? r2 : r11;
    r2 = r37 * r32;
    r11 = r20 < r31 ? r2 : r11;
    r11 = r20 < r27 ? r1 : r11;
    r11 = fmaf(r11, r22, r37 * r26);
    r11 = r27 * r11;
    r11 = r11 * r34;
    r11 = r24 <= r15 ? r0 : r11;
    r11 = fmaf(r18, r11, r35 * r37);
    r11 = r11 + r25;
    r16 = r16 * r35;
    r16 = r16 * r11;
    WriteSum4<float, float>((float*)inout_shared, r23, r42, r16, r0);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r16 = r17 * r18;
    r39 = r0 < r3 ? r39 : r0;
    r3 = r39 * r19;
    r41 = r39 * r41;
    r41 = r24 <= r28 ? r3 : r41;
    r43 = r39 * r43;
    r43 = r43 * r19;
    r41 = r20 < r29 ? r43 : r41;
    r32 = r39 * r32;
    r41 = r20 < r31 ? r32 : r41;
    r41 = r20 < r27 ? r3 : r41;
    r22 = fmaf(r41, r22, r39 * r26);
    r22 = r27 * r22;
    r22 = r22 * r34;
    r22 = r24 <= r15 ? r0 : r22;
    r39 = fmaf(r35, r39, r18 * r22);
    r39 = r39 + r25;
    r16 = r16 * r35;
    r16 = r16 * r39;
    WriteSum2<float, float>((float*)inout_shared, r0, r16);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r16 = r38 * r38;
    r25 = r4 * r4;
    r22 = r11 * r11;
    WriteSum4<float, float>((float*)inout_shared, r16, r25, r22, r0);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r22 = r39 * r39;
    WriteSum2<float, float>((float*)inout_shared, r0, r22);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r22 = r38 * r4;
    r25 = r38 * r11;
    WriteSum4<float, float>((float*)inout_shared, r22, r25, r0, r0);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r38 = r38 * r39;
    r25 = r4 * r11;
    WriteSum4<float, float>((float*)inout_shared, r38, r25, r0, r0);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r4 = r4 * r39;
    r39 = r11 * r39;
    WriteSum4<float, float>((float*)inout_shared, r4, r0, r0, r39);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void PinholeLogDepthFixedScaleFixedPointResJacFirst(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* log_depth,
    unsigned int log_depth_num_alloc,
    float* loss,
    unsigned int loss_num_alloc,
    float* scale,
    unsigned int scale_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
    float* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    float* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeLogDepthFixedScaleFixedPointResJacFirstKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      log_depth,
      log_depth_num_alloc,
      loss,
      loss_num_alloc,
      scale,
      scale_num_alloc,
      point,
      point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_rTr,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_pose_precond_diag,
      out_pose_precond_diag_num_alloc,
      out_pose_precond_tril,
      out_pose_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar