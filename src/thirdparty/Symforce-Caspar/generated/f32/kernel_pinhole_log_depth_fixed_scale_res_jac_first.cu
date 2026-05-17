#include "kernel_pinhole_log_depth_fixed_scale_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeLogDepthFixedScaleResJacFirstKernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* log_depth,
        unsigned int log_depth_num_alloc,
        float* loss,
        unsigned int loss_num_alloc,
        float* scale,
        unsigned int scale_num_alloc,
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

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
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
        scale, 0 * scale_num_alloc, global_thread_idx, r17);
    r18 = -1.00000000000000000e+00;
    ReadIdx1<1024, float, float, float>(
        log_depth, 0 * log_depth_num_alloc, global_thread_idx, r19);
    r19 = fmaf(r19, r18, r17 * r18);
    r17 = 9.99999999999999955e-07;
    r20 = fmaxf(r3, r17);
    r21 = logf(r20);
    r19 = r19 + r21;
    r19 = r0 < r3 ? r19 : r0;
    ReadIdx3<1024, float, float, float4>(
        loss, 0 * loss_num_alloc, global_thread_idx, r21, r22, r23);
    r23 = fmaxf(r23, r0);
    r24 = sqrtf(r23);
    r25 = r19 * r19;
    r26 = fmaxf(r17, r25);
    r27 = 1.0 / r26;
    r28 = 5.00000000000000000e-01;
    r22 = fmaxf(r22, r17);
    r29 = r22 * r22;
    r30 = r18 * r22;
    r31 = r10 * r22;
    r32 = sqrtf(r26);
    r31 = fmaf(r32, r31, r22 * r30);
    r31 = r25 <= r29 ? r25 : r31;
    r30 = 2.50000000000000000e+00;
    r32 = r22 * r22;
    r33 = r19 * r19;
    r34 = 1.0 / r29;
    r33 = fmaf(r34, r33, r14);
    r34 = logf(r33);
    r32 = r32 * r34;
    r31 = r21 < r30 ? r32 : r31;
    r32 = 1.50000000000000000e+00;
    r34 = r10 * r22;
    r35 = sqrtf(r33);
    r35 = r18 + r35;
    r34 = r34 * r22;
    r34 = r34 * r35;
    r31 = r21 < r32 ? r34 : r31;
    r31 = r21 < r28 ? r25 : r31;
    r34 = fmaxf(r0, r31);
    r34 = r23 * r34;
    r35 = r27 * r34;
    r36 = sqrtf(r35);
    r36 = r25 <= r17 ? r24 : r36;
    r24 = r19 * r36;
    WriteIdx1<1024, float, float, float>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r24);
    r24 = r19 * r19;
    r37 = r36 * r36;
    r24 = r24 * r37;
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r24);
  if (global_thread_idx < problem_size) {
    r24 = r5 * r8;
    r13 = fmaf(r7, r13, r12 * r24);
    r24 = r8 * r8;
    r37 = r7 * r7;
    r38 = r24 + r37;
    r39 = r6 * r6;
    r38 = fmaf(r18, r39, r38);
    r38 = fmaf(r18, r16, r38);
    r38 = fmaf(r1, r38, r4 * r13);
    r13 = -9.99999999999999955e-07;
    r39 = r13 + r3;
    r39 = copysign(1.0, r39);
    r39 = r14 + r39;
    r39 = r28 * r39;
    r20 = 1.0 / r20;
    r39 = r39 * r20;
    r38 = r38 * r39;
    r38 = r0 < r3 ? r38 : r0;
    r20 = r10 * r19;
    r40 = r38 * r20;
    r41 = r19 * r22;
    r13 = r13 + r25;
    r13 = copysign(1.0, r13);
    r13 = r14 + r13;
    r42 = rsqrtf(r26);
    r41 = r41 * r13;
    r41 = r41 * r42;
    r42 = r38 * r41;
    r42 = r25 <= r29 ? r40 : r42;
    r43 = 1.0 / r33;
    r44 = r38 * r43;
    r44 = r44 * r20;
    r42 = r21 < r30 ? r44 : r42;
    r33 = rsqrtf(r33);
    r33 = r33 * r20;
    r44 = r38 * r33;
    r42 = r21 < r32 ? r44 : r42;
    r42 = r21 < r28 ? r40 : r42;
    r23 = r23 * r28;
    r31 = copysign(1.0, r31);
    r31 = r14 + r31;
    r23 = r23 * r31;
    r23 = r23 * r27;
    r27 = r18 * r19;
    r26 = r26 * r26;
    r26 = 1.0 / r26;
    r27 = r27 * r13;
    r27 = r27 * r26;
    r27 = r27 * r34;
    r42 = fmaf(r38, r27, r42 * r23);
    r42 = r28 * r42;
    r35 = rsqrtf(r35);
    r42 = r42 * r35;
    r42 = r25 <= r17 ? r0 : r42;
    r38 = fmaf(r36, r38, r19 * r42);
    r42 = r18 * r19;
    r34 = r0 < r3 ? r0 : r0;
    r26 = r34 * r20;
    r13 = r34 * r41;
    r13 = r25 <= r29 ? r26 : r13;
    r31 = r34 * r43;
    r31 = r31 * r20;
    r13 = r21 < r30 ? r31 : r13;
    r31 = r34 * r33;
    r13 = r21 < r32 ? r31 : r13;
    r13 = r21 < r28 ? r26 : r13;
    r13 = fmaf(r13, r23, r34 * r27);
    r13 = r28 * r13;
    r13 = r13 * r35;
    r13 = r25 <= r17 ? r0 : r13;
    r26 = r18 * r36;
    r26 = fmaf(r34, r26, r13 * r42);
    r38 = r38 + r26;
    r16 = fmaf(r6, r6, r16);
    r16 = fmaf(r18, r24, r16);
    r16 = fmaf(r18, r37, r16);
    r16 = fmaf(r2, r16, r4 * r9);
    r16 = r16 * r39;
    r16 = r0 < r3 ? r16 : r0;
    r4 = r16 * r20;
    r37 = r16 * r41;
    r37 = r25 <= r29 ? r4 : r37;
    r24 = r16 * r43;
    r24 = r24 * r20;
    r37 = r21 < r30 ? r24 : r37;
    r24 = r16 * r33;
    r37 = r21 < r32 ? r24 : r37;
    r37 = r21 < r28 ? r4 : r37;
    r37 = fmaf(r16, r27, r37 * r23);
    r37 = r28 * r37;
    r37 = r37 * r35;
    r37 = r25 <= r17 ? r0 : r37;
    r37 = fmaf(r19, r37, r36 * r16);
    r37 = r37 + r26;
    r16 = r5 * r7;
    r4 = r6 * r8;
    r4 = fmaf(r10, r4, r12 * r16);
    r4 = fmaf(r1, r4, r2 * r11);
    r4 = r4 * r39;
    r4 = r0 < r3 ? r4 : r0;
    r1 = r4 * r20;
    r2 = r4 * r41;
    r2 = r25 <= r29 ? r1 : r2;
    r16 = r4 * r43;
    r16 = r16 * r20;
    r2 = r21 < r30 ? r16 : r2;
    r16 = r4 * r33;
    r2 = r21 < r32 ? r16 : r2;
    r2 = r21 < r28 ? r1 : r2;
    r2 = fmaf(r2, r23, r4 * r27);
    r2 = r28 * r2;
    r2 = r2 * r35;
    r2 = r25 <= r17 ? r0 : r2;
    r2 = fmaf(r19, r2, r36 * r4);
    r2 = r2 + r26;
    r4 = r0 < r3 ? r39 : r0;
    r1 = r4 * r20;
    r16 = r4 * r41;
    r16 = r25 <= r29 ? r1 : r16;
    r12 = r4 * r43;
    r12 = r12 * r20;
    r16 = r21 < r30 ? r12 : r16;
    r12 = r4 * r33;
    r16 = r21 < r32 ? r12 : r16;
    r16 = r21 < r28 ? r1 : r16;
    r16 = fmaf(r16, r23, r4 * r27);
    r16 = r28 * r16;
    r16 = r16 * r35;
    r16 = r25 <= r17 ? r0 : r16;
    r4 = fmaf(r36, r4, r19 * r16);
    r4 = r4 + r26;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r38,
                                          r37,
                                          r2,
                                          r4);
    r16 = r18 * r19;
    r16 = r16 * r36;
    r16 = r16 * r38;
    r1 = r18 * r19;
    r1 = r1 * r36;
    r1 = r1 * r37;
    r12 = r18 * r19;
    r12 = r12 * r36;
    r12 = r12 * r2;
    WriteSum4<float, float>((float*)inout_shared, r16, r1, r12, r0);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r12 = r18 * r19;
    r12 = r12 * r36;
    r12 = r12 * r4;
    WriteSum2<float, float>((float*)inout_shared, r0, r12);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r12 = r38 * r38;
    r1 = r37 * r37;
    r16 = r2 * r2;
    WriteSum4<float, float>((float*)inout_shared, r12, r1, r16, r0);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r16 = r4 * r4;
    WriteSum2<float, float>((float*)inout_shared, r0, r16);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r16 = r38 * r37;
    r1 = r38 * r2;
    WriteSum4<float, float>((float*)inout_shared, r16, r1, r0, r0);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r38 = r38 * r4;
    r1 = r37 * r2;
    WriteSum4<float, float>((float*)inout_shared, r38, r1, r0, r0);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = r37 * r4;
    r4 = r2 * r4;
    WriteSum4<float, float>((float*)inout_shared, r37, r0, r0, r4);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r9 = r9 * r39;
    r9 = r0 < r3 ? r9 : r0;
    r4 = r9 * r20;
    r37 = r9 * r41;
    r37 = r25 <= r29 ? r4 : r37;
    r2 = r9 * r43;
    r2 = r2 * r20;
    r37 = r21 < r30 ? r2 : r37;
    r2 = r9 * r33;
    r37 = r21 < r32 ? r2 : r37;
    r37 = r21 < r28 ? r4 : r37;
    r37 = fmaf(r37, r23, r9 * r27);
    r37 = r28 * r37;
    r37 = r37 * r35;
    r37 = r25 <= r17 ? r0 : r37;
    r37 = fmaf(r19, r37, r36 * r9);
    r37 = r37 + r26;
    r11 = r11 * r39;
    r11 = r0 < r3 ? r11 : r0;
    r9 = r11 * r20;
    r4 = r11 * r41;
    r4 = r25 <= r29 ? r9 : r4;
    r2 = r11 * r43;
    r2 = r2 * r20;
    r4 = r21 < r30 ? r2 : r4;
    r2 = r11 * r33;
    r4 = r21 < r32 ? r2 : r4;
    r4 = r21 < r28 ? r9 : r4;
    r4 = fmaf(r11, r27, r4 * r23);
    r4 = r28 * r4;
    r4 = r4 * r35;
    r4 = r25 <= r17 ? r0 : r4;
    r4 = fmaf(r19, r4, r36 * r11);
    r4 = r4 + r26;
    r39 = r15 * r39;
    r39 = r0 < r3 ? r39 : r0;
    r3 = r39 * r20;
    r41 = r39 * r41;
    r41 = r25 <= r29 ? r3 : r41;
    r43 = r39 * r43;
    r43 = r43 * r20;
    r41 = r21 < r30 ? r43 : r41;
    r33 = r39 * r33;
    r41 = r21 < r32 ? r33 : r41;
    r41 = r21 < r28 ? r3 : r41;
    r27 = fmaf(r39, r27, r41 * r23);
    r27 = r28 * r27;
    r27 = r27 * r35;
    r27 = r25 <= r17 ? r0 : r27;
    r39 = fmaf(r36, r39, r19 * r27);
    r39 = r39 + r26;
    WriteIdx3<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r37,
                                          r4,
                                          r39);
    r26 = r18 * r19;
    r26 = r26 * r36;
    r26 = r26 * r37;
    r27 = r18 * r19;
    r27 = r27 * r36;
    r27 = r27 * r4;
    r0 = r18 * r19;
    r0 = r0 * r36;
    r0 = r0 * r39;
    WriteSum3<float, float>((float*)inout_shared, r26, r27, r0);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = r37 * r37;
    r27 = r4 * r4;
    r26 = r39 * r39;
    WriteSum3<float, float>((float*)inout_shared, r0, r27, r26);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = r37 * r4;
    r37 = r37 * r39;
    r39 = r4 * r39;
    WriteSum3<float, float>((float*)inout_shared, r26, r37, r39);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void PinholeLogDepthFixedScaleResJacFirst(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* log_depth,
    unsigned int log_depth_num_alloc,
    float* loss,
    unsigned int loss_num_alloc,
    float* scale,
    unsigned int scale_num_alloc,
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
  PinholeLogDepthFixedScaleResJacFirstKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      point,
      point_num_alloc,
      point_indices,
      log_depth,
      log_depth_num_alloc,
      loss,
      loss_num_alloc,
      scale,
      scale_num_alloc,
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