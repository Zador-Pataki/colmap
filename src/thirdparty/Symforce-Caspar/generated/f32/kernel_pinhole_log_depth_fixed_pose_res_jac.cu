#include "kernel_pinhole_log_depth_fixed_pose_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) PinholeLogDepthFixedPoseResJacKernel(
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
    float* pose,
    unsigned int pose_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
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

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36;

  if (global_thread_idx < problem_size) {
    r0 = 0.00000000000000000e+00;
    ReadIdx3<1024, float, float, float4>(
        pose, 4 * pose_num_alloc, global_thread_idx, r1, r2, r3);
  };
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
    ReadIdx4<1024, float, float, float4>(
        pose, 0 * pose_num_alloc, global_thread_idx, r5, r6, r7, r8);
    r9 = r6 * r7;
    r10 = 2.00000000000000000e+00;
    r11 = r5 * r8;
    r11 = fmaf(r10, r11, r10 * r9);
    r1 = fmaf(r1, r11, r3);
    r3 = r5 * r7;
    r9 = -2.00000000000000000e+00;
    r12 = r6 * r9;
    r3 = fmaf(r8, r12, r10 * r3);
    r13 = 1.00000000000000000e+00;
    r12 = fmaf(r6, r12, r13);
    r14 = r5 * r5;
    r12 = fmaf(r9, r14, r12);
    r1 = fmaf(r2, r3, r1);
    r1 = fmaf(r4, r12, r1);
  };
  LoadShared<1, float, float>(
      scale, 0 * scale_num_alloc, scale_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>(
        (float*)inout_shared, scale_indices_loc[threadIdx.x].target, r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r2 = -1.00000000000000000e+00;
    ReadIdx1<1024, float, float, float>(
        log_depth, 0 * log_depth_num_alloc, global_thread_idx, r14);
    r14 = fmaf(r14, r2, r4 * r2);
    r4 = 9.99999999999999955e-07;
    r9 = fmaxf(r1, r4);
    r15 = logf(r9);
    r14 = r14 + r15;
    r14 = r0 < r1 ? r14 : r0;
    ReadIdx3<1024, float, float, float4>(
        loss, 0 * loss_num_alloc, global_thread_idx, r15, r16, r17);
    r17 = fmaxf(r17, r0);
    r18 = sqrtf(r17);
    r19 = r14 * r14;
    r20 = fmaxf(r4, r19);
    r21 = 1.0 / r20;
    r22 = 5.00000000000000000e-01;
    r16 = fmaxf(r16, r4);
    r23 = r16 * r16;
    r24 = r2 * r16;
    r25 = r10 * r16;
    r26 = sqrtf(r20);
    r25 = fmaf(r26, r25, r16 * r24);
    r25 = r19 <= r23 ? r19 : r25;
    r24 = 2.50000000000000000e+00;
    r26 = r16 * r16;
    r27 = r14 * r14;
    r28 = 1.0 / r23;
    r27 = fmaf(r28, r27, r13);
    r28 = logf(r27);
    r26 = r26 * r28;
    r25 = r15 < r24 ? r26 : r25;
    r26 = 1.50000000000000000e+00;
    r28 = r10 * r16;
    r29 = sqrtf(r27);
    r29 = r2 + r29;
    r28 = r28 * r16;
    r28 = r28 * r29;
    r25 = r15 < r26 ? r28 : r25;
    r25 = r15 < r22 ? r19 : r25;
    r28 = fmaxf(r0, r25);
    r28 = r17 * r28;
    r29 = r21 * r28;
    r30 = sqrtf(r29);
    r30 = r19 <= r4 ? r18 : r30;
    r18 = r14 * r30;
    WriteIdx1<1024, float, float, float>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r18);
    r18 = r0 < r1 ? r2 : r0;
    r31 = r2 * r14;
    r32 = -9.99999999999999955e-07;
    r33 = r32 + r19;
    r33 = copysign(1.0, r33);
    r33 = r13 + r33;
    r34 = r20 * r20;
    r34 = 1.0 / r34;
    r31 = r31 * r33;
    r31 = r31 * r34;
    r31 = r31 * r28;
    r28 = r10 * r14;
    r34 = r18 * r28;
    r35 = r14 * r16;
    r20 = rsqrtf(r20);
    r35 = r35 * r33;
    r35 = r35 * r20;
    r20 = r18 * r35;
    r20 = r19 <= r23 ? r34 : r20;
    r33 = 1.0 / r27;
    r36 = r18 * r33;
    r36 = r36 * r28;
    r20 = r15 < r24 ? r36 : r20;
    r27 = rsqrtf(r27);
    r27 = r27 * r28;
    r36 = r18 * r27;
    r20 = r15 < r26 ? r36 : r20;
    r20 = r15 < r22 ? r34 : r20;
    r17 = r17 * r22;
    r25 = copysign(1.0, r25);
    r25 = r13 + r25;
    r17 = r17 * r25;
    r17 = r17 * r21;
    r20 = fmaf(r20, r17, r18 * r31);
    r20 = r22 * r20;
    r29 = rsqrtf(r29);
    r20 = r20 * r29;
    r20 = r19 <= r4 ? r0 : r20;
    r18 = fmaf(r30, r18, r14 * r20);
    r20 = r2 * r14;
    r21 = r0 < r1 ? r0 : r0;
    r25 = r21 * r28;
    r34 = r21 * r35;
    r34 = r19 <= r23 ? r25 : r34;
    r36 = r21 * r33;
    r36 = r36 * r28;
    r34 = r15 < r24 ? r36 : r34;
    r36 = r21 * r27;
    r34 = r15 < r26 ? r36 : r34;
    r34 = r15 < r22 ? r25 : r34;
    r34 = fmaf(r34, r17, r21 * r31);
    r34 = r22 * r34;
    r34 = r34 * r29;
    r34 = r19 <= r4 ? r0 : r34;
    r25 = r2 * r30;
    r25 = fmaf(r21, r25, r34 * r20);
    r18 = r18 + r25;
    WriteIdx1<1024, float, float, float>(
        out_scale_jac, 0 * out_scale_jac_num_alloc, global_thread_idx, r18);
    r20 = r2 * r14;
    r20 = r20 * r30;
    r20 = r20 * r18;
    WriteSum1<float, float>((float*)inout_shared, r20);
  };
  FlushSumShared<1, float>(out_scale_njtr,
                           0 * out_scale_njtr_num_alloc,
                           scale_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r18 = r18 * r18;
    WriteSum1<float, float>((float*)inout_shared, r18);
  };
  FlushSumShared<1, float>(out_scale_precond_diag,
                           0 * out_scale_precond_diag_num_alloc,
                           scale_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = r32 + r1;
    r32 = copysign(1.0, r32);
    r32 = r13 + r32;
    r32 = r22 * r32;
    r9 = 1.0 / r9;
    r32 = r32 * r9;
    r3 = r3 * r32;
    r3 = r0 < r1 ? r3 : r0;
    r9 = r3 * r28;
    r13 = r3 * r35;
    r13 = r19 <= r23 ? r9 : r13;
    r18 = r3 * r33;
    r18 = r18 * r28;
    r13 = r15 < r24 ? r18 : r13;
    r18 = r3 * r27;
    r13 = r15 < r26 ? r18 : r13;
    r13 = r15 < r22 ? r9 : r13;
    r13 = fmaf(r13, r17, r3 * r31);
    r13 = r22 * r13;
    r13 = r13 * r29;
    r13 = r19 <= r4 ? r0 : r13;
    r13 = fmaf(r14, r13, r30 * r3);
    r13 = r13 + r25;
    r11 = r11 * r32;
    r11 = r0 < r1 ? r11 : r0;
    r3 = r11 * r28;
    r9 = r11 * r35;
    r9 = r19 <= r23 ? r3 : r9;
    r18 = r11 * r33;
    r18 = r18 * r28;
    r9 = r15 < r24 ? r18 : r9;
    r18 = r11 * r27;
    r9 = r15 < r26 ? r18 : r9;
    r9 = r15 < r22 ? r3 : r9;
    r9 = fmaf(r11, r31, r9 * r17);
    r9 = r22 * r9;
    r9 = r9 * r29;
    r9 = r19 <= r4 ? r0 : r9;
    r9 = fmaf(r14, r9, r30 * r11);
    r9 = r9 + r25;
    r32 = r12 * r32;
    r32 = r0 < r1 ? r32 : r0;
    r1 = r32 * r28;
    r35 = r32 * r35;
    r35 = r19 <= r23 ? r1 : r35;
    r33 = r32 * r33;
    r33 = r33 * r28;
    r35 = r15 < r24 ? r33 : r35;
    r27 = r32 * r27;
    r35 = r15 < r26 ? r27 : r35;
    r35 = r15 < r22 ? r1 : r35;
    r31 = fmaf(r32, r31, r35 * r17);
    r31 = r22 * r31;
    r31 = r31 * r29;
    r31 = r19 <= r4 ? r0 : r31;
    r32 = fmaf(r30, r32, r14 * r31);
    r32 = r32 + r25;
    WriteIdx3<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r13,
                                          r9,
                                          r32);
    r25 = r2 * r14;
    r25 = r25 * r30;
    r25 = r25 * r13;
    r31 = r2 * r14;
    r31 = r31 * r30;
    r31 = r31 * r9;
    r0 = r2 * r14;
    r0 = r0 * r30;
    r0 = r0 * r32;
    WriteSum3<float, float>((float*)inout_shared, r25, r31, r0);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = r13 * r13;
    r31 = r9 * r9;
    r25 = r32 * r32;
    WriteSum3<float, float>((float*)inout_shared, r0, r31, r25);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = r13 * r9;
    r13 = r13 * r32;
    r32 = r9 * r32;
    WriteSum3<float, float>((float*)inout_shared, r25, r13, r32);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
}

void PinholeLogDepthFixedPoseResJac(
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
    float* pose,
    unsigned int pose_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
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
  PinholeLogDepthFixedPoseResJacKernel<<<n_blocks, 1024>>>(
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
      pose,
      pose_num_alloc,
      out_res,
      out_res_num_alloc,
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