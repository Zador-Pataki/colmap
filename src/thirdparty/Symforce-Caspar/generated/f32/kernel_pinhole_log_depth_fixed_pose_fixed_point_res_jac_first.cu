#include "kernel_pinhole_log_depth_fixed_pose_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeLogDepthFixedPoseFixedPointResJacFirstKernel(
        float* scale,
        unsigned int scale_num_alloc,
        SharedIndex* scale_indices,
        float* log_depth,
        unsigned int log_depth_num_alloc,
        float* loss,
        unsigned int loss_num_alloc,
        float* pose,
        unsigned int pose_num_alloc,
        float* point,
        unsigned int point_num_alloc,
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
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34;

  if (global_thread_idx < problem_size) {
    r0 = 0.00000000000000000e+00;
    ReadIdx3<1024, float, float, float4>(
        pose, 4 * pose_num_alloc, global_thread_idx, r1, r2, r3);
    ReadIdx3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r2, r1, r4);
    ReadIdx4<1024, float, float, float4>(
        pose, 0 * pose_num_alloc, global_thread_idx, r5, r6, r7, r8);
    r9 = r6 * r7;
    r10 = 2.00000000000000000e+00;
    r11 = r5 * r8;
    r11 = fmaf(r10, r11, r10 * r9);
    r11 = fmaf(r1, r11, r3);
    r1 = r5 * r7;
    r3 = -2.00000000000000000e+00;
    r9 = r6 * r3;
    r1 = fmaf(r8, r9, r10 * r1);
    r12 = 1.00000000000000000e+00;
    r9 = fmaf(r6, r9, r12);
    r13 = r5 * r5;
    r9 = fmaf(r3, r13, r9);
    r11 = fmaf(r2, r1, r11);
    r11 = fmaf(r4, r9, r11);
  };
  LoadShared<1, float, float>(
      scale, 0 * scale_num_alloc, scale_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>(
        (float*)inout_shared, scale_indices_loc[threadIdx.x].target, r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r4 = -1.00000000000000000e+00;
    ReadIdx1<1024, float, float, float>(
        log_depth, 0 * log_depth_num_alloc, global_thread_idx, r1);
    r1 = fmaf(r1, r4, r9 * r4);
    r9 = 9.99999999999999955e-07;
    r2 = fmaxf(r11, r9);
    r2 = logf(r2);
    r1 = r1 + r2;
    r1 = r0 < r11 ? r1 : r0;
    ReadIdx3<1024, float, float, float4>(
        loss, 0 * loss_num_alloc, global_thread_idx, r2, r13, r3);
    r3 = fmaxf(r3, r0);
    r14 = sqrtf(r3);
    r15 = r1 * r1;
    r16 = fmaxf(r9, r15);
    r17 = 1.0 / r16;
    r18 = 5.00000000000000000e-01;
    r13 = fmaxf(r13, r9);
    r19 = r13 * r13;
    r20 = r10 * r13;
    r21 = sqrtf(r16);
    r20 = fmaf(r21, r20, r4 * r19);
    r20 = r15 <= r19 ? r15 : r20;
    r21 = 2.50000000000000000e+00;
    r22 = r1 * r1;
    r23 = 1.0 / r19;
    r22 = fmaf(r23, r22, r12);
    r23 = logf(r22);
    r23 = r23 * r19;
    r20 = r2 < r21 ? r23 : r20;
    r23 = 1.50000000000000000e+00;
    r24 = sqrtf(r22);
    r24 = r4 + r24;
    r24 = r10 * r24;
    r24 = r24 * r19;
    r20 = r2 < r23 ? r24 : r20;
    r20 = r2 < r18 ? r15 : r20;
    r24 = fmaxf(r0, r20);
    r24 = r3 * r24;
    r25 = r17 * r24;
    r26 = sqrtf(r25);
    r26 = r15 <= r9 ? r14 : r26;
    r14 = r1 * r26;
    WriteIdx1<1024, float, float, float>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r14);
    r14 = r1 * r1;
    r27 = r26 * r26;
    r14 = r14 * r27;
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r14);
  if (global_thread_idx < problem_size) {
    r14 = r4 * r1;
    r27 = r4 * r1;
    r28 = r0 < r11 ? r0 : r0;
    r29 = r4 * r1;
    r30 = -9.99999999999999955e-07;
    r30 = r30 + r15;
    r30 = copysign(1.0, r30);
    r30 = r12 + r30;
    r31 = r16 * r16;
    r31 = 1.0 / r31;
    r29 = r29 * r30;
    r29 = r29 * r31;
    r29 = r29 * r24;
    r24 = r10 * r1;
    r31 = r28 * r24;
    r32 = r1 * r13;
    r16 = rsqrtf(r16);
    r32 = r32 * r30;
    r32 = r32 * r28;
    r32 = r32 * r16;
    r32 = r15 <= r19 ? r31 : r32;
    r33 = 1.0 / r22;
    r34 = r28 * r33;
    r34 = r34 * r24;
    r32 = r2 < r21 ? r34 : r32;
    r22 = rsqrtf(r22);
    r34 = r28 * r22;
    r34 = r34 * r24;
    r32 = r2 < r23 ? r34 : r32;
    r32 = r2 < r18 ? r31 : r32;
    r3 = r3 * r18;
    r20 = copysign(1.0, r20);
    r20 = r12 + r20;
    r3 = r3 * r20;
    r3 = r3 * r17;
    r32 = fmaf(r32, r3, r28 * r29);
    r32 = r18 * r32;
    r25 = rsqrtf(r25);
    r32 = r32 * r25;
    r32 = r15 <= r9 ? r0 : r32;
    r11 = r0 < r11 ? r4 : r0;
    r17 = r10 * r1;
    r17 = r17 * r11;
    r20 = r1 * r13;
    r20 = r20 * r30;
    r20 = r20 * r11;
    r20 = r20 * r16;
    r20 = r15 <= r19 ? r17 : r20;
    r33 = r33 * r17;
    r20 = r2 < r21 ? r33 : r20;
    r22 = r22 * r17;
    r20 = r2 < r23 ? r22 : r20;
    r20 = r2 < r18 ? r17 : r20;
    r3 = fmaf(r20, r3, r11 * r29);
    r3 = r18 * r3;
    r3 = r3 * r25;
    r3 = r15 <= r9 ? r0 : r3;
    r3 = fmaf(r1, r3, r32 * r27);
    r27 = r4 * r26;
    r3 = fmaf(r28, r27, r3);
    r3 = fmaf(r26, r11, r3);
    r14 = r14 * r26;
    r14 = r14 * r3;
    WriteSum1<float, float>((float*)inout_shared, r14);
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

void PinholeLogDepthFixedPoseFixedPointResJacFirst(
    float* scale,
    unsigned int scale_num_alloc,
    SharedIndex* scale_indices,
    float* log_depth,
    unsigned int log_depth_num_alloc,
    float* loss,
    unsigned int loss_num_alloc,
    float* pose,
    unsigned int pose_num_alloc,
    float* point,
    unsigned int point_num_alloc,
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
  PinholeLogDepthFixedPoseFixedPointResJacFirstKernel<<<n_blocks, 1024>>>(
      scale,
      scale_num_alloc,
      scale_indices,
      log_depth,
      log_depth_num_alloc,
      loss,
      loss_num_alloc,
      pose,
      pose_num_alloc,
      point,
      point_num_alloc,
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