#include "kernel_pinhole_split_fixed_pose_fixed_principal_point_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedPoseFixedPrincipalPointFixedPointResJacFirstKernel(
        float* focal,
        unsigned int focal_num_alloc,
        SharedIndex* focal_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* weight_loss,
        unsigned int weight_loss_num_alloc,
        float* pose,
        unsigned int pose_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
        float* point,
        unsigned int point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_rTr,
        float* const out_focal_njtr,
        unsigned int out_focal_njtr_num_alloc,
        float* const out_focal_precond_diag,
        unsigned int out_focal_precond_diag_num_alloc,
        float* const out_focal_precond_tril,
        unsigned int out_focal_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ SharedIndex focal_indices_loc[1024];
  focal_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(weight_loss,
                                         0 * weight_loss_num_alloc,
                                         global_thread_idx,
                                         r0,
                                         r1,
                                         r2,
                                         r3);
    ReadIdx2<1024, float, float, float2>(principal_point,
                                         0 * principal_point_num_alloc,
                                         global_thread_idx,
                                         r4,
                                         r5);
    ReadIdx2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r6, r7);
    r8 = -1.00000000000000000e+00;
    r6 = fmaf(r6, r8, r4);
  };
  LoadShared<2, float, float>(
      focal, 0 * focal_num_alloc, focal_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>(
        (float*)inout_shared, focal_indices_loc[threadIdx.x].target, r4, r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx3<1024, float, float, float4>(
        pose, 4 * pose_num_alloc, global_thread_idx, r10, r11, r12);
    ReadIdx3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r13, r14, r15);
    ReadIdx4<1024, float, float, float4>(
        pose, 0 * pose_num_alloc, global_thread_idx, r16, r17, r18, r19);
    r20 = -2.00000000000000000e+00;
    r21 = r19 * r20;
    r22 = 2.00000000000000000e+00;
    r23 = r16 * r22;
    r24 = r17 * r23;
    r25 = fmaf(r18, r21, r24);
    r25 = fmaf(r14, r25, r10);
    r10 = r17 * r19;
    r26 = r18 * r23;
    r10 = fmaf(r22, r10, r26);
    r27 = r18 * r18;
    r27 = r20 * r27;
    r28 = 1.00000000000000000e+00;
    r29 = r17 * r17;
    r29 = fmaf(r20, r29, r28);
    r30 = r27 + r29;
    r25 = fmaf(r15, r10, r25);
    r25 = fmaf(r13, r30, r25);
    r30 = 9.99999999999999955e-07;
    r10 = r17 * r18;
    r10 = r10 * r22;
    r23 = fmaf(r19, r23, r10);
    r23 = fmaf(r14, r23, r12);
    r26 = fmaf(r17, r21, r26);
    r12 = r16 * r16;
    r12 = r12 * r20;
    r29 = r12 + r29;
    r23 = fmaf(r13, r26, r23);
    r23 = fmaf(r15, r29, r23);
    r29 = copysign(1.0, r23);
    r29 = fmaf(r30, r29, r23);
    r29 = 1.0 / r29;
    r25 = r25 * r29;
    r6 = fmaf(r4, r25, r6);
    r7 = fmaf(r7, r8, r5);
    r5 = r18 * r19;
    r5 = fmaf(r22, r5, r24);
    r5 = fmaf(r13, r5, r11);
    r21 = fmaf(r16, r21, r10);
    r27 = r28 + r27;
    r27 = r27 + r12;
    r5 = fmaf(r15, r21, r5);
    r5 = fmaf(r14, r27, r5);
    r27 = r9 * r5;
    r7 = fmaf(r29, r27, r7);
    r27 = fmaf(r1, r7, r0 * r6);
    ReadIdx3<1024, float, float, float4>(weight_loss,
                                         4 * weight_loss_num_alloc,
                                         global_thread_idx,
                                         r14,
                                         r21,
                                         r15);
    r12 = 0.00000000000000000e+00;
    r15 = fmaxf(r15, r12);
    r16 = sqrtf(r15);
    r7 = fmaf(r3, r7, r2 * r6);
    r6 = fmaf(r27, r27, r7 * r7);
    r10 = 5.00000000000000000e-01;
    r21 = fmaxf(r21, r30);
    r13 = r21 * r21;
    r11 = r22 * r21;
    r24 = fmaxf(r30, r6);
    r4 = sqrtf(r24);
    r11 = fmaf(r8, r13, r4 * r11);
    r11 = r6 <= r13 ? r6 : r11;
    r4 = 2.50000000000000000e+00;
    r23 = 1.0 / r13;
    r23 = fmaf(r6, r23, r28);
    r26 = logf(r23);
    r26 = r26 * r13;
    r11 = r14 < r4 ? r26 : r11;
    r26 = 1.50000000000000000e+00;
    r20 = sqrtf(r23);
    r20 = r8 + r20;
    r20 = r22 * r20;
    r20 = r20 * r13;
    r11 = r14 < r26 ? r20 : r11;
    r11 = r14 < r10 ? r6 : r11;
    r20 = fmaxf(r12, r11);
    r31 = 1.0 / r24;
    r31 = r15 * r31;
    r32 = r20 * r31;
    r33 = sqrtf(r32);
    r33 = r6 <= r30 ? r16 : r33;
    r16 = r27 * r33;
    r34 = r7 * r33;
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r16, r34);
    r34 = r7 * r7;
    r34 = r34 * r33;
    r35 = r27 * r33;
    r35 = fmaf(r16, r35, r33 * r34);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r35);
  if (global_thread_idx < problem_size) {
    r35 = r8 * r7;
    r34 = r33 * r35;
    r0 = r0 * r25;
    r36 = r22 * r27;
    r37 = r2 * r22;
    r37 = r37 * r7;
    r37 = fmaf(r25, r37, r0 * r36);
    r38 = r10 * r21;
    r39 = -9.99999999999999955e-07;
    r39 = r39 + r6;
    r39 = copysign(1.0, r39);
    r39 = r28 + r39;
    r40 = rsqrtf(r24);
    r38 = r38 * r39;
    r38 = r38 * r40;
    r40 = r37 * r38;
    r40 = r6 <= r13 ? r37 : r40;
    r41 = 1.0 / r23;
    r42 = r37 * r41;
    r40 = r14 < r4 ? r42 : r40;
    r23 = rsqrtf(r23);
    r42 = r37 * r23;
    r40 = r14 < r26 ? r42 : r40;
    r40 = r14 < r10 ? r37 : r40;
    r42 = r10 * r40;
    r11 = copysign(1.0, r11);
    r11 = r28 + r11;
    r31 = r11 * r31;
    r20 = r15 * r20;
    r15 = -5.00000000000000000e-01;
    r24 = r24 * r24;
    r24 = 1.0 / r24;
    r20 = r20 * r15;
    r20 = r20 * r39;
    r20 = r20 * r24;
    r37 = fmaf(r37, r20, r31 * r42);
    r37 = r10 * r37;
    r32 = rsqrtf(r32);
    r37 = r37 * r32;
    r37 = r6 <= r30 ? r12 : r37;
    r42 = 2.50000000000000000e-01;
    r24 = r6 <= r13 ? r12 : r12;
    r24 = r14 < r4 ? r12 : r24;
    r24 = r14 < r26 ? r12 : r24;
    r24 = r14 < r10 ? r12 : r24;
    r24 = r42 * r24;
    r24 = r24 * r32;
    r24 = r24 * r31;
    r24 = r6 <= r30 ? r12 : r24;
    r35 = r24 * r35;
    r42 = fmaf(r7, r37, r35);
    r39 = r2 * r33;
    r42 = fmaf(r25, r39, r42);
    r39 = r8 * r27;
    r39 = r39 * r24;
    r37 = fmaf(r27, r37, r39);
    r37 = fmaf(r33, r0, r37);
    r0 = r8 * r37;
    r0 = fmaf(r16, r0, r42 * r34);
    r1 = r1 * r5;
    r1 = r1 * r29;
    r24 = r3 * r22;
    r24 = r24 * r5;
    r24 = r24 * r7;
    r24 = fmaf(r29, r24, r1 * r36);
    r38 = r24 * r38;
    r38 = r6 <= r13 ? r24 : r38;
    r41 = r24 * r41;
    r38 = r14 < r4 ? r41 : r38;
    r23 = r24 * r23;
    r38 = r14 < r26 ? r23 : r38;
    r38 = r14 < r10 ? r24 : r38;
    r14 = r10 * r38;
    r14 = fmaf(r31, r14, r24 * r20);
    r14 = r10 * r14;
    r14 = r14 * r32;
    r14 = r6 <= r30 ? r12 : r14;
    r39 = fmaf(r27, r14, r39);
    r39 = fmaf(r33, r1, r39);
    r1 = r8 * r39;
    r14 = fmaf(r7, r14, r35);
    r35 = r3 * r5;
    r35 = r35 * r33;
    r14 = fmaf(r29, r35, r14);
    r34 = fmaf(r14, r34, r16 * r1);
    WriteSum2<float, float>((float*)inout_shared, r0, r34);
  };
  FlushSumShared<2, float>(out_focal_njtr,
                           0 * out_focal_njtr_num_alloc,
                           focal_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = fmaf(r37, r37, r42 * r42);
    r0 = fmaf(r39, r39, r14 * r14);
    WriteSum2<float, float>((float*)inout_shared, r34, r0);
  };
  FlushSumShared<2, float>(out_focal_precond_diag,
                           0 * out_focal_precond_diag_num_alloc,
                           focal_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r14 = fmaf(r37, r39, r42 * r14);
    WriteSum1<float, float>((float*)inout_shared, r14);
  };
  FlushSumShared<1, float>(out_focal_precond_tril,
                           0 * out_focal_precond_tril_num_alloc,
                           focal_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void PinholeSplitFixedPoseFixedPrincipalPointFixedPointResJacFirst(
    float* focal,
    unsigned int focal_num_alloc,
    SharedIndex* focal_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* weight_loss,
    unsigned int weight_loss_num_alloc,
    float* pose,
    unsigned int pose_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
    float* const out_focal_njtr,
    unsigned int out_focal_njtr_num_alloc,
    float* const out_focal_precond_diag,
    unsigned int out_focal_precond_diag_num_alloc,
    float* const out_focal_precond_tril,
    unsigned int out_focal_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitFixedPoseFixedPrincipalPointFixedPointResJacFirstKernel<<<
      n_blocks,
      1024>>>(focal,
              focal_num_alloc,
              focal_indices,
              pixel,
              pixel_num_alloc,
              weight_loss,
              weight_loss_num_alloc,
              pose,
              pose_num_alloc,
              principal_point,
              principal_point_num_alloc,
              point,
              point_num_alloc,
              out_res,
              out_res_num_alloc,
              out_rTr,
              out_focal_njtr,
              out_focal_njtr_num_alloc,
              out_focal_precond_diag,
              out_focal_precond_diag_num_alloc,
              out_focal_precond_tril,
              out_focal_precond_tril_num_alloc,
              problem_size);
}

}  // namespace caspar