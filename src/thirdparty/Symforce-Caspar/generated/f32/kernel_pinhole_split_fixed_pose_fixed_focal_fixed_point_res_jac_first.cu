#include "kernel_pinhole_split_fixed_pose_fixed_focal_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedPoseFixedFocalFixedPointResJacFirstKernel(
        float* principal_point,
        unsigned int principal_point_num_alloc,
        SharedIndex* principal_point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* weight_loss,
        unsigned int weight_loss_num_alloc,
        float* pose,
        unsigned int pose_num_alloc,
        float* focal,
        unsigned int focal_num_alloc,
        float* point,
        unsigned int point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_rTr,
        float* const out_principal_point_njtr,
        unsigned int out_principal_point_njtr_num_alloc,
        float* const out_principal_point_precond_diag,
        unsigned int out_principal_point_precond_diag_num_alloc,
        float* const out_principal_point_precond_tril,
        unsigned int out_principal_point_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ SharedIndex principal_point_indices_loc[1024];
  principal_point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? principal_point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(weight_loss,
                                         0 * weight_loss_num_alloc,
                                         global_thread_idx,
                                         r0,
                                         r1,
                                         r2,
                                         r3);
  };
  LoadShared<2, float, float>(principal_point,
                              0 * principal_point_num_alloc,
                              principal_point_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared,
                       principal_point_indices_loc[threadIdx.x].target,
                       r4,
                       r5);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r6, r7);
    r8 = -1.00000000000000000e+00;
    r6 = fmaf(r6, r8, r4);
    ReadIdx2<1024, float, float, float2>(
        focal, 0 * focal_num_alloc, global_thread_idx, r4, r9);
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
    r30 = r4 * r25;
    r10 = 9.99999999999999955e-07;
    r31 = r17 * r18;
    r31 = r31 * r22;
    r23 = fmaf(r19, r23, r31);
    r23 = fmaf(r14, r23, r12);
    r26 = fmaf(r17, r21, r26);
    r12 = r16 * r16;
    r12 = r12 * r20;
    r29 = r12 + r29;
    r23 = fmaf(r13, r26, r23);
    r23 = fmaf(r15, r29, r23);
    r29 = copysign(1.0, r23);
    r29 = fmaf(r10, r29, r23);
    r29 = 1.0 / r29;
    r6 = fmaf(r29, r30, r6);
    r7 = fmaf(r7, r8, r5);
    r5 = r18 * r19;
    r5 = fmaf(r22, r5, r24);
    r5 = fmaf(r13, r5, r11);
    r21 = fmaf(r16, r21, r31);
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
                                         r29,
                                         r14,
                                         r21);
    r15 = 0.00000000000000000e+00;
    r21 = fmaxf(r21, r15);
    r12 = sqrtf(r21);
    r7 = fmaf(r3, r7, r2 * r6);
    r6 = fmaf(r27, r27, r7 * r7);
    r16 = 5.00000000000000000e-01;
    r14 = fmaxf(r14, r10);
    r31 = r14 * r14;
    r13 = r22 * r14;
    r11 = fmaxf(r10, r6);
    r24 = sqrtf(r11);
    r13 = fmaf(r8, r31, r24 * r13);
    r13 = r6 <= r31 ? r6 : r13;
    r24 = 2.50000000000000000e+00;
    r30 = 1.0 / r31;
    r30 = fmaf(r6, r30, r28);
    r23 = logf(r30);
    r23 = r23 * r31;
    r13 = r29 < r24 ? r23 : r13;
    r23 = 1.50000000000000000e+00;
    r26 = sqrtf(r30);
    r26 = r8 + r26;
    r26 = r22 * r26;
    r26 = r26 * r31;
    r13 = r29 < r23 ? r26 : r13;
    r13 = r29 < r16 ? r6 : r13;
    r26 = fmaxf(r15, r13);
    r20 = 1.0 / r11;
    r20 = r21 * r20;
    r32 = r26 * r20;
    r33 = sqrtf(r32);
    r33 = r6 <= r10 ? r12 : r33;
    r12 = r27 * r33;
    r34 = r7 * r33;
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r12, r34);
    r34 = r7 * r7;
    r34 = r34 * r33;
    r35 = r27 * r33;
    r35 = fmaf(r12, r35, r33 * r34);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r35);
  if (global_thread_idx < problem_size) {
    r35 = r8 * r7;
    r34 = r33 * r35;
    r36 = r0 * r22;
    r37 = r2 * r22;
    r37 = fmaf(r7, r37, r27 * r36);
    r36 = r16 * r14;
    r38 = -9.99999999999999955e-07;
    r38 = r38 + r6;
    r38 = copysign(1.0, r38);
    r38 = r28 + r38;
    r39 = rsqrtf(r11);
    r36 = r36 * r38;
    r36 = r36 * r39;
    r39 = r37 * r36;
    r39 = r6 <= r31 ? r37 : r39;
    r40 = 1.0 / r30;
    r41 = r37 * r40;
    r39 = r29 < r24 ? r41 : r39;
    r30 = rsqrtf(r30);
    r41 = r37 * r30;
    r39 = r29 < r23 ? r41 : r39;
    r39 = r29 < r16 ? r37 : r39;
    r41 = r16 * r39;
    r13 = copysign(1.0, r13);
    r13 = r28 + r13;
    r20 = r13 * r20;
    r26 = r21 * r26;
    r21 = -5.00000000000000000e-01;
    r11 = r11 * r11;
    r11 = 1.0 / r11;
    r26 = r26 * r21;
    r26 = r26 * r38;
    r26 = r26 * r11;
    r37 = fmaf(r37, r26, r20 * r41);
    r37 = r16 * r37;
    r32 = rsqrtf(r32);
    r37 = r37 * r32;
    r37 = r6 <= r10 ? r15 : r37;
    r41 = fmaf(r2, r33, r7 * r37);
    r11 = 2.50000000000000000e-01;
    r38 = r6 <= r31 ? r15 : r15;
    r38 = r29 < r24 ? r15 : r38;
    r38 = r29 < r23 ? r15 : r38;
    r38 = r29 < r16 ? r15 : r38;
    r38 = r11 * r38;
    r38 = r38 * r32;
    r38 = r38 * r20;
    r38 = r6 <= r10 ? r15 : r38;
    r35 = r38 * r35;
    r41 = r41 + r35;
    r11 = r8 * r27;
    r11 = r11 * r38;
    r37 = fmaf(r27, r37, r11);
    r37 = fmaf(r0, r33, r37);
    r38 = r8 * r37;
    r38 = fmaf(r12, r38, r41 * r34);
    r21 = r1 * r22;
    r13 = r3 * r22;
    r13 = fmaf(r7, r13, r27 * r21);
    r36 = r13 * r36;
    r36 = r6 <= r31 ? r13 : r36;
    r40 = r13 * r40;
    r36 = r29 < r24 ? r40 : r36;
    r30 = r13 * r30;
    r36 = r29 < r23 ? r30 : r36;
    r36 = r29 < r16 ? r13 : r36;
    r29 = r16 * r36;
    r26 = fmaf(r13, r26, r20 * r29);
    r26 = r16 * r26;
    r26 = r26 * r32;
    r26 = r6 <= r10 ? r15 : r26;
    r11 = fmaf(r27, r26, r11);
    r11 = fmaf(r1, r33, r11);
    r15 = r8 * r11;
    r26 = fmaf(r3, r33, r7 * r26);
    r26 = r26 + r35;
    r34 = fmaf(r26, r34, r12 * r15);
    WriteSum2<float, float>((float*)inout_shared, r38, r34);
  };
  FlushSumShared<2, float>(out_principal_point_njtr,
                           0 * out_principal_point_njtr_num_alloc,
                           principal_point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = fmaf(r41, r41, r37 * r37);
    r38 = fmaf(r26, r26, r11 * r11);
    WriteSum2<float, float>((float*)inout_shared, r34, r38);
  };
  FlushSumShared<2, float>(out_principal_point_precond_diag,
                           0 * out_principal_point_precond_diag_num_alloc,
                           principal_point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = fmaf(r37, r11, r41 * r26);
    WriteSum1<float, float>((float*)inout_shared, r26);
  };
  FlushSumShared<1, float>(out_principal_point_precond_tril,
                           0 * out_principal_point_precond_tril_num_alloc,
                           principal_point_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void PinholeSplitFixedPoseFixedFocalFixedPointResJacFirst(
    float* principal_point,
    unsigned int principal_point_num_alloc,
    SharedIndex* principal_point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* weight_loss,
    unsigned int weight_loss_num_alloc,
    float* pose,
    unsigned int pose_num_alloc,
    float* focal,
    unsigned int focal_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
    float* const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    float* const out_principal_point_precond_diag,
    unsigned int out_principal_point_precond_diag_num_alloc,
    float* const out_principal_point_precond_tril,
    unsigned int out_principal_point_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitFixedPoseFixedFocalFixedPointResJacFirstKernel<<<n_blocks,
                                                               1024>>>(
      principal_point,
      principal_point_num_alloc,
      principal_point_indices,
      pixel,
      pixel_num_alloc,
      weight_loss,
      weight_loss_num_alloc,
      pose,
      pose_num_alloc,
      focal,
      focal_num_alloc,
      point,
      point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_rTr,
      out_principal_point_njtr,
      out_principal_point_njtr_num_alloc,
      out_principal_point_precond_diag,
      out_principal_point_precond_diag_num_alloc,
      out_principal_point_precond_tril,
      out_principal_point_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar