#include "kernel_pinhole_fixed_pose_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeFixedPoseFixedPointResJacFirstKernel(
        float* calib,
        unsigned int calib_num_alloc,
        SharedIndex* calib_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* weight_loss,
        unsigned int weight_loss_num_alloc,
        float* pose,
        unsigned int pose_num_alloc,
        float* point,
        unsigned int point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_rTr,
        float* const out_calib_njtr,
        unsigned int out_calib_njtr_num_alloc,
        float* const out_calib_precond_diag,
        unsigned int out_calib_precond_diag_num_alloc,
        float* const out_calib_precond_tril,
        unsigned int out_calib_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex calib_indices_loc[1024];
  calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(weight_loss,
                                         0 * weight_loss_num_alloc,
                                         global_thread_idx,
                                         r0,
                                         r1,
                                         r2,
                                         r3);
  };
  LoadShared<4, float, float>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       calib_indices_loc[threadIdx.x].target,
                       r4,
                       r5,
                       r6,
                       r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r8, r9);
    r10 = -1.00000000000000000e+00;
    r8 = fmaf(r8, r10, r6);
    ReadIdx3<1024, float, float, float4>(
        pose, 4 * pose_num_alloc, global_thread_idx, r6, r11, r12);
    ReadIdx3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r13, r14, r15);
    ReadIdx4<1024, float, float, float4>(
        pose, 0 * pose_num_alloc, global_thread_idx, r16, r17, r18, r19);
    r20 = r16 * r17;
    r21 = 2.00000000000000000e+00;
    r20 = r20 * r21;
    r22 = -2.00000000000000000e+00;
    r23 = r19 * r22;
    r24 = fmaf(r18, r23, r20);
    r24 = fmaf(r14, r24, r6);
    r6 = r16 * r18;
    r6 = r6 * r21;
    r25 = r17 * r19;
    r25 = fmaf(r21, r25, r6);
    r26 = r18 * r18;
    r26 = r22 * r26;
    r27 = 1.00000000000000000e+00;
    r28 = r17 * r17;
    r28 = fmaf(r22, r28, r27);
    r29 = r26 + r28;
    r24 = fmaf(r15, r25, r24);
    r24 = fmaf(r13, r29, r24);
    r29 = 9.99999999999999955e-07;
    r25 = r17 * r18;
    r25 = r25 * r21;
    r30 = r16 * r19;
    r30 = fmaf(r21, r30, r25);
    r30 = fmaf(r14, r30, r12);
    r6 = fmaf(r17, r23, r6);
    r12 = r16 * r16;
    r12 = r22 * r12;
    r28 = r12 + r28;
    r30 = fmaf(r13, r6, r30);
    r30 = fmaf(r15, r28, r30);
    r28 = copysign(1.0, r30);
    r28 = fmaf(r29, r28, r30);
    r28 = 1.0 / r28;
    r24 = r24 * r28;
    r8 = fmaf(r4, r24, r8);
    r9 = fmaf(r9, r10, r7);
    r7 = r18 * r19;
    r7 = fmaf(r21, r7, r20);
    r7 = fmaf(r13, r7, r11);
    r23 = fmaf(r16, r23, r25);
    r26 = r27 + r26;
    r26 = r26 + r12;
    r7 = fmaf(r15, r23, r7);
    r7 = fmaf(r14, r26, r7);
    r26 = r5 * r7;
    r9 = fmaf(r28, r26, r9);
    r26 = fmaf(r1, r9, r0 * r8);
    ReadIdx3<1024, float, float, float4>(weight_loss,
                                         4 * weight_loss_num_alloc,
                                         global_thread_idx,
                                         r14,
                                         r23,
                                         r15);
    r12 = 0.00000000000000000e+00;
    r15 = fmaxf(r15, r12);
    r25 = sqrtf(r15);
    r9 = fmaf(r3, r9, r2 * r8);
    r8 = fmaf(r9, r9, r26 * r26);
    r13 = 5.00000000000000000e-01;
    r23 = fmaxf(r23, r29);
    r11 = r23 * r23;
    r20 = r21 * r23;
    r4 = fmaxf(r29, r8);
    r30 = sqrtf(r4);
    r6 = r10 * r23;
    r6 = fmaf(r23, r6, r30 * r20);
    r6 = r8 <= r11 ? r8 : r6;
    r20 = 2.50000000000000000e+00;
    r30 = r23 * r23;
    r22 = 1.0 / r11;
    r22 = fmaf(r8, r22, r27);
    r31 = logf(r22);
    r30 = r30 * r31;
    r6 = r14 < r20 ? r30 : r6;
    r30 = 1.50000000000000000e+00;
    r31 = r21 * r23;
    r32 = sqrtf(r22);
    r32 = r10 + r32;
    r31 = r31 * r23;
    r31 = r31 * r32;
    r6 = r14 < r30 ? r31 : r6;
    r6 = r14 < r13 ? r8 : r6;
    r31 = fmaxf(r12, r6);
    r32 = 1.0 / r4;
    r32 = r15 * r32;
    r33 = r31 * r32;
    r34 = sqrtf(r33);
    r34 = r8 <= r29 ? r25 : r34;
    r25 = r26 * r34;
    r35 = r9 * r34;
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r25, r35);
    r35 = r9 * r9;
    r35 = r35 * r34;
    r25 = r26 * r26;
    r25 = r25 * r34;
    r25 = fmaf(r34, r25, r34 * r35);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r25);
  if (global_thread_idx < problem_size) {
    r25 = r10 * r34;
    r35 = r9 * r25;
    r31 = r15 * r31;
    r15 = -5.00000000000000000e-01;
    r36 = -9.99999999999999955e-07;
    r36 = r36 + r8;
    r36 = copysign(1.0, r36);
    r36 = r27 + r36;
    r37 = r4 * r4;
    r37 = 1.0 / r37;
    r31 = r31 * r15;
    r31 = r31 * r36;
    r31 = r31 * r37;
    r37 = r21 * r26;
    r15 = r0 * r24;
    r38 = r2 * r21;
    r38 = r38 * r9;
    r38 = fmaf(r24, r38, r37 * r15);
    r39 = r13 * r23;
    r4 = rsqrtf(r4);
    r39 = r39 * r36;
    r39 = r39 * r4;
    r4 = r38 * r39;
    r4 = r8 <= r11 ? r38 : r4;
    r36 = 1.0 / r22;
    r40 = r38 * r36;
    r4 = r14 < r20 ? r40 : r4;
    r22 = rsqrtf(r22);
    r40 = r38 * r22;
    r4 = r14 < r30 ? r40 : r4;
    r4 = r14 < r13 ? r38 : r4;
    r40 = r13 * r4;
    r6 = copysign(1.0, r6);
    r6 = r27 + r6;
    r32 = r6 * r32;
    r40 = fmaf(r32, r40, r38 * r31);
    r40 = r13 * r40;
    r33 = rsqrtf(r33);
    r40 = r40 * r33;
    r40 = r8 <= r29 ? r12 : r40;
    r38 = r10 * r9;
    r6 = 2.50000000000000000e-01;
    r27 = r8 <= r11 ? r12 : r12;
    r27 = r14 < r20 ? r12 : r27;
    r27 = r14 < r30 ? r12 : r27;
    r27 = r14 < r13 ? r12 : r27;
    r27 = r6 * r27;
    r27 = r27 * r33;
    r27 = r27 * r32;
    r27 = r8 <= r29 ? r12 : r27;
    r38 = r38 * r27;
    r6 = fmaf(r9, r40, r38);
    r41 = r2 * r34;
    r6 = fmaf(r24, r41, r6);
    r41 = r10 * r26;
    r41 = r41 * r27;
    r40 = fmaf(r26, r40, r41);
    r40 = fmaf(r34, r15, r40);
    r15 = r26 * r40;
    r15 = fmaf(r25, r15, r6 * r35);
    r27 = r1 * r7;
    r27 = r27 * r28;
    r24 = r3 * r21;
    r24 = r24 * r7;
    r24 = r24 * r9;
    r24 = fmaf(r28, r24, r37 * r27);
    r42 = r24 * r39;
    r42 = r8 <= r11 ? r24 : r42;
    r43 = r24 * r36;
    r42 = r14 < r20 ? r43 : r42;
    r43 = r24 * r22;
    r42 = r14 < r30 ? r43 : r42;
    r42 = r14 < r13 ? r24 : r42;
    r43 = r13 * r42;
    r24 = fmaf(r24, r31, r32 * r43);
    r24 = r13 * r24;
    r24 = r24 * r33;
    r24 = r8 <= r29 ? r12 : r24;
    r43 = fmaf(r26, r24, r41);
    r43 = fmaf(r34, r27, r43);
    r27 = r26 * r43;
    r24 = fmaf(r9, r24, r38);
    r44 = r3 * r7;
    r44 = r44 * r34;
    r24 = fmaf(r28, r44, r24);
    r27 = fmaf(r24, r35, r25 * r27);
    r44 = fmaf(r2, r34, r38);
    r28 = r2 * r21;
    r28 = fmaf(r0, r37, r9 * r28);
    r45 = r28 * r39;
    r45 = r8 <= r11 ? r28 : r45;
    r46 = r28 * r36;
    r45 = r14 < r20 ? r46 : r45;
    r46 = r28 * r22;
    r45 = r14 < r30 ? r46 : r45;
    r45 = r14 < r13 ? r28 : r45;
    r46 = r13 * r45;
    r46 = fmaf(r32, r46, r28 * r31);
    r46 = r13 * r46;
    r46 = r46 * r33;
    r46 = r8 <= r29 ? r12 : r46;
    r44 = fmaf(r9, r46, r44);
    r0 = fmaf(r0, r34, r41);
    r0 = fmaf(r26, r46, r0);
    r46 = r26 * r0;
    r46 = fmaf(r25, r46, r44 * r35);
    r28 = r3 * r21;
    r37 = fmaf(r1, r37, r9 * r28);
    r39 = r37 * r39;
    r39 = r8 <= r11 ? r37 : r39;
    r36 = r37 * r36;
    r39 = r14 < r20 ? r36 : r39;
    r22 = r37 * r22;
    r39 = r14 < r30 ? r22 : r39;
    r39 = r14 < r13 ? r37 : r39;
    r14 = r13 * r39;
    r31 = fmaf(r37, r31, r32 * r14);
    r31 = r13 * r31;
    r31 = r31 * r33;
    r31 = r8 <= r29 ? r12 : r31;
    r38 = fmaf(r9, r31, r38);
    r38 = fmaf(r3, r34, r38);
    r31 = fmaf(r26, r31, r41);
    r31 = fmaf(r1, r34, r31);
    r1 = r26 * r31;
    r1 = fmaf(r25, r1, r38 * r35);
    WriteSum4<float, float>((float*)inout_shared, r15, r27, r46, r1);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           0 * out_calib_njtr_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = fmaf(r6, r6, r40 * r40);
    r46 = fmaf(r24, r24, r43 * r43);
    r27 = fmaf(r44, r44, r0 * r0);
    r15 = fmaf(r31, r31, r38 * r38);
    WriteSum4<float, float>((float*)inout_shared, r1, r46, r27, r15);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           0 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r15 = fmaf(r40, r43, r6 * r24);
    r27 = fmaf(r40, r0, r6 * r44);
    r6 = fmaf(r40, r31, r6 * r38);
    r46 = fmaf(r24, r44, r43 * r0);
    WriteSum4<float, float>((float*)inout_shared, r15, r27, r6, r46);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           0 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = fmaf(r43, r31, r24 * r38);
    r38 = fmaf(r0, r31, r44 * r38);
    WriteSum2<float, float>((float*)inout_shared, r24, r38);
  };
  FlushSumShared<2, float>(out_calib_precond_tril,
                           4 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void PinholeFixedPoseFixedPointResJacFirst(
    float* calib,
    unsigned int calib_num_alloc,
    SharedIndex* calib_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* weight_loss,
    unsigned int weight_loss_num_alloc,
    float* pose,
    unsigned int pose_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
    float* const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc,
    float* const out_calib_precond_diag,
    unsigned int out_calib_precond_diag_num_alloc,
    float* const out_calib_precond_tril,
    unsigned int out_calib_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeFixedPoseFixedPointResJacFirstKernel<<<n_blocks, 1024>>>(
      calib,
      calib_num_alloc,
      calib_indices,
      pixel,
      pixel_num_alloc,
      weight_loss,
      weight_loss_num_alloc,
      pose,
      pose_num_alloc,
      point,
      point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_rTr,
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      out_calib_precond_diag,
      out_calib_precond_diag_num_alloc,
      out_calib_precond_tril,
      out_calib_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar