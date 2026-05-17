#include "kernel_pinhole_split_fixed_principal_point_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedPrincipalPointFixedPointResJacKernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* focal,
        unsigned int focal_num_alloc,
        SharedIndex* focal_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* weight_loss,
        unsigned int weight_loss_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
        float* point,
        unsigned int point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* out_pose_jac,
        unsigned int out_pose_jac_num_alloc,
        float* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        float* const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        float* const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
        float* out_focal_jac,
        unsigned int out_focal_jac_num_alloc,
        float* const out_focal_njtr,
        unsigned int out_focal_njtr_num_alloc,
        float* const out_focal_precond_diag,
        unsigned int out_focal_precond_diag_num_alloc,
        float* const out_focal_precond_tril,
        unsigned int out_focal_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex focal_indices_loc[1024];
  focal_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65;

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
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r4,
                       r9,
                       r10);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r11, r12, r13);
  };
  LoadShared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r14,
                       r15,
                       r16,
                       r17);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r18 = r14 * r15;
    r19 = 2.00000000000000000e+00;
    r18 = r18 * r19;
    r20 = -2.00000000000000000e+00;
    r21 = r16 * r20;
    r22 = r17 * r21;
    r23 = r18 + r22;
    r4 = fmaf(r12, r23, r4);
    r24 = r14 * r16;
    r24 = r24 * r19;
    r25 = r15 * r17;
    r25 = r25 * r19;
    r26 = r24 + r25;
    r27 = r16 * r21;
    r28 = 1.00000000000000000e+00;
    r29 = r15 * r15;
    r30 = fmaf(r20, r29, r28);
    r31 = r27 + r30;
    r4 = fmaf(r13, r26, r4);
    r4 = fmaf(r11, r31, r4);
  };
  LoadShared<2, float, float>(
      focal, 0 * focal_num_alloc, focal_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>(
        (float*)inout_shared, focal_indices_loc[threadIdx.x].target, r31, r32);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r33 = 9.99999999999999955e-07;
    r34 = r15 * r16;
    r34 = r34 * r19;
    r35 = r14 * r17;
    r35 = r35 * r19;
    r36 = r34 + r35;
    r10 = fmaf(r12, r36, r10);
    r37 = r15 * r17;
    r37 = r37 * r20;
    r24 = r24 + r37;
    r38 = r14 * r14;
    r39 = r20 * r38;
    r30 = r39 + r30;
    r10 = fmaf(r11, r24, r10);
    r10 = fmaf(r13, r30, r10);
    r30 = copysign(1.0, r10);
    r30 = fmaf(r33, r30, r10);
    r10 = 1.0 / r30;
    r40 = r31 * r10;
    r6 = fmaf(r4, r40, r6);
    r7 = fmaf(r7, r8, r5);
    r5 = r16 * r17;
    r5 = r5 * r19;
    r18 = r18 + r5;
    r9 = fmaf(r11, r18, r9);
    r41 = r14 * r17;
    r41 = r41 * r20;
    r34 = r34 + r41;
    r27 = r28 + r27;
    r27 = r27 + r39;
    r9 = fmaf(r13, r34, r9);
    r9 = fmaf(r12, r27, r9);
    r27 = r32 * r9;
    r7 = fmaf(r10, r27, r7);
    r39 = fmaf(r1, r7, r0 * r6);
    ReadIdx3<1024, float, float, float4>(weight_loss,
                                         4 * weight_loss_num_alloc,
                                         global_thread_idx,
                                         r42,
                                         r43,
                                         r44);
    r45 = 0.00000000000000000e+00;
    r44 = fmaxf(r44, r45);
    r46 = sqrtf(r44);
    r7 = fmaf(r3, r7, r2 * r6);
    r6 = fmaf(r39, r39, r7 * r7);
    r47 = 5.00000000000000000e-01;
    r43 = fmaxf(r43, r33);
    r48 = r43 * r43;
    r49 = r19 * r43;
    r50 = fmaxf(r33, r6);
    r51 = sqrtf(r50);
    r52 = r8 * r43;
    r52 = fmaf(r43, r52, r51 * r49);
    r52 = r6 <= r48 ? r6 : r52;
    r49 = 2.50000000000000000e+00;
    r51 = r43 * r43;
    r53 = 1.0 / r48;
    r53 = fmaf(r6, r53, r28);
    r54 = logf(r53);
    r51 = r51 * r54;
    r52 = r42 < r49 ? r51 : r52;
    r51 = 1.50000000000000000e+00;
    r54 = r19 * r43;
    r55 = sqrtf(r53);
    r55 = r8 + r55;
    r54 = r54 * r43;
    r54 = r54 * r55;
    r52 = r42 < r51 ? r54 : r52;
    r52 = r42 < r47 ? r6 : r52;
    r54 = fmaxf(r45, r52);
    r55 = 1.0 / r50;
    r55 = r44 * r55;
    r56 = r54 * r55;
    r57 = sqrtf(r56);
    r57 = r6 <= r33 ? r46 : r57;
    r46 = r39 * r57;
    r58 = r7 * r57;
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r46, r58);
    r58 = r8 * r39;
    r46 = 2.50000000000000000e-01;
    r59 = r6 <= r48 ? r45 : r45;
    r59 = r42 < r49 ? r45 : r59;
    r59 = r42 < r51 ? r45 : r59;
    r59 = r42 < r47 ? r45 : r59;
    r59 = r46 * r59;
    r56 = rsqrtf(r56);
    r52 = copysign(1.0, r52);
    r52 = r28 + r52;
    r55 = r52 * r55;
    r59 = r59 * r56;
    r59 = r59 * r55;
    r59 = r6 <= r33 ? r45 : r59;
    r58 = r58 * r59;
    r54 = r44 * r54;
    r44 = -5.00000000000000000e-01;
    r52 = -9.99999999999999955e-07;
    r52 = r52 + r6;
    r52 = copysign(1.0, r52);
    r52 = r28 + r52;
    r28 = r50 * r50;
    r28 = 1.0 / r28;
    r54 = r54 * r44;
    r54 = r54 * r52;
    r54 = r54 * r28;
    r28 = r19 * r39;
    r44 = r16 * r16;
    r17 = r17 * r17;
    r46 = r8 * r17;
    r60 = r44 + r46;
    r61 = r8 * r29;
    r62 = r38 + r61;
    r63 = r60 + r62;
    r63 = fmaf(r13, r63, r12 * r34);
    r34 = r32 * r63;
    r64 = r15 * r21;
    r41 = r41 + r64;
    r61 = r44 + r61;
    r44 = r8 * r38;
    r65 = r17 + r44;
    r61 = r61 + r65;
    r61 = fmaf(r12, r61, r13 * r41);
    r41 = r8 * r61;
    r30 = r30 * r30;
    r30 = 1.0 / r30;
    r41 = r41 * r30;
    r41 = fmaf(r27, r41, r10 * r34);
    r15 = r14 * r15;
    r15 = r15 * r20;
    r5 = r5 + r15;
    r26 = fmaf(r12, r26, r13 * r5);
    r31 = r31 * r4;
    r31 = r31 * r8;
    r31 = r31 * r30;
    r26 = fmaf(r61, r31, r26 * r40);
    r5 = fmaf(r0, r26, r1 * r41);
    r26 = fmaf(r2, r26, r3 * r41);
    r41 = r19 * r7;
    r28 = fmaf(r26, r41, r5 * r28);
    r20 = r47 * r43;
    r50 = rsqrtf(r50);
    r20 = r20 * r52;
    r20 = r20 * r50;
    r50 = r28 * r20;
    r50 = r6 <= r48 ? r28 : r50;
    r52 = 1.0 / r53;
    r34 = r28 * r52;
    r50 = r42 < r49 ? r34 : r50;
    r53 = rsqrtf(r53);
    r34 = r28 * r53;
    r50 = r42 < r51 ? r34 : r50;
    r50 = r42 < r47 ? r28 : r50;
    r34 = r47 * r50;
    r34 = fmaf(r55, r34, r28 * r54);
    r34 = r47 * r34;
    r34 = r34 * r56;
    r34 = r6 <= r33 ? r45 : r34;
    r28 = fmaf(r39, r34, r58);
    r28 = fmaf(r57, r5, r28);
    r5 = r8 * r7;
    r5 = r5 * r59;
    r34 = fmaf(r7, r34, r5);
    r34 = fmaf(r57, r26, r34);
    r21 = r14 * r21;
    r37 = r37 + r21;
    r16 = r16 * r16;
    r16 = r16 * r8;
    r17 = r17 + r16;
    r17 = r17 + r62;
    r17 = fmaf(r13, r17, r11 * r37);
    r46 = r38 + r46;
    r16 = r29 + r16;
    r46 = r46 + r16;
    r46 = fmaf(r11, r46, r13 * r24);
    r17 = fmaf(r46, r31, r17 * r40);
    r64 = r35 + r64;
    r64 = fmaf(r11, r64, r13 * r18);
    r18 = r32 * r64;
    r13 = r8 * r46;
    r13 = r13 * r30;
    r13 = fmaf(r27, r13, r10 * r18);
    r18 = fmaf(r1, r13, r0 * r17);
    r35 = fmaf(r57, r18, r58);
    r24 = r19 * r39;
    r13 = fmaf(r3, r13, r2 * r17);
    r24 = fmaf(r13, r41, r18 * r24);
    r18 = r24 * r20;
    r18 = r6 <= r48 ? r24 : r18;
    r17 = r24 * r52;
    r18 = r42 < r49 ? r17 : r18;
    r17 = r24 * r53;
    r18 = r42 < r51 ? r17 : r18;
    r18 = r42 < r47 ? r24 : r18;
    r17 = r47 * r18;
    r24 = fmaf(r24, r54, r55 * r17);
    r24 = r47 * r24;
    r24 = r24 * r56;
    r24 = r6 <= r33 ? r45 : r24;
    r35 = fmaf(r39, r24, r35);
    r13 = fmaf(r57, r13, r5);
    r13 = fmaf(r7, r24, r13);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r28,
                                          r34,
                                          r35,
                                          r13);
    r24 = r19 * r39;
    r21 = r25 + r21;
    r21 = fmaf(r12, r21, r11 * r36);
    r36 = r8 * r21;
    r36 = r36 * r30;
    r22 = r15 + r22;
    r16 = r65 + r16;
    r16 = fmaf(r11, r16, r12 * r22);
    r22 = r32 * r16;
    r22 = fmaf(r10, r22, r27 * r36);
    r44 = r29 + r44;
    r44 = r44 + r60;
    r44 = fmaf(r12, r44, r11 * r23);
    r44 = fmaf(r21, r31, r44 * r40);
    r12 = fmaf(r0, r44, r1 * r22);
    r44 = fmaf(r2, r44, r3 * r22);
    r24 = fmaf(r44, r41, r12 * r24);
    r22 = r24 * r20;
    r22 = r6 <= r48 ? r24 : r22;
    r23 = r24 * r52;
    r22 = r42 < r49 ? r23 : r22;
    r23 = r24 * r53;
    r22 = r42 < r51 ? r23 : r22;
    r22 = r42 < r47 ? r24 : r22;
    r23 = r47 * r22;
    r24 = fmaf(r24, r54, r55 * r23);
    r24 = r47 * r24;
    r24 = r24 * r56;
    r24 = r6 <= r33 ? r45 : r24;
    r23 = fmaf(r39, r24, r58);
    r23 = fmaf(r57, r12, r23);
    r24 = fmaf(r7, r24, r5);
    r24 = fmaf(r57, r44, r24);
    r44 = r2 * r40;
    r12 = r0 * r19;
    r12 = r12 * r39;
    r12 = fmaf(r40, r12, r41 * r44);
    r11 = r12 * r20;
    r11 = r6 <= r48 ? r12 : r11;
    r60 = r12 * r52;
    r11 = r42 < r49 ? r60 : r11;
    r60 = r12 * r53;
    r11 = r42 < r51 ? r60 : r11;
    r11 = r42 < r47 ? r12 : r11;
    r60 = r47 * r11;
    r60 = fmaf(r55, r60, r12 * r54);
    r60 = r47 * r60;
    r60 = r60 * r56;
    r60 = r6 <= r33 ? r45 : r60;
    r12 = fmaf(r39, r60, r58);
    r29 = r0 * r57;
    r12 = fmaf(r40, r29, r12);
    r60 = fmaf(r7, r60, r5);
    r60 = fmaf(r57, r44, r60);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r23,
                                          r24,
                                          r12,
                                          r60);
    r44 = r19 * r39;
    r29 = r1 * r8;
    r29 = r29 * r30;
    r29 = fmaf(r27, r29, r0 * r31);
    r40 = r3 * r8;
    r40 = r40 * r30;
    r40 = fmaf(r27, r40, r2 * r31);
    r44 = fmaf(r40, r41, r29 * r44);
    r31 = r44 * r20;
    r31 = r6 <= r48 ? r44 : r31;
    r27 = r44 * r52;
    r31 = r42 < r49 ? r27 : r31;
    r27 = r44 * r53;
    r31 = r42 < r51 ? r27 : r31;
    r31 = r42 < r47 ? r44 : r31;
    r27 = r47 * r31;
    r44 = fmaf(r44, r54, r55 * r27);
    r44 = r47 * r44;
    r44 = r44 * r56;
    r44 = r6 <= r33 ? r45 : r44;
    r27 = fmaf(r39, r44, r58);
    r27 = fmaf(r57, r29, r27);
    r40 = fmaf(r57, r40, r5);
    r40 = fmaf(r7, r44, r40);
    r41 = r10 * r41;
    r44 = r3 * r41;
    r29 = r32 * r1;
    r29 = r29 * r19;
    r29 = r29 * r39;
    r29 = fmaf(r10, r29, r32 * r44);
    r30 = r29 * r20;
    r30 = r6 <= r48 ? r29 : r30;
    r36 = r29 * r52;
    r30 = r42 < r49 ? r36 : r30;
    r36 = r29 * r53;
    r30 = r42 < r51 ? r36 : r30;
    r30 = r42 < r47 ? r29 : r30;
    r36 = r47 * r30;
    r36 = fmaf(r55, r36, r29 * r54);
    r36 = r47 * r36;
    r36 = r36 * r56;
    r36 = r6 <= r33 ? r45 : r36;
    r29 = fmaf(r39, r36, r58);
    r65 = r32 * r1;
    r65 = r65 * r57;
    r29 = fmaf(r10, r65, r29);
    r36 = fmaf(r7, r36, r5);
    r65 = r32 * r3;
    r65 = r65 * r57;
    r36 = fmaf(r10, r65, r36);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r29,
                                          r36,
                                          r27,
                                          r40);
    r65 = r7 * r34;
    r15 = r8 * r57;
    r25 = r39 * r15;
    r65 = fmaf(r28, r25, r15 * r65);
    r17 = r7 * r13;
    r17 = fmaf(r15, r17, r35 * r25);
    r38 = r7 * r24;
    r38 = fmaf(r15, r38, r23 * r25);
    r37 = r7 * r60;
    r37 = fmaf(r12, r25, r15 * r37);
    WriteSum4<float, float>((float*)inout_shared, r65, r17, r38, r37);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = r7 * r36;
    r37 = fmaf(r15, r37, r29 * r25);
    r38 = r7 * r40;
    r38 = fmaf(r27, r25, r15 * r38);
    WriteSum2<float, float>((float*)inout_shared, r37, r38);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r38 = fmaf(r28, r28, r34 * r34);
    r37 = fmaf(r35, r35, r13 * r13);
    r17 = fmaf(r23, r23, r24 * r24);
    r65 = fmaf(r60, r60, r12 * r12);
    WriteSum4<float, float>((float*)inout_shared, r38, r37, r17, r65);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r65 = fmaf(r36, r36, r29 * r29);
    r17 = fmaf(r40, r40, r27 * r27);
    WriteSum2<float, float>((float*)inout_shared, r65, r17);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r17 = fmaf(r34, r13, r28 * r35);
    r65 = fmaf(r34, r24, r28 * r23);
    r37 = fmaf(r28, r12, r34 * r60);
    r38 = fmaf(r28, r29, r34 * r36);
    WriteSum4<float, float>((float*)inout_shared, r17, r65, r37, r38);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r28 = fmaf(r34, r40, r28 * r27);
    r38 = fmaf(r13, r24, r35 * r23);
    r37 = fmaf(r35, r12, r13 * r60);
    r65 = fmaf(r13, r36, r35 * r29);
    WriteSum4<float, float>((float*)inout_shared, r28, r38, r37, r65);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = fmaf(r35, r27, r13 * r40);
    r65 = fmaf(r23, r12, r24 * r60);
    r37 = fmaf(r23, r29, r24 * r36);
    r23 = fmaf(r23, r27, r24 * r40);
    WriteSum4<float, float>((float*)inout_shared, r35, r65, r37, r23);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = fmaf(r12, r29, r60 * r36);
    r12 = fmaf(r12, r27, r60 * r40);
    r27 = fmaf(r29, r27, r36 * r40);
    WriteSum3<float, float>((float*)inout_shared, r23, r12, r27);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = r0 * r19;
    r27 = r27 * r4;
    r27 = r27 * r39;
    r12 = r2 * r4;
    r12 = fmaf(r41, r12, r10 * r27);
    r27 = r12 * r20;
    r27 = r6 <= r48 ? r12 : r27;
    r41 = r12 * r52;
    r27 = r42 < r49 ? r41 : r27;
    r41 = r12 * r53;
    r27 = r42 < r51 ? r41 : r27;
    r27 = r42 < r47 ? r12 : r27;
    r41 = r47 * r27;
    r12 = fmaf(r12, r54, r55 * r41);
    r12 = r47 * r12;
    r12 = r12 * r56;
    r12 = r6 <= r33 ? r45 : r12;
    r41 = fmaf(r39, r12, r58);
    r23 = r0 * r4;
    r23 = r23 * r57;
    r41 = fmaf(r10, r23, r41);
    r12 = fmaf(r7, r12, r5);
    r23 = r2 * r4;
    r23 = r23 * r57;
    r12 = fmaf(r10, r23, r12);
    r23 = r1 * r19;
    r23 = r23 * r9;
    r23 = r23 * r39;
    r44 = fmaf(r9, r44, r10 * r23);
    r20 = r44 * r20;
    r20 = r6 <= r48 ? r44 : r20;
    r52 = r44 * r52;
    r20 = r42 < r49 ? r52 : r20;
    r53 = r44 * r53;
    r20 = r42 < r51 ? r53 : r20;
    r20 = r42 < r47 ? r44 : r20;
    r42 = r47 * r20;
    r42 = fmaf(r55, r42, r44 * r54);
    r42 = r47 * r42;
    r42 = r42 * r56;
    r42 = r6 <= r33 ? r45 : r42;
    r58 = fmaf(r39, r42, r58);
    r45 = r1 * r9;
    r45 = r45 * r57;
    r58 = fmaf(r10, r45, r58);
    r42 = fmaf(r7, r42, r5);
    r5 = r3 * r9;
    r5 = r5 * r57;
    r42 = fmaf(r10, r5, r42);
    WriteIdx4<1024, float, float, float4>(out_focal_jac,
                                          0 * out_focal_jac_num_alloc,
                                          global_thread_idx,
                                          r41,
                                          r12,
                                          r58,
                                          r42);
    r5 = r7 * r12;
    r5 = fmaf(r41, r25, r15 * r5);
    r10 = r7 * r42;
    r10 = fmaf(r15, r10, r58 * r25);
    WriteSum2<float, float>((float*)inout_shared, r5, r10);
  };
  FlushSumShared<2, float>(out_focal_njtr,
                           0 * out_focal_njtr_num_alloc,
                           focal_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r10 = fmaf(r41, r41, r12 * r12);
    r5 = fmaf(r58, r58, r42 * r42);
    WriteSum2<float, float>((float*)inout_shared, r10, r5);
  };
  FlushSumShared<2, float>(out_focal_precond_diag,
                           0 * out_focal_precond_diag_num_alloc,
                           focal_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r58 = fmaf(r41, r58, r12 * r42);
    WriteSum1<float, float>((float*)inout_shared, r58);
  };
  FlushSumShared<1, float>(out_focal_precond_tril,
                           0 * out_focal_precond_tril_num_alloc,
                           focal_indices_loc,
                           (float*)inout_shared);
}

void PinholeSplitFixedPrincipalPointFixedPointResJac(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* focal,
    unsigned int focal_num_alloc,
    SharedIndex* focal_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* weight_loss,
    unsigned int weight_loss_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* out_pose_jac,
    unsigned int out_pose_jac_num_alloc,
    float* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    float* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    float* out_focal_jac,
    unsigned int out_focal_jac_num_alloc,
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
  PinholeSplitFixedPrincipalPointFixedPointResJacKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      focal,
      focal_num_alloc,
      focal_indices,
      pixel,
      pixel_num_alloc,
      weight_loss,
      weight_loss_num_alloc,
      principal_point,
      principal_point_num_alloc,
      point,
      point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_pose_jac,
      out_pose_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_pose_precond_diag,
      out_pose_precond_diag_num_alloc,
      out_pose_precond_tril,
      out_pose_precond_tril_num_alloc,
      out_focal_jac,
      out_focal_jac_num_alloc,
      out_focal_njtr,
      out_focal_njtr_num_alloc,
      out_focal_precond_diag,
      out_focal_precond_diag_num_alloc,
      out_focal_precond_tril,
      out_focal_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar