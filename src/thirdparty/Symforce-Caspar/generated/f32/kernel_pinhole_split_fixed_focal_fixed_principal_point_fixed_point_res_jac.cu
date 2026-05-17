#include "kernel_pinhole_split_fixed_focal_fixed_principal_point_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedFocalFixedPrincipalPointFixedPointResJacKernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* weight_loss,
        unsigned int weight_loss_num_alloc,
        float* focal,
        unsigned int focal_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
        float* point,
        unsigned int point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
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
    ReadIdx2<1024, float, float, float2>(
        focal, 0 * focal_num_alloc, global_thread_idx, r31, r32);
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
    r9 = r32 * r9;
    r7 = fmaf(r10, r9, r7);
    r27 = fmaf(r1, r7, r0 * r6);
    ReadIdx3<1024, float, float, float4>(weight_loss,
                                         4 * weight_loss_num_alloc,
                                         global_thread_idx,
                                         r39,
                                         r42,
                                         r43);
    r44 = 0.00000000000000000e+00;
    r43 = fmaxf(r43, r44);
    r45 = sqrtf(r43);
    r7 = fmaf(r3, r7, r2 * r6);
    r6 = fmaf(r27, r27, r7 * r7);
    r46 = 5.00000000000000000e-01;
    r42 = fmaxf(r42, r33);
    r47 = r42 * r42;
    r48 = r19 * r42;
    r49 = fmaxf(r33, r6);
    r50 = sqrtf(r49);
    r51 = r8 * r42;
    r51 = fmaf(r42, r51, r50 * r48);
    r51 = r6 <= r47 ? r6 : r51;
    r48 = 2.50000000000000000e+00;
    r50 = r42 * r42;
    r52 = 1.0 / r47;
    r52 = fmaf(r6, r52, r28);
    r53 = logf(r52);
    r50 = r50 * r53;
    r51 = r39 < r48 ? r50 : r51;
    r50 = 1.50000000000000000e+00;
    r53 = r19 * r42;
    r54 = sqrtf(r52);
    r54 = r8 + r54;
    r53 = r53 * r42;
    r53 = r53 * r54;
    r51 = r39 < r50 ? r53 : r51;
    r51 = r39 < r46 ? r6 : r51;
    r53 = fmaxf(r44, r51);
    r54 = 1.0 / r49;
    r54 = r43 * r54;
    r55 = r53 * r54;
    r56 = sqrtf(r55);
    r56 = r6 <= r33 ? r45 : r56;
    r45 = r27 * r56;
    r57 = r7 * r56;
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r45, r57);
    r57 = r8 * r7;
    r45 = 2.50000000000000000e-01;
    r58 = r6 <= r47 ? r44 : r44;
    r58 = r39 < r48 ? r44 : r58;
    r58 = r39 < r50 ? r44 : r58;
    r58 = r39 < r46 ? r44 : r58;
    r58 = r45 * r58;
    r55 = rsqrtf(r55);
    r51 = copysign(1.0, r51);
    r51 = r28 + r51;
    r54 = r51 * r54;
    r58 = r58 * r55;
    r58 = r58 * r54;
    r58 = r6 <= r33 ? r44 : r58;
    r57 = r57 * r58;
    r53 = r43 * r53;
    r43 = -5.00000000000000000e-01;
    r51 = -9.99999999999999955e-07;
    r51 = r51 + r6;
    r51 = copysign(1.0, r51);
    r51 = r28 + r51;
    r28 = r49 * r49;
    r28 = 1.0 / r28;
    r53 = r53 * r43;
    r53 = r53 * r51;
    r53 = r53 * r28;
    r28 = r19 * r27;
    r43 = r16 * r16;
    r17 = r17 * r17;
    r45 = r8 * r17;
    r59 = r43 + r45;
    r60 = r8 * r29;
    r61 = r38 + r60;
    r62 = r59 + r61;
    r62 = fmaf(r13, r62, r12 * r34);
    r34 = r32 * r62;
    r63 = r15 * r21;
    r41 = r41 + r63;
    r60 = r43 + r60;
    r43 = r8 * r38;
    r64 = r17 + r43;
    r60 = r60 + r64;
    r60 = fmaf(r12, r60, r13 * r41);
    r41 = r8 * r60;
    r30 = r30 * r30;
    r30 = 1.0 / r30;
    r41 = r41 * r30;
    r41 = fmaf(r9, r41, r10 * r34);
    r15 = r14 * r15;
    r15 = r15 * r20;
    r5 = r5 + r15;
    r26 = fmaf(r12, r26, r13 * r5);
    r4 = r31 * r4;
    r4 = r4 * r8;
    r4 = r4 * r30;
    r26 = fmaf(r60, r4, r26 * r40);
    r31 = fmaf(r0, r26, r1 * r41);
    r26 = fmaf(r2, r26, r3 * r41);
    r41 = r19 * r7;
    r28 = fmaf(r26, r41, r31 * r28);
    r5 = r46 * r42;
    r49 = rsqrtf(r49);
    r5 = r5 * r51;
    r5 = r5 * r49;
    r49 = r28 * r5;
    r49 = r6 <= r47 ? r28 : r49;
    r51 = 1.0 / r52;
    r20 = r28 * r51;
    r49 = r39 < r48 ? r20 : r49;
    r52 = rsqrtf(r52);
    r20 = r28 * r52;
    r49 = r39 < r50 ? r20 : r49;
    r49 = r39 < r46 ? r28 : r49;
    r20 = r46 * r49;
    r20 = fmaf(r54, r20, r28 * r53);
    r20 = r46 * r20;
    r20 = r20 * r55;
    r20 = r6 <= r33 ? r44 : r20;
    r28 = fmaf(r7, r20, r57);
    r28 = fmaf(r56, r26, r28);
    r26 = r7 * r28;
    r34 = r8 * r56;
    r65 = r8 * r27;
    r65 = r65 * r58;
    r20 = fmaf(r27, r20, r65);
    r20 = fmaf(r56, r31, r20);
    r31 = r27 * r34;
    r26 = fmaf(r20, r31, r34 * r26);
    r21 = r14 * r21;
    r37 = r37 + r21;
    r16 = r16 * r16;
    r16 = r16 * r8;
    r17 = r17 + r16;
    r17 = r17 + r61;
    r17 = fmaf(r13, r17, r11 * r37);
    r45 = r38 + r45;
    r16 = r29 + r16;
    r45 = r45 + r16;
    r45 = fmaf(r11, r45, r13 * r24);
    r17 = fmaf(r45, r4, r17 * r40);
    r63 = r35 + r63;
    r63 = fmaf(r11, r63, r13 * r18);
    r18 = r32 * r63;
    r13 = r8 * r45;
    r13 = r13 * r30;
    r13 = fmaf(r9, r13, r10 * r18);
    r18 = fmaf(r1, r13, r0 * r17);
    r35 = fmaf(r56, r18, r65);
    r24 = r19 * r27;
    r13 = fmaf(r3, r13, r2 * r17);
    r24 = fmaf(r13, r41, r18 * r24);
    r18 = r24 * r5;
    r18 = r6 <= r47 ? r24 : r18;
    r17 = r24 * r51;
    r18 = r39 < r48 ? r17 : r18;
    r17 = r24 * r52;
    r18 = r39 < r50 ? r17 : r18;
    r18 = r39 < r46 ? r24 : r18;
    r17 = r46 * r18;
    r24 = fmaf(r24, r53, r54 * r17);
    r24 = r46 * r24;
    r24 = r24 * r55;
    r24 = r6 <= r33 ? r44 : r24;
    r35 = fmaf(r27, r24, r35);
    r13 = fmaf(r56, r13, r57);
    r13 = fmaf(r7, r24, r13);
    r24 = r7 * r13;
    r24 = fmaf(r34, r24, r35 * r31);
    r17 = r19 * r27;
    r21 = r25 + r21;
    r21 = fmaf(r12, r21, r11 * r36);
    r36 = r8 * r21;
    r36 = r36 * r30;
    r22 = r15 + r22;
    r16 = r64 + r16;
    r16 = fmaf(r11, r16, r12 * r22);
    r22 = r32 * r16;
    r22 = fmaf(r10, r22, r9 * r36);
    r43 = r29 + r43;
    r43 = r43 + r59;
    r43 = fmaf(r12, r43, r11 * r23);
    r43 = fmaf(r21, r4, r43 * r40);
    r12 = fmaf(r0, r43, r1 * r22);
    r43 = fmaf(r2, r43, r3 * r22);
    r17 = fmaf(r43, r41, r12 * r17);
    r22 = r17 * r5;
    r22 = r6 <= r47 ? r17 : r22;
    r23 = r17 * r51;
    r22 = r39 < r48 ? r23 : r22;
    r23 = r17 * r52;
    r22 = r39 < r50 ? r23 : r22;
    r22 = r39 < r46 ? r17 : r22;
    r23 = r46 * r22;
    r17 = fmaf(r17, r53, r54 * r23);
    r17 = r46 * r17;
    r17 = r17 * r55;
    r17 = r6 <= r33 ? r44 : r17;
    r23 = fmaf(r27, r17, r65);
    r23 = fmaf(r56, r12, r23);
    r17 = fmaf(r7, r17, r57);
    r17 = fmaf(r56, r43, r17);
    r43 = r7 * r17;
    r43 = fmaf(r34, r43, r23 * r31);
    r12 = r2 * r40;
    r11 = r0 * r19;
    r11 = r11 * r27;
    r11 = fmaf(r40, r11, r41 * r12);
    r59 = r11 * r5;
    r59 = r6 <= r47 ? r11 : r59;
    r29 = r11 * r51;
    r59 = r39 < r48 ? r29 : r59;
    r29 = r11 * r52;
    r59 = r39 < r50 ? r29 : r59;
    r59 = r39 < r46 ? r11 : r59;
    r29 = r46 * r59;
    r29 = fmaf(r54, r29, r11 * r53);
    r29 = r46 * r29;
    r29 = r29 * r55;
    r29 = r6 <= r33 ? r44 : r29;
    r11 = fmaf(r7, r29, r57);
    r11 = fmaf(r56, r12, r11);
    r12 = r7 * r11;
    r29 = fmaf(r27, r29, r65);
    r36 = r0 * r56;
    r29 = fmaf(r40, r36, r29);
    r12 = fmaf(r29, r31, r34 * r12);
    WriteSum4<float, float>((float*)inout_shared, r26, r24, r43, r12);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r12 = r3 * r32;
    r12 = r12 * r10;
    r43 = r1 * r32;
    r43 = r43 * r19;
    r43 = r43 * r27;
    r43 = fmaf(r10, r43, r41 * r12);
    r12 = r43 * r5;
    r12 = r6 <= r47 ? r43 : r12;
    r24 = r43 * r51;
    r12 = r39 < r48 ? r24 : r12;
    r24 = r43 * r52;
    r12 = r39 < r50 ? r24 : r12;
    r12 = r39 < r46 ? r43 : r12;
    r24 = r46 * r12;
    r24 = fmaf(r54, r24, r43 * r53);
    r24 = r46 * r24;
    r24 = r24 * r55;
    r24 = r6 <= r33 ? r44 : r24;
    r43 = fmaf(r27, r24, r65);
    r26 = r1 * r32;
    r26 = r26 * r56;
    r43 = fmaf(r10, r26, r43);
    r24 = fmaf(r7, r24, r57);
    r26 = r3 * r32;
    r26 = r26 * r56;
    r24 = fmaf(r10, r26, r24);
    r26 = r7 * r24;
    r26 = fmaf(r34, r26, r43 * r31);
    r10 = r3 * r8;
    r10 = r10 * r30;
    r10 = fmaf(r9, r10, r2 * r4);
    r57 = fmaf(r56, r10, r57);
    r2 = r19 * r27;
    r36 = r1 * r8;
    r36 = r36 * r30;
    r36 = fmaf(r9, r36, r0 * r4);
    r41 = fmaf(r10, r41, r36 * r2);
    r5 = r41 * r5;
    r5 = r6 <= r47 ? r41 : r5;
    r51 = r41 * r51;
    r5 = r39 < r48 ? r51 : r5;
    r52 = r41 * r52;
    r5 = r39 < r50 ? r52 : r5;
    r5 = r39 < r46 ? r41 : r5;
    r39 = r46 * r5;
    r53 = fmaf(r41, r53, r54 * r39);
    r53 = r46 * r53;
    r53 = r53 * r55;
    r53 = r6 <= r33 ? r44 : r53;
    r57 = fmaf(r7, r53, r57);
    r44 = r7 * r57;
    r53 = fmaf(r27, r53, r65);
    r53 = fmaf(r56, r36, r53);
    r31 = fmaf(r53, r31, r34 * r44);
    WriteSum2<float, float>((float*)inout_shared, r26, r31);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = fmaf(r20, r20, r28 * r28);
    r26 = fmaf(r35, r35, r13 * r13);
    r44 = fmaf(r23, r23, r17 * r17);
    r34 = fmaf(r11, r11, r29 * r29);
    WriteSum4<float, float>((float*)inout_shared, r31, r26, r44, r34);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = fmaf(r24, r24, r43 * r43);
    r44 = fmaf(r57, r57, r53 * r53);
    WriteSum2<float, float>((float*)inout_shared, r34, r44);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r44 = fmaf(r28, r13, r20 * r35);
    r34 = fmaf(r28, r17, r20 * r23);
    r26 = fmaf(r20, r29, r28 * r11);
    r31 = fmaf(r20, r43, r28 * r24);
    WriteSum4<float, float>((float*)inout_shared, r44, r34, r26, r31);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r20 = fmaf(r28, r57, r20 * r53);
    r31 = fmaf(r13, r17, r35 * r23);
    r26 = fmaf(r35, r29, r13 * r11);
    r34 = fmaf(r13, r24, r35 * r43);
    WriteSum4<float, float>((float*)inout_shared, r20, r31, r26, r34);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = fmaf(r35, r53, r13 * r57);
    r34 = fmaf(r23, r29, r17 * r11);
    r26 = fmaf(r23, r43, r17 * r24);
    r23 = fmaf(r23, r53, r17 * r57);
    WriteSum4<float, float>((float*)inout_shared, r35, r34, r26, r23);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = fmaf(r29, r43, r11 * r24);
    r29 = fmaf(r29, r53, r11 * r57);
    r53 = fmaf(r43, r53, r24 * r57);
    WriteSum3<float, float>((float*)inout_shared, r23, r29, r53);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
}

void PinholeSplitFixedFocalFixedPrincipalPointFixedPointResJac(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* weight_loss,
    unsigned int weight_loss_num_alloc,
    float* focal,
    unsigned int focal_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
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
  PinholeSplitFixedFocalFixedPrincipalPointFixedPointResJacKernel<<<n_blocks,
                                                                    1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      pixel,
      pixel_num_alloc,
      weight_loss,
      weight_loss_num_alloc,
      focal,
      focal_num_alloc,
      principal_point,
      principal_point_num_alloc,
      point,
      point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_pose_precond_diag,
      out_pose_precond_diag_num_alloc,
      out_pose_precond_tril,
      out_pose_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar