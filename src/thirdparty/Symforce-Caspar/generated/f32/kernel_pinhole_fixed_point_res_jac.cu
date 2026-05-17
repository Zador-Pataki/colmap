#include "kernel_pinhole_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeFixedPointResJacKernel(float* pose,
                                  unsigned int pose_num_alloc,
                                  SharedIndex* pose_indices,
                                  float* calib,
                                  unsigned int calib_num_alloc,
                                  SharedIndex* calib_indices,
                                  float* pixel,
                                  unsigned int pixel_num_alloc,
                                  float* weight_loss,
                                  unsigned int weight_loss_num_alloc,
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
                                  float* out_calib_jac,
                                  unsigned int out_calib_jac_num_alloc,
                                  float* const out_calib_njtr,
                                  unsigned int out_calib_njtr_num_alloc,
                                  float* const out_calib_precond_diag,
                                  unsigned int out_calib_precond_diag_num_alloc,
                                  float* const out_calib_precond_tril,
                                  unsigned int out_calib_precond_tril_num_alloc,
                                  size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex calib_indices_loc[1024];
  calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66;

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
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r6,
                       r11,
                       r12);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r13, r14, r15);
  };
  LoadShared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r16,
                       r17,
                       r18,
                       r19);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r20 = r16 * r17;
    r21 = 2.00000000000000000e+00;
    r20 = r20 * r21;
    r22 = -2.00000000000000000e+00;
    r23 = r18 * r22;
    r24 = r19 * r23;
    r25 = r20 + r24;
    r6 = fmaf(r14, r25, r6);
    r26 = r16 * r18;
    r26 = r26 * r21;
    r27 = r17 * r19;
    r27 = r27 * r21;
    r28 = r26 + r27;
    r29 = r18 * r23;
    r30 = 1.00000000000000000e+00;
    r31 = r17 * r17;
    r32 = fmaf(r22, r31, r30);
    r33 = r29 + r32;
    r6 = fmaf(r15, r28, r6);
    r6 = fmaf(r13, r33, r6);
    r33 = 9.99999999999999955e-07;
    r34 = r17 * r18;
    r34 = r34 * r21;
    r35 = r16 * r19;
    r35 = r35 * r21;
    r36 = r34 + r35;
    r12 = fmaf(r14, r36, r12);
    r37 = r17 * r19;
    r37 = r37 * r22;
    r26 = r26 + r37;
    r38 = r16 * r16;
    r39 = r22 * r38;
    r32 = r39 + r32;
    r12 = fmaf(r13, r26, r12);
    r12 = fmaf(r15, r32, r12);
    r32 = copysign(1.0, r12);
    r32 = fmaf(r33, r32, r12);
    r12 = 1.0 / r32;
    r40 = r4 * r12;
    r8 = fmaf(r6, r40, r8);
    r9 = fmaf(r9, r10, r7);
    r7 = r18 * r19;
    r7 = r7 * r21;
    r20 = r20 + r7;
    r11 = fmaf(r13, r20, r11);
    r41 = r16 * r19;
    r41 = r41 * r22;
    r34 = r34 + r41;
    r29 = r30 + r29;
    r29 = r29 + r39;
    r11 = fmaf(r15, r34, r11);
    r11 = fmaf(r14, r29, r11);
    r29 = r5 * r11;
    r9 = fmaf(r12, r29, r9);
    r39 = fmaf(r1, r9, r0 * r8);
    ReadIdx3<1024, float, float, float4>(weight_loss,
                                         4 * weight_loss_num_alloc,
                                         global_thread_idx,
                                         r42,
                                         r43,
                                         r44);
    r45 = 0.00000000000000000e+00;
    r44 = fmaxf(r44, r45);
    r46 = sqrtf(r44);
    r9 = fmaf(r3, r9, r2 * r8);
    r8 = fmaf(r9, r9, r39 * r39);
    r47 = 5.00000000000000000e-01;
    r43 = fmaxf(r43, r33);
    r48 = r43 * r43;
    r49 = r21 * r43;
    r50 = fmaxf(r33, r8);
    r51 = sqrtf(r50);
    r52 = r10 * r43;
    r52 = fmaf(r43, r52, r51 * r49);
    r52 = r8 <= r48 ? r8 : r52;
    r49 = 2.50000000000000000e+00;
    r51 = r43 * r43;
    r53 = 1.0 / r48;
    r53 = fmaf(r8, r53, r30);
    r54 = logf(r53);
    r51 = r51 * r54;
    r52 = r42 < r49 ? r51 : r52;
    r51 = 1.50000000000000000e+00;
    r54 = r21 * r43;
    r55 = sqrtf(r53);
    r55 = r10 + r55;
    r54 = r54 * r43;
    r54 = r54 * r55;
    r52 = r42 < r51 ? r54 : r52;
    r52 = r42 < r47 ? r8 : r52;
    r54 = fmaxf(r45, r52);
    r55 = 1.0 / r50;
    r55 = r44 * r55;
    r56 = r54 * r55;
    r57 = sqrtf(r56);
    r57 = r8 <= r33 ? r46 : r57;
    r46 = r39 * r57;
    r58 = r9 * r57;
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r46, r58);
    r54 = r44 * r54;
    r44 = -5.00000000000000000e-01;
    r58 = -9.99999999999999955e-07;
    r58 = r58 + r8;
    r58 = copysign(1.0, r58);
    r58 = r30 + r58;
    r46 = r50 * r50;
    r46 = 1.0 / r46;
    r54 = r54 * r44;
    r54 = r54 * r58;
    r54 = r54 * r46;
    r46 = r21 * r9;
    r44 = r17 * r23;
    r41 = r41 + r44;
    r59 = r18 * r18;
    r60 = r10 * r31;
    r61 = r59 + r60;
    r19 = r19 * r19;
    r62 = r10 * r38;
    r63 = r19 + r62;
    r64 = r61 + r63;
    r64 = fmaf(r14, r64, r15 * r41);
    r41 = r10 * r64;
    r32 = r32 * r32;
    r32 = 1.0 / r32;
    r41 = r41 * r32;
    r65 = r10 * r19;
    r66 = r38 + r65;
    r61 = r61 + r66;
    r61 = fmaf(r15, r61, r14 * r34);
    r34 = r5 * r61;
    r34 = fmaf(r12, r34, r29 * r41);
    r17 = r16 * r17;
    r17 = r17 * r22;
    r7 = r7 + r17;
    r28 = fmaf(r14, r28, r15 * r7);
    r4 = r4 * r6;
    r4 = r4 * r10;
    r4 = r4 * r32;
    r28 = fmaf(r64, r4, r28 * r40);
    r7 = fmaf(r2, r28, r3 * r34);
    r28 = fmaf(r0, r28, r1 * r34);
    r34 = r21 * r39;
    r46 = fmaf(r28, r34, r7 * r46);
    r22 = r47 * r43;
    r50 = rsqrtf(r50);
    r22 = r22 * r58;
    r22 = r22 * r50;
    r50 = r46 * r22;
    r50 = r8 <= r48 ? r46 : r50;
    r58 = 1.0 / r53;
    r41 = r46 * r58;
    r50 = r42 < r49 ? r41 : r50;
    r53 = rsqrtf(r53);
    r41 = r46 * r53;
    r50 = r42 < r51 ? r41 : r50;
    r50 = r42 < r47 ? r46 : r50;
    r41 = r47 * r50;
    r52 = copysign(1.0, r52);
    r52 = r30 + r52;
    r55 = r52 * r55;
    r41 = fmaf(r55, r41, r46 * r54);
    r41 = r47 * r41;
    r56 = rsqrtf(r56);
    r41 = r41 * r56;
    r41 = r8 <= r33 ? r45 : r41;
    r28 = fmaf(r57, r28, r39 * r41);
    r46 = r10 * r39;
    r52 = 2.50000000000000000e-01;
    r30 = r8 <= r48 ? r45 : r45;
    r30 = r42 < r49 ? r45 : r30;
    r30 = r42 < r51 ? r45 : r30;
    r30 = r42 < r47 ? r45 : r30;
    r30 = r52 * r30;
    r30 = r30 * r56;
    r30 = r30 * r55;
    r30 = r8 <= r33 ? r45 : r30;
    r46 = r46 * r30;
    r28 = r28 + r46;
    r7 = fmaf(r57, r7, r9 * r41);
    r41 = r10 * r9;
    r41 = r41 * r30;
    r7 = r7 + r41;
    r23 = r16 * r23;
    r37 = r37 + r23;
    r19 = r38 + r19;
    r18 = r18 * r18;
    r18 = r18 * r10;
    r19 = r19 + r60;
    r19 = r19 + r18;
    r19 = fmaf(r15, r19, r13 * r37);
    r18 = r31 + r18;
    r66 = r66 + r18;
    r66 = fmaf(r13, r66, r15 * r26);
    r19 = fmaf(r66, r4, r19 * r40);
    r44 = r35 + r44;
    r44 = fmaf(r13, r44, r15 * r20);
    r20 = r5 * r44;
    r15 = r10 * r66;
    r15 = r15 * r32;
    r15 = fmaf(r29, r15, r12 * r20);
    r20 = fmaf(r1, r15, r0 * r19);
    r35 = fmaf(r57, r20, r46);
    r26 = r21 * r9;
    r15 = fmaf(r3, r15, r2 * r19);
    r20 = fmaf(r20, r34, r15 * r26);
    r26 = r20 * r22;
    r26 = r8 <= r48 ? r20 : r26;
    r19 = r20 * r58;
    r26 = r42 < r49 ? r19 : r26;
    r19 = r20 * r53;
    r26 = r42 < r51 ? r19 : r26;
    r26 = r42 < r47 ? r20 : r26;
    r19 = r47 * r26;
    r20 = fmaf(r20, r54, r55 * r19);
    r20 = r47 * r20;
    r20 = r20 * r56;
    r20 = r8 <= r33 ? r45 : r20;
    r35 = fmaf(r39, r20, r35);
    r15 = fmaf(r57, r15, r41);
    r15 = fmaf(r9, r20, r15);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r28,
                                          r7,
                                          r35,
                                          r15);
    r59 = r31 + r59;
    r59 = r59 + r62;
    r59 = r59 + r65;
    r59 = fmaf(r14, r59, r13 * r25);
    r23 = r27 + r23;
    r23 = fmaf(r14, r23, r13 * r36);
    r59 = fmaf(r23, r4, r59 * r40);
    r36 = r10 * r23;
    r36 = r36 * r32;
    r24 = r17 + r24;
    r18 = r63 + r18;
    r18 = fmaf(r13, r18, r14 * r24);
    r13 = r5 * r18;
    r13 = fmaf(r12, r13, r29 * r36);
    r36 = fmaf(r1, r13, r0 * r59);
    r24 = fmaf(r57, r36, r46);
    r14 = r21 * r9;
    r13 = fmaf(r3, r13, r2 * r59);
    r36 = fmaf(r36, r34, r13 * r14);
    r14 = r36 * r22;
    r14 = r8 <= r48 ? r36 : r14;
    r59 = r36 * r58;
    r14 = r42 < r49 ? r59 : r14;
    r59 = r36 * r53;
    r14 = r42 < r51 ? r59 : r14;
    r14 = r42 < r47 ? r36 : r14;
    r59 = r47 * r14;
    r36 = fmaf(r36, r54, r55 * r59);
    r36 = r47 * r36;
    r36 = r36 * r56;
    r36 = r8 <= r33 ? r45 : r36;
    r24 = fmaf(r39, r36, r24);
    r13 = fmaf(r57, r13, r41);
    r13 = fmaf(r9, r36, r13);
    r36 = r0 * r21;
    r36 = r36 * r39;
    r59 = r21 * r9;
    r63 = r2 * r40;
    r59 = fmaf(r63, r59, r40 * r36);
    r17 = r59 * r22;
    r17 = r8 <= r48 ? r59 : r17;
    r27 = r59 * r58;
    r17 = r42 < r49 ? r27 : r17;
    r27 = r59 * r53;
    r17 = r42 < r51 ? r27 : r17;
    r17 = r42 < r47 ? r59 : r17;
    r27 = r47 * r17;
    r27 = fmaf(r55, r27, r59 * r54);
    r27 = r47 * r27;
    r27 = r27 * r56;
    r27 = r8 <= r33 ? r45 : r27;
    r59 = fmaf(r39, r27, r46);
    r25 = r0 * r57;
    r59 = fmaf(r40, r25, r59);
    r27 = fmaf(r9, r27, r41);
    r27 = fmaf(r57, r63, r27);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r24,
                                          r13,
                                          r59,
                                          r27);
    r63 = r21 * r9;
    r25 = r3 * r10;
    r25 = r25 * r32;
    r25 = fmaf(r29, r25, r2 * r4);
    r40 = r1 * r10;
    r40 = r40 * r32;
    r40 = fmaf(r29, r40, r0 * r4);
    r63 = fmaf(r40, r34, r25 * r63);
    r4 = r63 * r22;
    r4 = r8 <= r48 ? r63 : r4;
    r29 = r63 * r58;
    r4 = r42 < r49 ? r29 : r4;
    r29 = r63 * r53;
    r4 = r42 < r51 ? r29 : r4;
    r4 = r42 < r47 ? r63 : r4;
    r29 = r47 * r4;
    r63 = fmaf(r63, r54, r55 * r29);
    r63 = r47 * r63;
    r63 = r63 * r56;
    r63 = r8 <= r33 ? r45 : r63;
    r29 = fmaf(r39, r63, r46);
    r29 = fmaf(r57, r40, r29);
    r25 = fmaf(r57, r25, r41);
    r25 = fmaf(r9, r63, r25);
    r63 = r5 * r1;
    r63 = r63 * r12;
    r40 = r5 * r3;
    r40 = r40 * r21;
    r40 = r40 * r9;
    r40 = fmaf(r12, r40, r34 * r63);
    r63 = r40 * r22;
    r63 = r8 <= r48 ? r40 : r63;
    r32 = r40 * r58;
    r63 = r42 < r49 ? r32 : r63;
    r32 = r40 * r53;
    r63 = r42 < r51 ? r32 : r63;
    r63 = r42 < r47 ? r40 : r63;
    r32 = r47 * r63;
    r32 = fmaf(r55, r32, r40 * r54);
    r32 = r47 * r32;
    r32 = r32 * r56;
    r32 = r8 <= r33 ? r45 : r32;
    r40 = fmaf(r39, r32, r46);
    r65 = r5 * r1;
    r65 = r65 * r57;
    r40 = fmaf(r12, r65, r40);
    r32 = fmaf(r9, r32, r41);
    r65 = r5 * r3;
    r65 = r65 * r57;
    r32 = fmaf(r12, r65, r32);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r40,
                                          r32,
                                          r29,
                                          r25);
    r65 = r10 * r57;
    r62 = r9 * r65;
    r31 = r39 * r28;
    r31 = fmaf(r65, r31, r7 * r62);
    r20 = r39 * r35;
    r20 = fmaf(r15, r62, r65 * r20);
    r19 = r39 * r24;
    r19 = fmaf(r65, r19, r13 * r62);
    r37 = r39 * r59;
    r37 = fmaf(r65, r37, r27 * r62);
    WriteSum4<float, float>((float*)inout_shared, r31, r20, r19, r37);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = r39 * r40;
    r37 = fmaf(r32, r62, r65 * r37);
    r19 = r39 * r29;
    r19 = fmaf(r65, r19, r25 * r62);
    WriteSum2<float, float>((float*)inout_shared, r37, r19);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = fmaf(r28, r28, r7 * r7);
    r37 = fmaf(r15, r15, r35 * r35);
    r20 = fmaf(r13, r13, r24 * r24);
    r31 = fmaf(r59, r59, r27 * r27);
    WriteSum4<float, float>((float*)inout_shared, r19, r37, r20, r31);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = fmaf(r32, r32, r40 * r40);
    r20 = fmaf(r25, r25, r29 * r29);
    WriteSum2<float, float>((float*)inout_shared, r31, r20);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r20 = fmaf(r7, r15, r28 * r35);
    r31 = fmaf(r28, r24, r7 * r13);
    r37 = fmaf(r28, r59, r7 * r27);
    r19 = fmaf(r28, r40, r7 * r32);
    WriteSum4<float, float>((float*)inout_shared, r20, r31, r37, r19);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = fmaf(r7, r25, r28 * r29);
    r19 = fmaf(r35, r24, r15 * r13);
    r37 = fmaf(r15, r27, r35 * r59);
    r31 = fmaf(r35, r40, r15 * r32);
    WriteSum4<float, float>((float*)inout_shared, r7, r19, r37, r31);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r15 = fmaf(r15, r25, r35 * r29);
    r31 = fmaf(r13, r27, r24 * r59);
    r37 = fmaf(r13, r32, r24 * r40);
    r13 = fmaf(r13, r25, r24 * r29);
    WriteSum4<float, float>((float*)inout_shared, r15, r31, r37, r13);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = fmaf(r59, r40, r27 * r32);
    r27 = fmaf(r59, r29, r27 * r25);
    r25 = fmaf(r40, r29, r32 * r25);
    WriteSum3<float, float>((float*)inout_shared, r13, r27, r25);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = r6 * r12;
    r27 = r2 * r21;
    r27 = r27 * r6;
    r27 = r27 * r9;
    r27 = fmaf(r12, r27, r36 * r25);
    r25 = r27 * r22;
    r25 = r8 <= r48 ? r27 : r25;
    r13 = r27 * r58;
    r25 = r42 < r49 ? r13 : r25;
    r13 = r27 * r53;
    r25 = r42 < r51 ? r13 : r25;
    r25 = r42 < r47 ? r27 : r25;
    r13 = r47 * r25;
    r13 = fmaf(r55, r13, r27 * r54);
    r13 = r47 * r13;
    r13 = r13 * r56;
    r13 = r8 <= r33 ? r45 : r13;
    r27 = fmaf(r39, r13, r46);
    r32 = r0 * r6;
    r32 = r32 * r57;
    r27 = fmaf(r12, r32, r27);
    r13 = fmaf(r9, r13, r41);
    r32 = r2 * r6;
    r32 = r32 * r57;
    r13 = fmaf(r12, r32, r13);
    r32 = r1 * r11;
    r32 = r32 * r12;
    r37 = r3 * r21;
    r37 = r37 * r11;
    r37 = r37 * r9;
    r37 = fmaf(r12, r37, r34 * r32);
    r32 = r37 * r22;
    r32 = r8 <= r48 ? r37 : r32;
    r31 = r37 * r58;
    r32 = r42 < r49 ? r31 : r32;
    r31 = r37 * r53;
    r32 = r42 < r51 ? r31 : r32;
    r32 = r42 < r47 ? r37 : r32;
    r31 = r47 * r32;
    r37 = fmaf(r37, r54, r55 * r31);
    r37 = r47 * r37;
    r37 = r37 * r56;
    r37 = r8 <= r33 ? r45 : r37;
    r31 = fmaf(r39, r37, r46);
    r15 = r1 * r11;
    r15 = r15 * r57;
    r31 = fmaf(r12, r15, r31);
    r37 = fmaf(r9, r37, r41);
    r15 = r3 * r11;
    r15 = r15 * r57;
    r37 = fmaf(r12, r15, r37);
    WriteIdx4<1024, float, float, float4>(out_calib_jac,
                                          0 * out_calib_jac_num_alloc,
                                          global_thread_idx,
                                          r27,
                                          r13,
                                          r31,
                                          r37);
    r15 = fmaf(r0, r57, r46);
    r19 = r2 * r21;
    r19 = fmaf(r9, r19, r36);
    r36 = r19 * r22;
    r36 = r8 <= r48 ? r19 : r36;
    r7 = r19 * r58;
    r36 = r42 < r49 ? r7 : r36;
    r7 = r19 * r53;
    r36 = r42 < r51 ? r7 : r36;
    r36 = r42 < r47 ? r19 : r36;
    r7 = r47 * r36;
    r7 = fmaf(r55, r7, r19 * r54);
    r7 = r47 * r7;
    r7 = r7 * r56;
    r7 = r8 <= r33 ? r45 : r7;
    r15 = fmaf(r39, r7, r15);
    r19 = fmaf(r2, r57, r41);
    r19 = fmaf(r9, r7, r19);
    r7 = r3 * r21;
    r34 = fmaf(r1, r34, r9 * r7);
    r22 = r34 * r22;
    r22 = r8 <= r48 ? r34 : r22;
    r58 = r34 * r58;
    r22 = r42 < r49 ? r58 : r22;
    r53 = r34 * r53;
    r22 = r42 < r51 ? r53 : r22;
    r22 = r42 < r47 ? r34 : r22;
    r42 = r47 * r22;
    r54 = fmaf(r34, r54, r55 * r42);
    r54 = r47 * r54;
    r54 = r54 * r56;
    r54 = r8 <= r33 ? r45 : r54;
    r46 = fmaf(r39, r54, r46);
    r46 = fmaf(r1, r57, r46);
    r54 = fmaf(r9, r54, r41);
    r54 = fmaf(r3, r57, r54);
    WriteIdx4<1024, float, float, float4>(out_calib_jac,
                                          4 * out_calib_jac_num_alloc,
                                          global_thread_idx,
                                          r15,
                                          r19,
                                          r46,
                                          r54);
    r41 = r39 * r27;
    r41 = fmaf(r65, r41, r13 * r62);
    r45 = r39 * r31;
    r45 = fmaf(r37, r62, r65 * r45);
    r33 = r39 * r15;
    r33 = fmaf(r65, r33, r19 * r62);
    r8 = r39 * r46;
    r8 = fmaf(r65, r8, r54 * r62);
    WriteSum4<float, float>((float*)inout_shared, r41, r45, r33, r8);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           0 * out_calib_njtr_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r8 = fmaf(r13, r13, r27 * r27);
    r33 = fmaf(r37, r37, r31 * r31);
    r45 = fmaf(r19, r19, r15 * r15);
    r41 = fmaf(r46, r46, r54 * r54);
    WriteSum4<float, float>((float*)inout_shared, r8, r33, r45, r41);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           0 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = fmaf(r27, r31, r13 * r37);
    r45 = fmaf(r27, r15, r13 * r19);
    r13 = fmaf(r27, r46, r13 * r54);
    r33 = fmaf(r37, r19, r31 * r15);
    WriteSum4<float, float>((float*)inout_shared, r41, r45, r13, r33);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           0 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = fmaf(r31, r46, r37 * r54);
    r54 = fmaf(r15, r46, r19 * r54);
    WriteSum2<float, float>((float*)inout_shared, r37, r54);
  };
  FlushSumShared<2, float>(out_calib_precond_tril,
                           4 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
}

void PinholeFixedPointResJac(float* pose,
                             unsigned int pose_num_alloc,
                             SharedIndex* pose_indices,
                             float* calib,
                             unsigned int calib_num_alloc,
                             SharedIndex* calib_indices,
                             float* pixel,
                             unsigned int pixel_num_alloc,
                             float* weight_loss,
                             unsigned int weight_loss_num_alloc,
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
                             float* out_calib_jac,
                             unsigned int out_calib_jac_num_alloc,
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
  PinholeFixedPointResJacKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      calib,
      calib_num_alloc,
      calib_indices,
      pixel,
      pixel_num_alloc,
      weight_loss,
      weight_loss_num_alloc,
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
      out_calib_jac,
      out_calib_jac_num_alloc,
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      out_calib_precond_diag,
      out_calib_precond_diag_num_alloc,
      out_calib_precond_tril,
      out_calib_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar