#include "kernel_pinhole_fixed_pose_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeFixedPoseResJacKernel(float* calib,
                                 unsigned int calib_num_alloc,
                                 SharedIndex* calib_indices,
                                 float* point,
                                 unsigned int point_num_alloc,
                                 SharedIndex* point_indices,
                                 float* pixel,
                                 unsigned int pixel_num_alloc,
                                 float* weight_loss,
                                 unsigned int weight_loss_num_alloc,
                                 float* pose,
                                 unsigned int pose_num_alloc,
                                 float* out_res,
                                 unsigned int out_res_num_alloc,
                                 float* out_calib_jac,
                                 unsigned int out_calib_jac_num_alloc,
                                 float* const out_calib_njtr,
                                 unsigned int out_calib_njtr_num_alloc,
                                 float* const out_calib_precond_diag,
                                 unsigned int out_calib_precond_diag_num_alloc,
                                 float* const out_calib_precond_tril,
                                 unsigned int out_calib_precond_tril_num_alloc,
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

  __shared__ SharedIndex calib_indices_loc[1024];
  calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
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
  };
  LoadShared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_indices_loc[threadIdx.x].target,
                       r13,
                       r14,
                       r15);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(
        pose, 0 * pose_num_alloc, global_thread_idx, r16, r17, r18, r19);
    r20 = r16 * r17;
    r21 = 2.00000000000000000e+00;
    r20 = r20 * r21;
    r22 = -2.00000000000000000e+00;
    r23 = r19 * r22;
    r24 = fmaf(r18, r23, r20);
    r6 = fmaf(r14, r24, r6);
    r25 = r16 * r18;
    r25 = r25 * r21;
    r26 = r17 * r19;
    r26 = fmaf(r21, r26, r25);
    r27 = r18 * r18;
    r27 = r22 * r27;
    r28 = 1.00000000000000000e+00;
    r29 = r17 * r17;
    r29 = fmaf(r22, r29, r28);
    r30 = r27 + r29;
    r6 = fmaf(r15, r26, r6);
    r6 = fmaf(r13, r30, r6);
    r31 = 9.99999999999999955e-07;
    r32 = r17 * r18;
    r32 = r32 * r21;
    r33 = r16 * r19;
    r33 = fmaf(r21, r33, r32);
    r12 = fmaf(r14, r33, r12);
    r25 = fmaf(r17, r23, r25);
    r34 = r16 * r16;
    r34 = r22 * r34;
    r29 = r34 + r29;
    r12 = fmaf(r13, r25, r12);
    r12 = fmaf(r15, r29, r12);
    r22 = copysign(1.0, r12);
    r22 = fmaf(r31, r22, r12);
    r12 = 1.0 / r22;
    r35 = r6 * r12;
    r8 = fmaf(r4, r35, r8);
    r9 = fmaf(r9, r10, r7);
    r7 = r18 * r19;
    r7 = fmaf(r21, r7, r20);
    r13 = fmaf(r13, r7, r11);
    r23 = fmaf(r16, r23, r32);
    r27 = r28 + r27;
    r27 = r27 + r34;
    r13 = fmaf(r15, r23, r13);
    r13 = fmaf(r14, r27, r13);
    r14 = r5 * r13;
    r9 = fmaf(r12, r14, r9);
    r15 = fmaf(r1, r9, r0 * r8);
    ReadIdx3<1024, float, float, float4>(weight_loss,
                                         4 * weight_loss_num_alloc,
                                         global_thread_idx,
                                         r34,
                                         r32,
                                         r11);
    r20 = 0.00000000000000000e+00;
    r11 = fmaxf(r11, r20);
    r36 = sqrtf(r11);
    r9 = fmaf(r3, r9, r2 * r8);
    r8 = fmaf(r9, r9, r15 * r15);
    r37 = 5.00000000000000000e-01;
    r32 = fmaxf(r32, r31);
    r38 = r32 * r32;
    r39 = r21 * r32;
    r40 = fmaxf(r31, r8);
    r41 = sqrtf(r40);
    r42 = r10 * r32;
    r42 = fmaf(r32, r42, r41 * r39);
    r42 = r8 <= r38 ? r8 : r42;
    r39 = 2.50000000000000000e+00;
    r41 = r32 * r32;
    r43 = 1.0 / r38;
    r43 = fmaf(r8, r43, r28);
    r44 = logf(r43);
    r41 = r41 * r44;
    r42 = r34 < r39 ? r41 : r42;
    r41 = 1.50000000000000000e+00;
    r44 = r21 * r32;
    r45 = sqrtf(r43);
    r45 = r10 + r45;
    r44 = r44 * r32;
    r44 = r44 * r45;
    r42 = r34 < r41 ? r44 : r42;
    r42 = r34 < r37 ? r8 : r42;
    r44 = fmaxf(r20, r42);
    r45 = 1.0 / r40;
    r45 = r11 * r45;
    r46 = r44 * r45;
    r47 = sqrtf(r46);
    r47 = r8 <= r31 ? r36 : r47;
    r36 = r15 * r47;
    r48 = r9 * r47;
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r36, r48);
    r44 = r11 * r44;
    r11 = -5.00000000000000000e-01;
    r48 = -9.99999999999999955e-07;
    r48 = r48 + r8;
    r48 = copysign(1.0, r48);
    r48 = r28 + r48;
    r36 = r40 * r40;
    r36 = 1.0 / r36;
    r44 = r44 * r11;
    r44 = r44 * r48;
    r44 = r44 * r36;
    r36 = r0 * r21;
    r36 = r36 * r15;
    r11 = r21 * r9;
    r49 = r2 * r35;
    r11 = fmaf(r49, r11, r35 * r36);
    r50 = r37 * r32;
    r40 = rsqrtf(r40);
    r50 = r50 * r48;
    r50 = r50 * r40;
    r40 = r11 * r50;
    r40 = r8 <= r38 ? r11 : r40;
    r48 = 1.0 / r43;
    r51 = r11 * r48;
    r40 = r34 < r39 ? r51 : r40;
    r43 = rsqrtf(r43);
    r51 = r11 * r43;
    r40 = r34 < r41 ? r51 : r40;
    r40 = r34 < r37 ? r11 : r40;
    r51 = r37 * r40;
    r42 = copysign(1.0, r42);
    r42 = r28 + r42;
    r45 = r42 * r45;
    r51 = fmaf(r45, r51, r11 * r44);
    r51 = r37 * r51;
    r46 = rsqrtf(r46);
    r51 = r51 * r46;
    r51 = r8 <= r31 ? r20 : r51;
    r11 = r10 * r15;
    r42 = 2.50000000000000000e-01;
    r28 = r8 <= r38 ? r20 : r20;
    r28 = r34 < r39 ? r20 : r28;
    r28 = r34 < r41 ? r20 : r28;
    r28 = r34 < r37 ? r20 : r28;
    r28 = r42 * r28;
    r28 = r28 * r46;
    r28 = r28 * r45;
    r28 = r8 <= r31 ? r20 : r28;
    r11 = r11 * r28;
    r42 = fmaf(r15, r51, r11);
    r52 = r0 * r47;
    r42 = fmaf(r35, r52, r42);
    r52 = r10 * r9;
    r52 = r52 * r28;
    r51 = fmaf(r9, r51, r52);
    r51 = fmaf(r47, r49, r51);
    r49 = r1 * r13;
    r28 = r21 * r15;
    r49 = r49 * r12;
    r35 = r3 * r21;
    r35 = r35 * r13;
    r35 = r35 * r9;
    r35 = fmaf(r12, r35, r28 * r49);
    r49 = r35 * r50;
    r49 = r8 <= r38 ? r35 : r49;
    r53 = r35 * r48;
    r49 = r34 < r39 ? r53 : r49;
    r53 = r35 * r43;
    r49 = r34 < r41 ? r53 : r49;
    r49 = r34 < r37 ? r35 : r49;
    r53 = r37 * r49;
    r35 = fmaf(r35, r44, r45 * r53);
    r35 = r37 * r35;
    r35 = r35 * r46;
    r35 = r8 <= r31 ? r20 : r35;
    r53 = fmaf(r15, r35, r11);
    r54 = r1 * r13;
    r54 = r54 * r47;
    r53 = fmaf(r12, r54, r53);
    r35 = fmaf(r9, r35, r52);
    r54 = r3 * r13;
    r54 = r54 * r47;
    r35 = fmaf(r12, r54, r35);
    WriteIdx4<1024, float, float, float4>(out_calib_jac,
                                          0 * out_calib_jac_num_alloc,
                                          global_thread_idx,
                                          r42,
                                          r51,
                                          r53,
                                          r35);
    r54 = fmaf(r0, r47, r11);
    r55 = r2 * r21;
    r55 = fmaf(r9, r55, r36);
    r36 = r55 * r50;
    r36 = r8 <= r38 ? r55 : r36;
    r56 = r55 * r48;
    r36 = r34 < r39 ? r56 : r36;
    r56 = r55 * r43;
    r36 = r34 < r41 ? r56 : r36;
    r36 = r34 < r37 ? r55 : r36;
    r56 = r37 * r36;
    r56 = fmaf(r45, r56, r55 * r44);
    r56 = r37 * r56;
    r56 = r56 * r46;
    r56 = r8 <= r31 ? r20 : r56;
    r54 = fmaf(r15, r56, r54);
    r55 = fmaf(r2, r47, r52);
    r55 = fmaf(r9, r56, r55);
    r56 = r3 * r21;
    r56 = fmaf(r1, r28, r9 * r56);
    r57 = r56 * r50;
    r57 = r8 <= r38 ? r56 : r57;
    r58 = r56 * r48;
    r57 = r34 < r39 ? r58 : r57;
    r58 = r56 * r43;
    r57 = r34 < r41 ? r58 : r57;
    r57 = r34 < r37 ? r56 : r57;
    r58 = r37 * r57;
    r56 = fmaf(r56, r44, r45 * r58);
    r56 = r37 * r56;
    r56 = r56 * r46;
    r56 = r8 <= r31 ? r20 : r56;
    r58 = fmaf(r15, r56, r11);
    r58 = fmaf(r1, r47, r58);
    r56 = fmaf(r9, r56, r52);
    r56 = fmaf(r3, r47, r56);
    WriteIdx4<1024, float, float, float4>(out_calib_jac,
                                          4 * out_calib_jac_num_alloc,
                                          global_thread_idx,
                                          r54,
                                          r55,
                                          r58,
                                          r56);
    r59 = r10 * r47;
    r60 = r9 * r59;
    r61 = r15 * r42;
    r61 = fmaf(r59, r61, r51 * r60);
    r62 = r15 * r53;
    r62 = fmaf(r35, r60, r59 * r62);
    r63 = r15 * r54;
    r63 = fmaf(r59, r63, r55 * r60);
    r64 = r15 * r58;
    r64 = fmaf(r59, r64, r56 * r60);
    WriteSum4<float, float>((float*)inout_shared, r61, r62, r63, r64);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           0 * out_calib_njtr_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r64 = fmaf(r51, r51, r42 * r42);
    r63 = fmaf(r35, r35, r53 * r53);
    r62 = fmaf(r55, r55, r54 * r54);
    r61 = fmaf(r58, r58, r56 * r56);
    WriteSum4<float, float>((float*)inout_shared, r64, r63, r62, r61);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           0 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r61 = fmaf(r42, r53, r51 * r35);
    r62 = fmaf(r42, r54, r51 * r55);
    r51 = fmaf(r42, r58, r51 * r56);
    r63 = fmaf(r35, r55, r53 * r54);
    WriteSum4<float, float>((float*)inout_shared, r61, r62, r51, r63);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           0 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = fmaf(r53, r58, r35 * r56);
    r56 = fmaf(r54, r58, r55 * r56);
    WriteSum2<float, float>((float*)inout_shared, r35, r56);
  };
  FlushSumShared<2, float>(out_calib_precond_tril,
                           4 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r56 = r21 * r9;
    r35 = r4 * r30;
    r6 = r4 * r6;
    r22 = r22 * r22;
    r22 = 1.0 / r22;
    r6 = r6 * r10;
    r6 = r6 * r22;
    r35 = fmaf(r25, r6, r12 * r35);
    r55 = r5 * r7;
    r63 = r25 * r10;
    r63 = r63 * r22;
    r63 = fmaf(r14, r63, r12 * r55);
    r55 = fmaf(r3, r63, r2 * r35);
    r63 = fmaf(r1, r63, r0 * r35);
    r56 = fmaf(r63, r28, r55 * r56);
    r35 = r56 * r50;
    r35 = r8 <= r38 ? r56 : r35;
    r51 = r56 * r48;
    r35 = r34 < r39 ? r51 : r35;
    r51 = r56 * r43;
    r35 = r34 < r41 ? r51 : r35;
    r35 = r34 < r37 ? r56 : r35;
    r51 = r37 * r35;
    r51 = fmaf(r45, r51, r56 * r44);
    r51 = r37 * r51;
    r51 = r51 * r46;
    r51 = r8 <= r31 ? r20 : r51;
    r56 = fmaf(r15, r51, r11);
    r56 = fmaf(r47, r63, r56);
    r51 = fmaf(r9, r51, r52);
    r51 = fmaf(r47, r55, r51);
    r55 = r21 * r9;
    r63 = r4 * r24;
    r63 = fmaf(r33, r6, r12 * r63);
    r62 = r5 * r27;
    r61 = r33 * r10;
    r61 = r61 * r22;
    r61 = fmaf(r14, r61, r12 * r62);
    r62 = fmaf(r3, r61, r2 * r63);
    r61 = fmaf(r1, r61, r0 * r63);
    r55 = fmaf(r61, r28, r62 * r55);
    r63 = r55 * r50;
    r63 = r8 <= r38 ? r55 : r63;
    r64 = r55 * r48;
    r63 = r34 < r39 ? r64 : r63;
    r64 = r55 * r43;
    r63 = r34 < r41 ? r64 : r63;
    r63 = r34 < r37 ? r55 : r63;
    r64 = r37 * r63;
    r55 = fmaf(r55, r44, r45 * r64);
    r55 = r37 * r55;
    r55 = r55 * r46;
    r55 = r8 <= r31 ? r20 : r55;
    r64 = fmaf(r15, r55, r11);
    r64 = fmaf(r47, r61, r64);
    r62 = fmaf(r47, r62, r52);
    r62 = fmaf(r9, r55, r62);
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r56,
                                          r51,
                                          r64,
                                          r62);
    r55 = r21 * r9;
    r61 = r4 * r26;
    r6 = fmaf(r29, r6, r12 * r61);
    r61 = r5 * r23;
    r65 = r29 * r10;
    r65 = r65 * r22;
    r65 = fmaf(r14, r65, r12 * r61);
    r61 = fmaf(r3, r65, r2 * r6);
    r65 = fmaf(r1, r65, r0 * r6);
    r28 = fmaf(r65, r28, r61 * r55);
    r50 = r28 * r50;
    r50 = r8 <= r38 ? r28 : r50;
    r48 = r28 * r48;
    r50 = r34 < r39 ? r48 : r50;
    r43 = r28 * r43;
    r50 = r34 < r41 ? r43 : r50;
    r50 = r34 < r37 ? r28 : r50;
    r34 = r37 * r50;
    r44 = fmaf(r28, r44, r45 * r34);
    r44 = r37 * r44;
    r44 = r44 * r46;
    r44 = r8 <= r31 ? r20 : r44;
    r11 = fmaf(r15, r44, r11);
    r11 = fmaf(r47, r65, r11);
    r61 = fmaf(r47, r61, r52);
    r61 = fmaf(r9, r44, r61);
    WriteIdx2<1024, float, float, float2>(out_point_jac,
                                          4 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r11,
                                          r61);
    r44 = r15 * r56;
    r44 = fmaf(r51, r60, r59 * r44);
    r52 = r15 * r64;
    r52 = fmaf(r62, r60, r59 * r52);
    r65 = r15 * r11;
    r60 = fmaf(r61, r60, r59 * r65);
    WriteSum3<float, float>((float*)inout_shared, r44, r52, r60);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r60 = fmaf(r51, r51, r56 * r56);
    r52 = fmaf(r62, r62, r64 * r64);
    r44 = fmaf(r11, r11, r61 * r61);
    WriteSum3<float, float>((float*)inout_shared, r60, r52, r44);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r44 = fmaf(r56, r64, r51 * r62);
    r51 = fmaf(r51, r61, r56 * r11);
    r61 = fmaf(r64, r11, r62 * r61);
    WriteSum3<float, float>((float*)inout_shared, r44, r51, r61);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
}

void PinholeFixedPoseResJac(float* calib,
                            unsigned int calib_num_alloc,
                            SharedIndex* calib_indices,
                            float* point,
                            unsigned int point_num_alloc,
                            SharedIndex* point_indices,
                            float* pixel,
                            unsigned int pixel_num_alloc,
                            float* weight_loss,
                            unsigned int weight_loss_num_alloc,
                            float* pose,
                            unsigned int pose_num_alloc,
                            float* out_res,
                            unsigned int out_res_num_alloc,
                            float* out_calib_jac,
                            unsigned int out_calib_jac_num_alloc,
                            float* const out_calib_njtr,
                            unsigned int out_calib_njtr_num_alloc,
                            float* const out_calib_precond_diag,
                            unsigned int out_calib_precond_diag_num_alloc,
                            float* const out_calib_precond_tril,
                            unsigned int out_calib_precond_tril_num_alloc,
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
  PinholeFixedPoseResJacKernel<<<n_blocks, 1024>>>(
      calib,
      calib_num_alloc,
      calib_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      weight_loss,
      weight_loss_num_alloc,
      pose,
      pose_num_alloc,
      out_res,
      out_res_num_alloc,
      out_calib_jac,
      out_calib_jac_num_alloc,
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      out_calib_precond_diag,
      out_calib_precond_diag_num_alloc,
      out_calib_precond_tril,
      out_calib_precond_tril_num_alloc,
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