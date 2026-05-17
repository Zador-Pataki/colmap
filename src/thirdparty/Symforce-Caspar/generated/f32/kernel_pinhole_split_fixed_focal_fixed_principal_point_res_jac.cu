#include "kernel_pinhole_split_fixed_focal_fixed_principal_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedFocalFixedPrincipalPointResJacKernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* weight_loss,
        unsigned int weight_loss_num_alloc,
        float* focal,
        unsigned int focal_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
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

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71;

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
  LoadShared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_indices_loc[threadIdx.x].target,
                       r11,
                       r12,
                       r13);
  };
  __syncthreads();
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
        focal, 0 * focal_num_alloc, global_thread_idx, r32, r33);
    r34 = 9.99999999999999955e-07;
    r35 = r15 * r16;
    r35 = r35 * r19;
    r36 = r14 * r17;
    r36 = r36 * r19;
    r37 = r35 + r36;
    r10 = fmaf(r12, r37, r10);
    r38 = r15 * r17;
    r38 = r38 * r20;
    r24 = r24 + r38;
    r39 = r14 * r14;
    r40 = r20 * r39;
    r30 = r40 + r30;
    r10 = fmaf(r11, r24, r10);
    r10 = fmaf(r13, r30, r10);
    r41 = copysign(1.0, r10);
    r41 = fmaf(r34, r41, r10);
    r10 = 1.0 / r41;
    r42 = r32 * r10;
    r6 = fmaf(r4, r42, r6);
    r7 = fmaf(r7, r8, r5);
    r5 = r16 * r17;
    r5 = r5 * r19;
    r18 = r18 + r5;
    r9 = fmaf(r11, r18, r9);
    r43 = r14 * r17;
    r43 = r43 * r20;
    r35 = r35 + r43;
    r27 = r28 + r27;
    r27 = r27 + r40;
    r9 = fmaf(r13, r35, r9);
    r9 = fmaf(r12, r27, r9);
    r9 = r33 * r9;
    r7 = fmaf(r10, r9, r7);
    r40 = fmaf(r1, r7, r0 * r6);
    ReadIdx3<1024, float, float, float4>(weight_loss,
                                         4 * weight_loss_num_alloc,
                                         global_thread_idx,
                                         r44,
                                         r45,
                                         r46);
    r47 = 0.00000000000000000e+00;
    r46 = fmaxf(r46, r47);
    r48 = sqrtf(r46);
    r7 = fmaf(r3, r7, r2 * r6);
    r6 = fmaf(r40, r40, r7 * r7);
    r49 = 5.00000000000000000e-01;
    r45 = fmaxf(r45, r34);
    r50 = r45 * r45;
    r51 = r19 * r45;
    r52 = fmaxf(r34, r6);
    r53 = sqrtf(r52);
    r54 = r8 * r45;
    r54 = fmaf(r45, r54, r53 * r51);
    r54 = r6 <= r50 ? r6 : r54;
    r51 = 2.50000000000000000e+00;
    r53 = r45 * r45;
    r55 = 1.0 / r50;
    r55 = fmaf(r6, r55, r28);
    r56 = logf(r55);
    r53 = r53 * r56;
    r54 = r44 < r51 ? r53 : r54;
    r53 = 1.50000000000000000e+00;
    r56 = r19 * r45;
    r57 = sqrtf(r55);
    r57 = r8 + r57;
    r56 = r56 * r45;
    r56 = r56 * r57;
    r54 = r44 < r53 ? r56 : r54;
    r54 = r44 < r49 ? r6 : r54;
    r56 = fmaxf(r47, r54);
    r57 = 1.0 / r52;
    r57 = r46 * r57;
    r58 = r56 * r57;
    r59 = sqrtf(r58);
    r59 = r6 <= r34 ? r48 : r59;
    r48 = r40 * r59;
    r60 = r7 * r59;
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r48, r60);
    r60 = r8 * r40;
    r48 = 2.50000000000000000e-01;
    r61 = r6 <= r50 ? r47 : r47;
    r61 = r44 < r51 ? r47 : r61;
    r61 = r44 < r53 ? r47 : r61;
    r61 = r44 < r49 ? r47 : r61;
    r61 = r48 * r61;
    r58 = rsqrtf(r58);
    r54 = copysign(1.0, r54);
    r54 = r28 + r54;
    r57 = r54 * r57;
    r61 = r61 * r58;
    r61 = r61 * r57;
    r61 = r6 <= r34 ? r47 : r61;
    r60 = r60 * r61;
    r56 = r46 * r56;
    r46 = -5.00000000000000000e-01;
    r54 = -9.99999999999999955e-07;
    r54 = r54 + r6;
    r54 = copysign(1.0, r54);
    r54 = r28 + r54;
    r28 = r52 * r52;
    r28 = 1.0 / r28;
    r56 = r56 * r46;
    r56 = r56 * r54;
    r56 = r56 * r28;
    r28 = r19 * r40;
    r46 = r16 * r16;
    r17 = r17 * r17;
    r48 = r8 * r17;
    r62 = r46 + r48;
    r63 = r8 * r29;
    r64 = r39 + r63;
    r65 = r62 + r64;
    r65 = fmaf(r13, r65, r12 * r35);
    r66 = r33 * r65;
    r67 = r15 * r21;
    r43 = r43 + r67;
    r63 = r46 + r63;
    r46 = r8 * r39;
    r68 = r17 + r46;
    r63 = r63 + r68;
    r63 = fmaf(r12, r63, r13 * r43);
    r43 = r8 * r63;
    r41 = r41 * r41;
    r41 = 1.0 / r41;
    r43 = r43 * r41;
    r43 = fmaf(r9, r43, r10 * r66);
    r15 = r14 * r15;
    r15 = r15 * r20;
    r5 = r5 + r15;
    r5 = fmaf(r12, r26, r13 * r5);
    r4 = r32 * r4;
    r4 = r4 * r8;
    r4 = r4 * r41;
    r5 = fmaf(r63, r4, r5 * r42);
    r32 = fmaf(r0, r5, r1 * r43);
    r5 = fmaf(r2, r5, r3 * r43);
    r43 = r19 * r7;
    r28 = fmaf(r5, r43, r32 * r28);
    r20 = r49 * r45;
    r52 = rsqrtf(r52);
    r20 = r20 * r54;
    r20 = r20 * r52;
    r52 = r28 * r20;
    r52 = r6 <= r50 ? r28 : r52;
    r54 = 1.0 / r55;
    r66 = r28 * r54;
    r52 = r44 < r51 ? r66 : r52;
    r55 = rsqrtf(r55);
    r66 = r28 * r55;
    r52 = r44 < r53 ? r66 : r52;
    r52 = r44 < r49 ? r28 : r52;
    r66 = r49 * r52;
    r66 = fmaf(r57, r66, r28 * r56);
    r66 = r49 * r66;
    r66 = r66 * r58;
    r66 = r6 <= r34 ? r47 : r66;
    r28 = fmaf(r40, r66, r60);
    r28 = fmaf(r59, r32, r28);
    r32 = r8 * r7;
    r32 = r32 * r61;
    r66 = fmaf(r7, r66, r32);
    r66 = fmaf(r59, r5, r66);
    r21 = r14 * r21;
    r38 = r38 + r21;
    r16 = r16 * r16;
    r16 = r16 * r8;
    r17 = r17 + r16;
    r17 = r17 + r64;
    r17 = fmaf(r13, r17, r11 * r38);
    r48 = r39 + r48;
    r16 = r29 + r16;
    r48 = r48 + r16;
    r48 = fmaf(r11, r48, r13 * r24);
    r17 = fmaf(r48, r4, r17 * r42);
    r67 = r36 + r67;
    r67 = fmaf(r11, r67, r13 * r18);
    r13 = r33 * r67;
    r36 = r8 * r48;
    r36 = r36 * r41;
    r36 = fmaf(r9, r36, r10 * r13);
    r13 = fmaf(r1, r36, r0 * r17);
    r39 = fmaf(r59, r13, r60);
    r38 = r19 * r40;
    r36 = fmaf(r3, r36, r2 * r17);
    r38 = fmaf(r36, r43, r13 * r38);
    r13 = r38 * r20;
    r13 = r6 <= r50 ? r38 : r13;
    r17 = r38 * r54;
    r13 = r44 < r51 ? r17 : r13;
    r17 = r38 * r55;
    r13 = r44 < r53 ? r17 : r13;
    r13 = r44 < r49 ? r38 : r13;
    r17 = r49 * r13;
    r38 = fmaf(r38, r56, r57 * r17);
    r38 = r49 * r38;
    r38 = r38 * r58;
    r38 = r6 <= r34 ? r47 : r38;
    r39 = fmaf(r40, r38, r39);
    r36 = fmaf(r59, r36, r32);
    r36 = fmaf(r7, r38, r36);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r28,
                                          r66,
                                          r39,
                                          r36);
    r38 = r19 * r40;
    r21 = r25 + r21;
    r21 = fmaf(r12, r21, r11 * r37);
    r25 = r8 * r21;
    r25 = r25 * r41;
    r22 = r15 + r22;
    r16 = r68 + r16;
    r16 = fmaf(r11, r16, r12 * r22);
    r22 = r33 * r16;
    r22 = fmaf(r10, r22, r9 * r25);
    r46 = r29 + r46;
    r46 = r46 + r62;
    r46 = fmaf(r12, r46, r11 * r23);
    r46 = fmaf(r21, r4, r46 * r42);
    r12 = fmaf(r0, r46, r1 * r22);
    r46 = fmaf(r2, r46, r3 * r22);
    r38 = fmaf(r46, r43, r12 * r38);
    r22 = r38 * r20;
    r22 = r6 <= r50 ? r38 : r22;
    r11 = r38 * r54;
    r22 = r44 < r51 ? r11 : r22;
    r11 = r38 * r55;
    r22 = r44 < r53 ? r11 : r22;
    r22 = r44 < r49 ? r38 : r22;
    r11 = r49 * r22;
    r38 = fmaf(r38, r56, r57 * r11);
    r38 = r49 * r38;
    r38 = r38 * r58;
    r38 = r6 <= r34 ? r47 : r38;
    r11 = fmaf(r40, r38, r60);
    r11 = fmaf(r59, r12, r11);
    r38 = fmaf(r7, r38, r32);
    r38 = fmaf(r59, r46, r38);
    r46 = r2 * r42;
    r12 = r0 * r19;
    r12 = r12 * r40;
    r12 = fmaf(r42, r12, r43 * r46);
    r62 = r12 * r20;
    r62 = r6 <= r50 ? r12 : r62;
    r29 = r12 * r54;
    r62 = r44 < r51 ? r29 : r62;
    r29 = r12 * r55;
    r62 = r44 < r53 ? r29 : r62;
    r62 = r44 < r49 ? r12 : r62;
    r29 = r49 * r62;
    r29 = fmaf(r57, r29, r12 * r56);
    r29 = r49 * r29;
    r29 = r29 * r58;
    r29 = r6 <= r34 ? r47 : r29;
    r12 = fmaf(r40, r29, r60);
    r25 = r0 * r59;
    r12 = fmaf(r42, r25, r12);
    r29 = fmaf(r7, r29, r32);
    r29 = fmaf(r59, r46, r29);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r11,
                                          r38,
                                          r12,
                                          r29);
    r46 = r19 * r40;
    r25 = r1 * r8;
    r25 = r25 * r41;
    r25 = fmaf(r9, r25, r0 * r4);
    r68 = r3 * r8;
    r68 = r68 * r41;
    r68 = fmaf(r9, r68, r2 * r4);
    r46 = fmaf(r68, r43, r25 * r46);
    r15 = r46 * r20;
    r15 = r6 <= r50 ? r46 : r15;
    r17 = r46 * r54;
    r15 = r44 < r51 ? r17 : r15;
    r17 = r46 * r55;
    r15 = r44 < r53 ? r17 : r15;
    r15 = r44 < r49 ? r46 : r15;
    r17 = r49 * r15;
    r46 = fmaf(r46, r56, r57 * r17);
    r46 = r49 * r46;
    r46 = r46 * r58;
    r46 = r6 <= r34 ? r47 : r46;
    r17 = fmaf(r40, r46, r60);
    r17 = fmaf(r59, r25, r17);
    r68 = fmaf(r59, r68, r32);
    r68 = fmaf(r7, r46, r68);
    r46 = r3 * r33;
    r46 = r46 * r10;
    r25 = r1 * r33;
    r25 = r25 * r19;
    r25 = r25 * r40;
    r25 = fmaf(r10, r25, r43 * r46);
    r46 = r25 * r20;
    r46 = r6 <= r50 ? r25 : r46;
    r64 = r25 * r54;
    r46 = r44 < r51 ? r64 : r46;
    r64 = r25 * r55;
    r46 = r44 < r53 ? r64 : r46;
    r46 = r44 < r49 ? r25 : r46;
    r64 = r49 * r46;
    r64 = fmaf(r57, r64, r25 * r56);
    r64 = r49 * r64;
    r64 = r64 * r58;
    r64 = r6 <= r34 ? r47 : r64;
    r25 = fmaf(r40, r64, r60);
    r14 = r1 * r33;
    r14 = r14 * r59;
    r25 = fmaf(r10, r14, r25);
    r64 = fmaf(r7, r64, r32);
    r14 = r3 * r33;
    r14 = r14 * r59;
    r64 = fmaf(r10, r14, r64);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r25,
                                          r64,
                                          r17,
                                          r68);
    r14 = r7 * r66;
    r5 = r8 * r59;
    r61 = r40 * r5;
    r14 = fmaf(r28, r61, r5 * r14);
    r69 = r7 * r36;
    r69 = fmaf(r5, r69, r39 * r61);
    r70 = r7 * r38;
    r70 = fmaf(r5, r70, r11 * r61);
    r71 = r7 * r29;
    r71 = fmaf(r12, r61, r5 * r71);
    WriteSum4<float, float>((float*)inout_shared, r14, r69, r70, r71);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r71 = r7 * r64;
    r71 = fmaf(r5, r71, r25 * r61);
    r70 = r7 * r68;
    r70 = fmaf(r17, r61, r5 * r70);
    WriteSum2<float, float>((float*)inout_shared, r71, r70);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r70 = fmaf(r28, r28, r66 * r66);
    r71 = fmaf(r39, r39, r36 * r36);
    r69 = fmaf(r11, r11, r38 * r38);
    r14 = fmaf(r29, r29, r12 * r12);
    WriteSum4<float, float>((float*)inout_shared, r70, r71, r69, r14);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r14 = fmaf(r64, r64, r25 * r25);
    r69 = fmaf(r68, r68, r17 * r17);
    WriteSum2<float, float>((float*)inout_shared, r14, r69);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r69 = fmaf(r66, r36, r28 * r39);
    r14 = fmaf(r66, r38, r28 * r11);
    r71 = fmaf(r28, r12, r66 * r29);
    r70 = fmaf(r28, r25, r66 * r64);
    WriteSum4<float, float>((float*)inout_shared, r69, r14, r71, r70);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r28 = fmaf(r66, r68, r28 * r17);
    r70 = fmaf(r36, r38, r39 * r11);
    r71 = fmaf(r39, r12, r36 * r29);
    r14 = fmaf(r36, r64, r39 * r25);
    WriteSum4<float, float>((float*)inout_shared, r28, r70, r71, r14);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = fmaf(r39, r17, r36 * r68);
    r14 = fmaf(r11, r12, r38 * r29);
    r71 = fmaf(r11, r25, r38 * r64);
    r11 = fmaf(r11, r17, r38 * r68);
    WriteSum4<float, float>((float*)inout_shared, r39, r14, r71, r11);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = fmaf(r12, r25, r29 * r64);
    r12 = fmaf(r12, r17, r29 * r68);
    r17 = fmaf(r25, r17, r64 * r68);
    WriteSum3<float, float>((float*)inout_shared, r11, r12, r17);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r17 = r19 * r40;
    r31 = fmaf(r24, r4, r31 * r42);
    r12 = r24 * r8;
    r12 = r12 * r41;
    r11 = r33 * r18;
    r11 = fmaf(r10, r11, r9 * r12);
    r12 = fmaf(r1, r11, r0 * r31);
    r11 = fmaf(r3, r11, r2 * r31);
    r17 = fmaf(r11, r43, r12 * r17);
    r31 = r17 * r20;
    r31 = r6 <= r50 ? r17 : r31;
    r25 = r17 * r54;
    r31 = r44 < r51 ? r25 : r31;
    r25 = r17 * r55;
    r31 = r44 < r53 ? r25 : r31;
    r31 = r44 < r49 ? r17 : r31;
    r25 = r49 * r31;
    r25 = fmaf(r57, r25, r17 * r56);
    r25 = r49 * r25;
    r25 = r25 * r58;
    r25 = r6 <= r34 ? r47 : r25;
    r17 = fmaf(r40, r25, r60);
    r17 = fmaf(r59, r12, r17);
    r25 = fmaf(r7, r25, r32);
    r25 = fmaf(r59, r11, r25);
    r11 = r19 * r40;
    r23 = fmaf(r23, r42, r37 * r4);
    r12 = r33 * r27;
    r71 = r37 * r8;
    r71 = r71 * r41;
    r71 = fmaf(r9, r71, r10 * r12);
    r12 = fmaf(r1, r71, r0 * r23);
    r71 = fmaf(r3, r71, r2 * r23);
    r11 = fmaf(r71, r43, r12 * r11);
    r23 = r11 * r20;
    r23 = r6 <= r50 ? r11 : r23;
    r14 = r11 * r54;
    r23 = r44 < r51 ? r14 : r23;
    r14 = r11 * r55;
    r23 = r44 < r53 ? r14 : r23;
    r23 = r44 < r49 ? r11 : r23;
    r14 = r49 * r23;
    r11 = fmaf(r11, r56, r57 * r14);
    r11 = r49 * r11;
    r11 = r11 * r58;
    r11 = r6 <= r34 ? r47 : r11;
    r14 = fmaf(r40, r11, r60);
    r14 = fmaf(r59, r12, r14);
    r11 = fmaf(r7, r11, r32);
    r11 = fmaf(r59, r71, r11);
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r17,
                                          r25,
                                          r14,
                                          r11);
    r71 = r19 * r40;
    r42 = fmaf(r26, r42, r30 * r4);
    r26 = r30 * r8;
    r26 = r26 * r41;
    r41 = r33 * r35;
    r41 = fmaf(r10, r41, r9 * r26);
    r26 = fmaf(r1, r41, r0 * r42);
    r41 = fmaf(r3, r41, r2 * r42);
    r43 = fmaf(r41, r43, r26 * r71);
    r20 = r43 * r20;
    r20 = r6 <= r50 ? r43 : r20;
    r54 = r43 * r54;
    r20 = r44 < r51 ? r54 : r20;
    r55 = r43 * r55;
    r20 = r44 < r53 ? r55 : r20;
    r20 = r44 < r49 ? r43 : r20;
    r44 = r49 * r20;
    r44 = fmaf(r57, r44, r43 * r56);
    r44 = r49 * r44;
    r44 = r44 * r58;
    r44 = r6 <= r34 ? r47 : r44;
    r60 = fmaf(r40, r44, r60);
    r60 = fmaf(r59, r26, r60);
    r44 = fmaf(r7, r44, r32);
    r44 = fmaf(r59, r41, r44);
    WriteIdx2<1024, float, float, float2>(out_point_jac,
                                          4 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r60,
                                          r44);
    r41 = r7 * r25;
    r41 = fmaf(r17, r61, r5 * r41);
    r32 = r7 * r11;
    r32 = fmaf(r5, r32, r14 * r61);
    r26 = r7 * r44;
    r26 = fmaf(r5, r26, r60 * r61);
    WriteSum3<float, float>((float*)inout_shared, r41, r32, r26);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = fmaf(r25, r25, r17 * r17);
    r32 = fmaf(r11, r11, r14 * r14);
    r41 = fmaf(r60, r60, r44 * r44);
    WriteSum3<float, float>((float*)inout_shared, r26, r32, r41);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = fmaf(r25, r11, r17 * r14);
    r17 = fmaf(r17, r60, r25 * r44);
    r60 = fmaf(r14, r60, r11 * r44);
    WriteSum3<float, float>((float*)inout_shared, r41, r17, r60);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
}

void PinholeSplitFixedFocalFixedPrincipalPointResJac(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* weight_loss,
    unsigned int weight_loss_num_alloc,
    float* focal,
    unsigned int focal_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
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
  PinholeSplitFixedFocalFixedPrincipalPointResJacKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      weight_loss,
      weight_loss_num_alloc,
      focal,
      focal_num_alloc,
      principal_point,
      principal_point_num_alloc,
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