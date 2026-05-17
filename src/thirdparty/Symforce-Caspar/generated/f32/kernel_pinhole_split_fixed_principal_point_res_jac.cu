#include "kernel_pinhole_split_fixed_principal_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedPrincipalPointResJacKernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* focal,
        unsigned int focal_num_alloc,
        SharedIndex* focal_indices,
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* weight_loss,
        unsigned int weight_loss_num_alloc,
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
        float* out_focal_jac,
        unsigned int out_focal_jac_num_alloc,
        float* const out_focal_njtr,
        unsigned int out_focal_njtr_num_alloc,
        float* const out_focal_precond_diag,
        unsigned int out_focal_precond_diag_num_alloc,
        float* const out_focal_precond_tril,
        unsigned int out_focal_precond_tril_num_alloc,
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
  __shared__ SharedIndex focal_indices_loc[1024];
  focal_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_indices[global_thread_idx]
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
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75;

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
  };
  LoadShared<2, float, float>(
      focal, 0 * focal_num_alloc, focal_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>(
        (float*)inout_shared, focal_indices_loc[threadIdx.x].target, r32, r33);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
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
    r40 = r33 * r9;
    r7 = fmaf(r10, r40, r7);
    r44 = fmaf(r1, r7, r0 * r6);
    ReadIdx3<1024, float, float, float4>(weight_loss,
                                         4 * weight_loss_num_alloc,
                                         global_thread_idx,
                                         r45,
                                         r46,
                                         r47);
    r48 = 0.00000000000000000e+00;
    r47 = fmaxf(r47, r48);
    r49 = sqrtf(r47);
    r7 = fmaf(r3, r7, r2 * r6);
    r6 = fmaf(r44, r44, r7 * r7);
    r50 = 5.00000000000000000e-01;
    r46 = fmaxf(r46, r34);
    r51 = r46 * r46;
    r52 = r19 * r46;
    r53 = fmaxf(r34, r6);
    r54 = sqrtf(r53);
    r55 = r8 * r46;
    r55 = fmaf(r46, r55, r54 * r52);
    r55 = r6 <= r51 ? r6 : r55;
    r52 = 2.50000000000000000e+00;
    r54 = r46 * r46;
    r56 = 1.0 / r51;
    r56 = fmaf(r6, r56, r28);
    r57 = logf(r56);
    r54 = r54 * r57;
    r55 = r45 < r52 ? r54 : r55;
    r54 = 1.50000000000000000e+00;
    r57 = r19 * r46;
    r58 = sqrtf(r56);
    r58 = r8 + r58;
    r57 = r57 * r46;
    r57 = r57 * r58;
    r55 = r45 < r54 ? r57 : r55;
    r55 = r45 < r50 ? r6 : r55;
    r57 = fmaxf(r48, r55);
    r58 = 1.0 / r53;
    r58 = r47 * r58;
    r59 = r57 * r58;
    r60 = sqrtf(r59);
    r60 = r6 <= r34 ? r49 : r60;
    r49 = r44 * r60;
    r61 = r7 * r60;
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r49, r61);
    r61 = r8 * r44;
    r49 = 2.50000000000000000e-01;
    r62 = r6 <= r51 ? r48 : r48;
    r62 = r45 < r52 ? r48 : r62;
    r62 = r45 < r54 ? r48 : r62;
    r62 = r45 < r50 ? r48 : r62;
    r62 = r49 * r62;
    r59 = rsqrtf(r59);
    r55 = copysign(1.0, r55);
    r55 = r28 + r55;
    r58 = r55 * r58;
    r62 = r62 * r59;
    r62 = r62 * r58;
    r62 = r6 <= r34 ? r48 : r62;
    r61 = r61 * r62;
    r57 = r47 * r57;
    r47 = -5.00000000000000000e-01;
    r55 = -9.99999999999999955e-07;
    r55 = r55 + r6;
    r55 = copysign(1.0, r55);
    r55 = r28 + r55;
    r28 = r53 * r53;
    r28 = 1.0 / r28;
    r57 = r57 * r47;
    r57 = r57 * r55;
    r57 = r57 * r28;
    r28 = r19 * r44;
    r47 = r16 * r16;
    r17 = r17 * r17;
    r49 = r8 * r17;
    r63 = r47 + r49;
    r64 = r8 * r29;
    r65 = r39 + r64;
    r66 = r63 + r65;
    r66 = fmaf(r13, r66, r12 * r35);
    r67 = r33 * r66;
    r68 = r15 * r21;
    r43 = r43 + r68;
    r64 = r47 + r64;
    r47 = r8 * r39;
    r69 = r17 + r47;
    r64 = r64 + r69;
    r64 = fmaf(r12, r64, r13 * r43);
    r43 = r8 * r64;
    r41 = r41 * r41;
    r41 = 1.0 / r41;
    r43 = r43 * r41;
    r43 = fmaf(r40, r43, r10 * r67);
    r15 = r14 * r15;
    r15 = r15 * r20;
    r5 = r5 + r15;
    r5 = fmaf(r12, r26, r13 * r5);
    r32 = r32 * r4;
    r32 = r32 * r8;
    r32 = r32 * r41;
    r5 = fmaf(r64, r32, r5 * r42);
    r20 = fmaf(r0, r5, r1 * r43);
    r5 = fmaf(r2, r5, r3 * r43);
    r43 = r19 * r7;
    r28 = fmaf(r5, r43, r20 * r28);
    r67 = r50 * r46;
    r53 = rsqrtf(r53);
    r67 = r67 * r55;
    r67 = r67 * r53;
    r53 = r28 * r67;
    r53 = r6 <= r51 ? r28 : r53;
    r55 = 1.0 / r56;
    r70 = r28 * r55;
    r53 = r45 < r52 ? r70 : r53;
    r56 = rsqrtf(r56);
    r70 = r28 * r56;
    r53 = r45 < r54 ? r70 : r53;
    r53 = r45 < r50 ? r28 : r53;
    r70 = r50 * r53;
    r70 = fmaf(r58, r70, r28 * r57);
    r70 = r50 * r70;
    r70 = r70 * r59;
    r70 = r6 <= r34 ? r48 : r70;
    r28 = fmaf(r44, r70, r61);
    r28 = fmaf(r60, r20, r28);
    r20 = r8 * r7;
    r20 = r20 * r62;
    r70 = fmaf(r7, r70, r20);
    r70 = fmaf(r60, r5, r70);
    r21 = r14 * r21;
    r38 = r38 + r21;
    r16 = r16 * r16;
    r16 = r16 * r8;
    r17 = r17 + r16;
    r17 = r17 + r65;
    r17 = fmaf(r13, r17, r11 * r38);
    r49 = r39 + r49;
    r16 = r29 + r16;
    r49 = r49 + r16;
    r49 = fmaf(r11, r49, r13 * r24);
    r17 = fmaf(r49, r32, r17 * r42);
    r68 = r36 + r68;
    r68 = fmaf(r11, r68, r13 * r18);
    r13 = r33 * r68;
    r36 = r8 * r49;
    r36 = r36 * r41;
    r36 = fmaf(r40, r36, r10 * r13);
    r13 = fmaf(r1, r36, r0 * r17);
    r39 = fmaf(r60, r13, r61);
    r38 = r19 * r44;
    r36 = fmaf(r3, r36, r2 * r17);
    r38 = fmaf(r36, r43, r13 * r38);
    r13 = r38 * r67;
    r13 = r6 <= r51 ? r38 : r13;
    r17 = r38 * r55;
    r13 = r45 < r52 ? r17 : r13;
    r17 = r38 * r56;
    r13 = r45 < r54 ? r17 : r13;
    r13 = r45 < r50 ? r38 : r13;
    r17 = r50 * r13;
    r38 = fmaf(r38, r57, r58 * r17);
    r38 = r50 * r38;
    r38 = r38 * r59;
    r38 = r6 <= r34 ? r48 : r38;
    r39 = fmaf(r44, r38, r39);
    r36 = fmaf(r60, r36, r20);
    r36 = fmaf(r7, r38, r36);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r28,
                                          r70,
                                          r39,
                                          r36);
    r38 = r19 * r44;
    r21 = r25 + r21;
    r21 = fmaf(r12, r21, r11 * r37);
    r25 = r8 * r21;
    r25 = r25 * r41;
    r22 = r15 + r22;
    r16 = r69 + r16;
    r16 = fmaf(r11, r16, r12 * r22);
    r22 = r33 * r16;
    r22 = fmaf(r10, r22, r40 * r25);
    r47 = r29 + r47;
    r47 = r47 + r63;
    r47 = fmaf(r12, r47, r11 * r23);
    r47 = fmaf(r21, r32, r47 * r42);
    r12 = fmaf(r0, r47, r1 * r22);
    r47 = fmaf(r2, r47, r3 * r22);
    r38 = fmaf(r47, r43, r12 * r38);
    r22 = r38 * r67;
    r22 = r6 <= r51 ? r38 : r22;
    r11 = r38 * r55;
    r22 = r45 < r52 ? r11 : r22;
    r11 = r38 * r56;
    r22 = r45 < r54 ? r11 : r22;
    r22 = r45 < r50 ? r38 : r22;
    r11 = r50 * r22;
    r38 = fmaf(r38, r57, r58 * r11);
    r38 = r50 * r38;
    r38 = r38 * r59;
    r38 = r6 <= r34 ? r48 : r38;
    r11 = fmaf(r44, r38, r61);
    r11 = fmaf(r60, r12, r11);
    r38 = fmaf(r7, r38, r20);
    r38 = fmaf(r60, r47, r38);
    r47 = r2 * r42;
    r12 = r0 * r19;
    r12 = r12 * r44;
    r12 = fmaf(r42, r12, r43 * r47);
    r63 = r12 * r67;
    r63 = r6 <= r51 ? r12 : r63;
    r29 = r12 * r55;
    r63 = r45 < r52 ? r29 : r63;
    r29 = r12 * r56;
    r63 = r45 < r54 ? r29 : r63;
    r63 = r45 < r50 ? r12 : r63;
    r29 = r50 * r63;
    r29 = fmaf(r58, r29, r12 * r57);
    r29 = r50 * r29;
    r29 = r29 * r59;
    r29 = r6 <= r34 ? r48 : r29;
    r12 = fmaf(r44, r29, r61);
    r25 = r0 * r60;
    r12 = fmaf(r42, r25, r12);
    r29 = fmaf(r7, r29, r20);
    r29 = fmaf(r60, r47, r29);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r11,
                                          r38,
                                          r12,
                                          r29);
    r47 = r19 * r44;
    r25 = r1 * r8;
    r25 = r25 * r41;
    r25 = fmaf(r40, r25, r0 * r32);
    r69 = r3 * r8;
    r69 = r69 * r41;
    r69 = fmaf(r40, r69, r2 * r32);
    r47 = fmaf(r69, r43, r25 * r47);
    r15 = r47 * r67;
    r15 = r6 <= r51 ? r47 : r15;
    r17 = r47 * r55;
    r15 = r45 < r52 ? r17 : r15;
    r17 = r47 * r56;
    r15 = r45 < r54 ? r17 : r15;
    r15 = r45 < r50 ? r47 : r15;
    r17 = r50 * r15;
    r47 = fmaf(r47, r57, r58 * r17);
    r47 = r50 * r47;
    r47 = r47 * r59;
    r47 = r6 <= r34 ? r48 : r47;
    r17 = fmaf(r44, r47, r61);
    r17 = fmaf(r60, r25, r17);
    r69 = fmaf(r60, r69, r20);
    r69 = fmaf(r7, r47, r69);
    r47 = r10 * r43;
    r25 = r3 * r47;
    r65 = r33 * r1;
    r65 = r65 * r19;
    r65 = r65 * r44;
    r65 = fmaf(r10, r65, r33 * r25);
    r14 = r65 * r67;
    r14 = r6 <= r51 ? r65 : r14;
    r5 = r65 * r55;
    r14 = r45 < r52 ? r5 : r14;
    r5 = r65 * r56;
    r14 = r45 < r54 ? r5 : r14;
    r14 = r45 < r50 ? r65 : r14;
    r5 = r50 * r14;
    r5 = fmaf(r58, r5, r65 * r57);
    r5 = r50 * r5;
    r5 = r5 * r59;
    r5 = r6 <= r34 ? r48 : r5;
    r65 = fmaf(r44, r5, r61);
    r62 = r33 * r1;
    r62 = r62 * r60;
    r65 = fmaf(r10, r62, r65);
    r5 = fmaf(r7, r5, r20);
    r62 = r33 * r3;
    r62 = r62 * r60;
    r5 = fmaf(r10, r62, r5);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r65,
                                          r5,
                                          r17,
                                          r69);
    r62 = r7 * r70;
    r71 = r8 * r60;
    r72 = r44 * r71;
    r62 = fmaf(r28, r72, r71 * r62);
    r73 = r7 * r36;
    r73 = fmaf(r71, r73, r39 * r72);
    r74 = r7 * r38;
    r74 = fmaf(r71, r74, r11 * r72);
    r75 = r7 * r29;
    r75 = fmaf(r12, r72, r71 * r75);
    WriteSum4<float, float>((float*)inout_shared, r62, r73, r74, r75);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r75 = r7 * r5;
    r75 = fmaf(r71, r75, r65 * r72);
    r74 = r7 * r69;
    r74 = fmaf(r17, r72, r71 * r74);
    WriteSum2<float, float>((float*)inout_shared, r75, r74);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = fmaf(r28, r28, r70 * r70);
    r75 = fmaf(r39, r39, r36 * r36);
    r73 = fmaf(r11, r11, r38 * r38);
    r62 = fmaf(r29, r29, r12 * r12);
    WriteSum4<float, float>((float*)inout_shared, r74, r75, r73, r62);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r62 = fmaf(r5, r5, r65 * r65);
    r73 = fmaf(r69, r69, r17 * r17);
    WriteSum2<float, float>((float*)inout_shared, r62, r73);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fmaf(r70, r36, r28 * r39);
    r62 = fmaf(r70, r38, r28 * r11);
    r75 = fmaf(r28, r12, r70 * r29);
    r74 = fmaf(r28, r65, r70 * r5);
    WriteSum4<float, float>((float*)inout_shared, r73, r62, r75, r74);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r28 = fmaf(r70, r69, r28 * r17);
    r74 = fmaf(r36, r38, r39 * r11);
    r75 = fmaf(r39, r12, r36 * r29);
    r62 = fmaf(r36, r5, r39 * r65);
    WriteSum4<float, float>((float*)inout_shared, r28, r74, r75, r62);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = fmaf(r39, r17, r36 * r69);
    r62 = fmaf(r11, r12, r38 * r29);
    r75 = fmaf(r11, r65, r38 * r5);
    r11 = fmaf(r11, r17, r38 * r69);
    WriteSum4<float, float>((float*)inout_shared, r39, r62, r75, r11);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = fmaf(r12, r65, r29 * r5);
    r12 = fmaf(r12, r17, r29 * r69);
    r17 = fmaf(r65, r17, r5 * r69);
    WriteSum3<float, float>((float*)inout_shared, r11, r12, r17);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r17 = r0 * r19;
    r17 = r17 * r4;
    r17 = r17 * r44;
    r12 = r2 * r4;
    r12 = fmaf(r47, r12, r10 * r17);
    r17 = r12 * r67;
    r17 = r6 <= r51 ? r12 : r17;
    r47 = r12 * r55;
    r17 = r45 < r52 ? r47 : r17;
    r47 = r12 * r56;
    r17 = r45 < r54 ? r47 : r17;
    r17 = r45 < r50 ? r12 : r17;
    r47 = r50 * r17;
    r12 = fmaf(r12, r57, r58 * r47);
    r12 = r50 * r12;
    r12 = r12 * r59;
    r12 = r6 <= r34 ? r48 : r12;
    r47 = fmaf(r44, r12, r61);
    r11 = r0 * r4;
    r11 = r11 * r60;
    r47 = fmaf(r10, r11, r47);
    r12 = fmaf(r7, r12, r20);
    r11 = r2 * r4;
    r11 = r11 * r60;
    r12 = fmaf(r10, r11, r12);
    r11 = r1 * r19;
    r11 = r11 * r9;
    r11 = r11 * r44;
    r25 = fmaf(r9, r25, r10 * r11);
    r11 = r25 * r67;
    r11 = r6 <= r51 ? r25 : r11;
    r65 = r25 * r55;
    r11 = r45 < r52 ? r65 : r11;
    r65 = r25 * r56;
    r11 = r45 < r54 ? r65 : r11;
    r11 = r45 < r50 ? r25 : r11;
    r65 = r50 * r11;
    r65 = fmaf(r58, r65, r25 * r57);
    r65 = r50 * r65;
    r65 = r65 * r59;
    r65 = r6 <= r34 ? r48 : r65;
    r25 = fmaf(r44, r65, r61);
    r75 = r1 * r9;
    r75 = r75 * r60;
    r25 = fmaf(r10, r75, r25);
    r65 = fmaf(r7, r65, r20);
    r75 = r3 * r9;
    r75 = r75 * r60;
    r65 = fmaf(r10, r75, r65);
    WriteIdx4<1024, float, float, float4>(out_focal_jac,
                                          0 * out_focal_jac_num_alloc,
                                          global_thread_idx,
                                          r47,
                                          r12,
                                          r25,
                                          r65);
    r75 = r7 * r12;
    r75 = fmaf(r47, r72, r71 * r75);
    r62 = r7 * r65;
    r62 = fmaf(r71, r62, r25 * r72);
    WriteSum2<float, float>((float*)inout_shared, r75, r62);
  };
  FlushSumShared<2, float>(out_focal_njtr,
                           0 * out_focal_njtr_num_alloc,
                           focal_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r62 = fmaf(r47, r47, r12 * r12);
    r75 = fmaf(r25, r25, r65 * r65);
    WriteSum2<float, float>((float*)inout_shared, r62, r75);
  };
  FlushSumShared<2, float>(out_focal_precond_diag,
                           0 * out_focal_precond_diag_num_alloc,
                           focal_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = fmaf(r47, r25, r12 * r65);
    WriteSum1<float, float>((float*)inout_shared, r25);
  };
  FlushSumShared<1, float>(out_focal_precond_tril,
                           0 * out_focal_precond_tril_num_alloc,
                           focal_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = r19 * r44;
    r31 = fmaf(r24, r32, r31 * r42);
    r47 = r24 * r8;
    r47 = r47 * r41;
    r75 = r33 * r18;
    r75 = fmaf(r10, r75, r40 * r47);
    r47 = fmaf(r1, r75, r0 * r31);
    r75 = fmaf(r3, r75, r2 * r31);
    r25 = fmaf(r75, r43, r47 * r25);
    r31 = r25 * r67;
    r31 = r6 <= r51 ? r25 : r31;
    r62 = r25 * r55;
    r31 = r45 < r52 ? r62 : r31;
    r62 = r25 * r56;
    r31 = r45 < r54 ? r62 : r31;
    r31 = r45 < r50 ? r25 : r31;
    r62 = r50 * r31;
    r62 = fmaf(r58, r62, r25 * r57);
    r62 = r50 * r62;
    r62 = r62 * r59;
    r62 = r6 <= r34 ? r48 : r62;
    r25 = fmaf(r44, r62, r61);
    r25 = fmaf(r60, r47, r25);
    r62 = fmaf(r7, r62, r20);
    r62 = fmaf(r60, r75, r62);
    r75 = r19 * r44;
    r23 = fmaf(r23, r42, r37 * r32);
    r47 = r33 * r27;
    r39 = r37 * r8;
    r39 = r39 * r41;
    r39 = fmaf(r40, r39, r10 * r47);
    r47 = fmaf(r1, r39, r0 * r23);
    r39 = fmaf(r3, r39, r2 * r23);
    r75 = fmaf(r39, r43, r47 * r75);
    r23 = r75 * r67;
    r23 = r6 <= r51 ? r75 : r23;
    r74 = r75 * r55;
    r23 = r45 < r52 ? r74 : r23;
    r74 = r75 * r56;
    r23 = r45 < r54 ? r74 : r23;
    r23 = r45 < r50 ? r75 : r23;
    r74 = r50 * r23;
    r75 = fmaf(r75, r57, r58 * r74);
    r75 = r50 * r75;
    r75 = r75 * r59;
    r75 = r6 <= r34 ? r48 : r75;
    r74 = fmaf(r44, r75, r61);
    r74 = fmaf(r60, r47, r74);
    r75 = fmaf(r7, r75, r20);
    r75 = fmaf(r60, r39, r75);
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r25,
                                          r62,
                                          r74,
                                          r75);
    r39 = r19 * r44;
    r42 = fmaf(r26, r42, r30 * r32);
    r26 = r30 * r8;
    r26 = r26 * r41;
    r41 = r33 * r35;
    r41 = fmaf(r10, r41, r40 * r26);
    r26 = fmaf(r1, r41, r0 * r42);
    r41 = fmaf(r3, r41, r2 * r42);
    r43 = fmaf(r41, r43, r26 * r39);
    r67 = r43 * r67;
    r67 = r6 <= r51 ? r43 : r67;
    r55 = r43 * r55;
    r67 = r45 < r52 ? r55 : r67;
    r56 = r43 * r56;
    r67 = r45 < r54 ? r56 : r67;
    r67 = r45 < r50 ? r43 : r67;
    r45 = r50 * r67;
    r45 = fmaf(r58, r45, r43 * r57);
    r45 = r50 * r45;
    r45 = r45 * r59;
    r45 = r6 <= r34 ? r48 : r45;
    r61 = fmaf(r44, r45, r61);
    r61 = fmaf(r60, r26, r61);
    r45 = fmaf(r7, r45, r20);
    r45 = fmaf(r60, r41, r45);
    WriteIdx2<1024, float, float, float2>(out_point_jac,
                                          4 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r61,
                                          r45);
    r41 = r7 * r62;
    r41 = fmaf(r25, r72, r71 * r41);
    r20 = r7 * r75;
    r20 = fmaf(r71, r20, r74 * r72);
    r26 = r7 * r45;
    r26 = fmaf(r71, r26, r61 * r72);
    WriteSum3<float, float>((float*)inout_shared, r41, r20, r26);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = fmaf(r62, r62, r25 * r25);
    r20 = fmaf(r75, r75, r74 * r74);
    r41 = fmaf(r61, r61, r45 * r45);
    WriteSum3<float, float>((float*)inout_shared, r26, r20, r41);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = fmaf(r62, r75, r25 * r74);
    r25 = fmaf(r25, r61, r62 * r45);
    r61 = fmaf(r74, r61, r75 * r45);
    WriteSum3<float, float>((float*)inout_shared, r41, r25, r61);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
}

void PinholeSplitFixedPrincipalPointResJac(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* focal,
    unsigned int focal_num_alloc,
    SharedIndex* focal_indices,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* weight_loss,
    unsigned int weight_loss_num_alloc,
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
    float* out_focal_jac,
    unsigned int out_focal_jac_num_alloc,
    float* const out_focal_njtr,
    unsigned int out_focal_njtr_num_alloc,
    float* const out_focal_precond_diag,
    unsigned int out_focal_precond_diag_num_alloc,
    float* const out_focal_precond_tril,
    unsigned int out_focal_precond_tril_num_alloc,
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
  PinholeSplitFixedPrincipalPointResJacKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      focal,
      focal_num_alloc,
      focal_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      weight_loss,
      weight_loss_num_alloc,
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
      out_focal_jac,
      out_focal_jac_num_alloc,
      out_focal_njtr,
      out_focal_njtr_num_alloc,
      out_focal_precond_diag,
      out_focal_precond_diag_num_alloc,
      out_focal_precond_tril,
      out_focal_precond_tril_num_alloc,
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