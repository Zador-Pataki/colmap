#include "kernel_pinhole_split_fixed_rotation_fixed_principal_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedRotationFixedPrincipalPointResJacFirstKernel(
        float* rotation,
        unsigned int rotation_num_alloc,
        float* translation,
        unsigned int translation_num_alloc,
        SharedIndex* translation_indices,
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
        float* const out_rTr,
        float* out_translation_jac,
        unsigned int out_translation_jac_num_alloc,
        float* const out_translation_njtr,
        unsigned int out_translation_njtr_num_alloc,
        float* const out_translation_precond_diag,
        unsigned int out_translation_precond_diag_num_alloc,
        float* const out_translation_precond_tril,
        unsigned int out_translation_precond_tril_num_alloc,
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

  __shared__ SharedIndex translation_indices_loc[1024];
  translation_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? translation_indices[global_thread_idx]
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

  __shared__ float out_rTr_local[1];

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
  LoadShared<3, float, float>(translation,
                              0 * translation_num_alloc,
                              translation_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       translation_indices_loc[threadIdx.x].target,
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
  if (global_thread_idx < problem_size) {
    r14 = -2.00000000000000000e+00;
    ReadIdx4<1024, float, float, float4>(rotation,
                                         0 * rotation_num_alloc,
                                         global_thread_idx,
                                         r15,
                                         r16,
                                         r17,
                                         r18);
    r19 = r17 * r17;
    r19 = r14 * r19;
    r20 = 1.00000000000000000e+00;
    r21 = r16 * r16;
    r21 = fmaf(r14, r21, r20);
    r22 = r19 + r21;
    r4 = fmaf(r11, r22, r4);
    r23 = r15 * r17;
    r24 = 2.00000000000000000e+00;
    r23 = r23 * r24;
    r25 = r16 * r18;
    r25 = fmaf(r24, r25, r23);
    r26 = r15 * r16;
    r26 = r26 * r24;
    r27 = r18 * r14;
    r28 = fmaf(r17, r27, r26);
    r4 = fmaf(r13, r25, r4);
    r4 = fmaf(r12, r28, r4);
  };
  LoadShared<2, float, float>(
      focal, 0 * focal_num_alloc, focal_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>(
        (float*)inout_shared, focal_indices_loc[threadIdx.x].target, r29, r30);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r31 = 9.99999999999999955e-07;
    r32 = r15 * r15;
    r32 = r14 * r32;
    r21 = r32 + r21;
    r10 = fmaf(r13, r21, r10);
    r23 = fmaf(r16, r27, r23);
    r14 = r15 * r18;
    r33 = r16 * r17;
    r33 = r33 * r24;
    r14 = fmaf(r24, r14, r33);
    r10 = fmaf(r11, r23, r10);
    r10 = fmaf(r12, r14, r10);
    r34 = copysign(1.0, r10);
    r34 = fmaf(r31, r34, r10);
    r10 = 1.0 / r34;
    r35 = r29 * r10;
    r6 = fmaf(r4, r35, r6);
    r7 = fmaf(r7, r8, r5);
    r19 = r20 + r19;
    r19 = r19 + r32;
    r12 = fmaf(r12, r19, r9);
    r27 = fmaf(r15, r27, r33);
    r33 = r17 * r18;
    r33 = fmaf(r24, r33, r26);
    r12 = fmaf(r13, r27, r12);
    r12 = fmaf(r11, r33, r12);
    r11 = r30 * r12;
    r7 = fmaf(r10, r11, r7);
    r13 = fmaf(r1, r7, r0 * r6);
    ReadIdx3<1024, float, float, float4>(weight_loss,
                                         4 * weight_loss_num_alloc,
                                         global_thread_idx,
                                         r26,
                                         r9,
                                         r32);
    r5 = 0.00000000000000000e+00;
    r32 = fmaxf(r32, r5);
    r36 = sqrtf(r32);
    r7 = fmaf(r3, r7, r2 * r6);
    r6 = fmaf(r7, r7, r13 * r13);
    r37 = 5.00000000000000000e-01;
    r9 = fmaxf(r9, r31);
    r38 = r9 * r9;
    r39 = r8 * r9;
    r40 = r24 * r9;
    r41 = fmaxf(r31, r6);
    r42 = sqrtf(r41);
    r40 = fmaf(r42, r40, r9 * r39);
    r40 = r6 <= r38 ? r6 : r40;
    r39 = 2.50000000000000000e+00;
    r42 = r9 * r9;
    r43 = 1.0 / r38;
    r43 = fmaf(r6, r43, r20);
    r44 = logf(r43);
    r42 = r42 * r44;
    r40 = r26 < r39 ? r42 : r40;
    r42 = 1.50000000000000000e+00;
    r44 = r24 * r9;
    r45 = sqrtf(r43);
    r45 = r8 + r45;
    r44 = r44 * r9;
    r44 = r44 * r45;
    r40 = r26 < r42 ? r44 : r40;
    r40 = r26 < r37 ? r6 : r40;
    r44 = fmaxf(r5, r40);
    r45 = 1.0 / r41;
    r45 = r32 * r45;
    r46 = r44 * r45;
    r47 = sqrtf(r46);
    r47 = r6 <= r31 ? r36 : r47;
    r36 = r13 * r47;
    r48 = r7 * r47;
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r36, r48);
    r48 = r13 * r13;
    r48 = r48 * r47;
    r36 = r7 * r7;
    r36 = r36 * r47;
    r36 = fmaf(r47, r36, r47 * r48);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r36);
  if (global_thread_idx < problem_size) {
    r36 = r24 * r7;
    r48 = r2 * r35;
    r49 = r0 * r24;
    r49 = r49 * r13;
    r49 = fmaf(r35, r49, r36 * r48);
    r50 = r37 * r9;
    r51 = -9.99999999999999955e-07;
    r51 = r51 + r6;
    r51 = copysign(1.0, r51);
    r51 = r20 + r51;
    r52 = rsqrtf(r41);
    r50 = r50 * r51;
    r50 = r50 * r52;
    r52 = r49 * r50;
    r52 = r6 <= r38 ? r49 : r52;
    r53 = 1.0 / r43;
    r54 = r49 * r53;
    r52 = r26 < r39 ? r54 : r52;
    r43 = rsqrtf(r43);
    r54 = r49 * r43;
    r52 = r26 < r42 ? r54 : r52;
    r52 = r26 < r37 ? r49 : r52;
    r54 = r37 * r52;
    r40 = copysign(1.0, r40);
    r40 = r20 + r40;
    r45 = r40 * r45;
    r44 = r32 * r44;
    r32 = -5.00000000000000000e-01;
    r41 = r41 * r41;
    r41 = 1.0 / r41;
    r44 = r44 * r32;
    r44 = r44 * r51;
    r44 = r44 * r41;
    r49 = fmaf(r49, r44, r45 * r54);
    r49 = r37 * r49;
    r46 = rsqrtf(r46);
    r49 = r49 * r46;
    r49 = r6 <= r31 ? r5 : r49;
    r54 = r8 * r13;
    r41 = 2.50000000000000000e-01;
    r51 = r6 <= r38 ? r5 : r5;
    r51 = r26 < r39 ? r5 : r51;
    r51 = r26 < r42 ? r5 : r51;
    r51 = r26 < r37 ? r5 : r51;
    r51 = r41 * r51;
    r51 = r51 * r46;
    r51 = r51 * r45;
    r51 = r6 <= r31 ? r5 : r51;
    r54 = r54 * r51;
    r41 = fmaf(r13, r49, r54);
    r32 = r0 * r47;
    r41 = fmaf(r35, r32, r41);
    r32 = r8 * r7;
    r32 = r32 * r51;
    r49 = fmaf(r7, r49, r32);
    r49 = fmaf(r47, r48, r49);
    r48 = r10 * r36;
    r51 = r3 * r48;
    r40 = r30 * r1;
    r40 = r40 * r24;
    r40 = r40 * r13;
    r40 = fmaf(r10, r40, r30 * r51);
    r20 = r40 * r50;
    r20 = r6 <= r38 ? r40 : r20;
    r55 = r40 * r53;
    r20 = r26 < r39 ? r55 : r20;
    r55 = r40 * r43;
    r20 = r26 < r42 ? r55 : r20;
    r20 = r26 < r37 ? r40 : r20;
    r55 = r37 * r20;
    r40 = fmaf(r40, r44, r45 * r55);
    r40 = r37 * r40;
    r40 = r40 * r46;
    r40 = r6 <= r31 ? r5 : r40;
    r55 = fmaf(r13, r40, r54);
    r56 = r30 * r1;
    r56 = r56 * r47;
    r55 = fmaf(r10, r56, r55);
    r40 = fmaf(r7, r40, r32);
    r56 = r30 * r3;
    r56 = r56 * r47;
    r40 = fmaf(r10, r56, r40);
    WriteIdx4<1024, float, float, float4>(out_translation_jac,
                                          0 * out_translation_jac_num_alloc,
                                          global_thread_idx,
                                          r41,
                                          r49,
                                          r55,
                                          r40);
    r29 = r29 * r8;
    r34 = r34 * r34;
    r34 = 1.0 / r34;
    r29 = r29 * r4;
    r29 = r29 * r34;
    r56 = r1 * r8;
    r56 = r56 * r34;
    r56 = fmaf(r11, r56, r0 * r29);
    r57 = fmaf(r47, r56, r54);
    r58 = r24 * r13;
    r59 = r3 * r8;
    r59 = r59 * r34;
    r59 = fmaf(r11, r59, r2 * r29);
    r58 = fmaf(r59, r36, r56 * r58);
    r56 = r58 * r50;
    r56 = r6 <= r38 ? r58 : r56;
    r60 = r58 * r53;
    r56 = r26 < r39 ? r60 : r56;
    r60 = r58 * r43;
    r56 = r26 < r42 ? r60 : r56;
    r56 = r26 < r37 ? r58 : r56;
    r60 = r37 * r56;
    r60 = fmaf(r45, r60, r58 * r44);
    r60 = r37 * r60;
    r60 = r60 * r46;
    r60 = r6 <= r31 ? r5 : r60;
    r57 = fmaf(r13, r60, r57);
    r60 = fmaf(r7, r60, r32);
    r60 = fmaf(r47, r59, r60);
    WriteIdx2<1024, float, float, float2>(out_translation_jac,
                                          4 * out_translation_jac_num_alloc,
                                          global_thread_idx,
                                          r57,
                                          r60);
    r59 = r8 * r47;
    r58 = r13 * r59;
    r61 = r7 * r49;
    r61 = fmaf(r59, r61, r41 * r58);
    r62 = r7 * r40;
    r62 = fmaf(r55, r58, r59 * r62);
    r63 = r7 * r60;
    r63 = fmaf(r57, r58, r59 * r63);
    WriteSum3<float, float>((float*)inout_shared, r61, r62, r63);
  };
  FlushSumShared<3, float>(out_translation_njtr,
                           0 * out_translation_njtr_num_alloc,
                           translation_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r63 = fmaf(r41, r41, r49 * r49);
    r62 = fmaf(r55, r55, r40 * r40);
    r61 = fmaf(r60, r60, r57 * r57);
    WriteSum3<float, float>((float*)inout_shared, r63, r62, r61);
  };
  FlushSumShared<3, float>(out_translation_precond_diag,
                           0 * out_translation_precond_diag_num_alloc,
                           translation_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r61 = fmaf(r41, r55, r49 * r40);
    r41 = fmaf(r49, r60, r41 * r57);
    r57 = fmaf(r55, r57, r40 * r60);
    WriteSum3<float, float>((float*)inout_shared, r61, r41, r57);
  };
  FlushSumShared<3, float>(out_translation_precond_tril,
                           0 * out_translation_precond_tril_num_alloc,
                           translation_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r57 = r0 * r24;
    r57 = r57 * r4;
    r57 = r57 * r13;
    r41 = r2 * r4;
    r41 = fmaf(r48, r41, r10 * r57);
    r57 = r41 * r50;
    r57 = r6 <= r38 ? r41 : r57;
    r48 = r41 * r53;
    r57 = r26 < r39 ? r48 : r57;
    r48 = r41 * r43;
    r57 = r26 < r42 ? r48 : r57;
    r57 = r26 < r37 ? r41 : r57;
    r48 = r37 * r57;
    r48 = fmaf(r45, r48, r41 * r44);
    r48 = r37 * r48;
    r48 = r48 * r46;
    r48 = r6 <= r31 ? r5 : r48;
    r41 = fmaf(r13, r48, r54);
    r61 = r0 * r4;
    r61 = r61 * r47;
    r41 = fmaf(r10, r61, r41);
    r48 = fmaf(r7, r48, r32);
    r61 = r2 * r4;
    r61 = r61 * r47;
    r48 = fmaf(r10, r61, r48);
    r61 = r1 * r24;
    r61 = r61 * r12;
    r61 = r61 * r13;
    r51 = fmaf(r12, r51, r10 * r61);
    r61 = r51 * r50;
    r61 = r6 <= r38 ? r51 : r61;
    r55 = r51 * r53;
    r61 = r26 < r39 ? r55 : r61;
    r55 = r51 * r43;
    r61 = r26 < r42 ? r55 : r61;
    r61 = r26 < r37 ? r51 : r61;
    r55 = r37 * r61;
    r55 = fmaf(r45, r55, r51 * r44);
    r55 = r37 * r55;
    r55 = r55 * r46;
    r55 = r6 <= r31 ? r5 : r55;
    r51 = fmaf(r13, r55, r54);
    r62 = r1 * r12;
    r62 = r62 * r47;
    r51 = fmaf(r10, r62, r51);
    r55 = fmaf(r7, r55, r32);
    r62 = r3 * r12;
    r62 = r62 * r47;
    r55 = fmaf(r10, r62, r55);
    WriteIdx4<1024, float, float, float4>(out_focal_jac,
                                          0 * out_focal_jac_num_alloc,
                                          global_thread_idx,
                                          r41,
                                          r48,
                                          r51,
                                          r55);
    r62 = r7 * r48;
    r62 = fmaf(r41, r58, r59 * r62);
    r63 = r7 * r55;
    r63 = fmaf(r59, r63, r51 * r58);
    WriteSum2<float, float>((float*)inout_shared, r62, r63);
  };
  FlushSumShared<2, float>(out_focal_njtr,
                           0 * out_focal_njtr_num_alloc,
                           focal_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r63 = fmaf(r48, r48, r41 * r41);
    r62 = fmaf(r55, r55, r51 * r51);
    WriteSum2<float, float>((float*)inout_shared, r63, r62);
  };
  FlushSumShared<2, float>(out_focal_precond_diag,
                           0 * out_focal_precond_diag_num_alloc,
                           focal_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r51 = fmaf(r48, r55, r41 * r51);
    WriteSum1<float, float>((float*)inout_shared, r51);
  };
  FlushSumShared<1, float>(out_focal_precond_tril,
                           0 * out_focal_precond_tril_num_alloc,
                           focal_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r51 = r24 * r13;
    r41 = r30 * r33;
    r62 = r23 * r8;
    r62 = r62 * r34;
    r62 = fmaf(r11, r62, r10 * r41);
    r22 = fmaf(r23, r29, r22 * r35);
    r41 = fmaf(r0, r22, r1 * r62);
    r22 = fmaf(r2, r22, r3 * r62);
    r51 = fmaf(r22, r36, r41 * r51);
    r62 = r51 * r50;
    r62 = r6 <= r38 ? r51 : r62;
    r63 = r51 * r53;
    r62 = r26 < r39 ? r63 : r62;
    r63 = r51 * r43;
    r62 = r26 < r42 ? r63 : r62;
    r62 = r26 < r37 ? r51 : r62;
    r63 = r37 * r62;
    r51 = fmaf(r51, r44, r45 * r63);
    r51 = r37 * r51;
    r51 = r51 * r46;
    r51 = r6 <= r31 ? r5 : r51;
    r63 = fmaf(r13, r51, r54);
    r63 = fmaf(r47, r41, r63);
    r51 = fmaf(r7, r51, r32);
    r51 = fmaf(r47, r22, r51);
    r22 = r30 * r19;
    r41 = r14 * r8;
    r41 = r41 * r34;
    r41 = fmaf(r11, r41, r10 * r22);
    r28 = fmaf(r14, r29, r28 * r35);
    r22 = fmaf(r0, r28, r1 * r41);
    r64 = fmaf(r47, r22, r54);
    r65 = r24 * r13;
    r28 = fmaf(r2, r28, r3 * r41);
    r65 = fmaf(r28, r36, r22 * r65);
    r22 = r65 * r50;
    r22 = r6 <= r38 ? r65 : r22;
    r41 = r65 * r53;
    r22 = r26 < r39 ? r41 : r22;
    r41 = r65 * r43;
    r22 = r26 < r42 ? r41 : r22;
    r22 = r26 < r37 ? r65 : r22;
    r41 = r37 * r22;
    r65 = fmaf(r65, r44, r45 * r41);
    r65 = r37 * r65;
    r65 = r65 * r46;
    r65 = r6 <= r31 ? r5 : r65;
    r64 = fmaf(r13, r65, r64);
    r28 = fmaf(r47, r28, r32);
    r28 = fmaf(r7, r65, r28);
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r63,
                                          r51,
                                          r64,
                                          r28);
    r65 = r24 * r13;
    r41 = r30 * r27;
    r66 = r21 * r8;
    r66 = r66 * r34;
    r66 = fmaf(r11, r66, r10 * r41);
    r29 = fmaf(r21, r29, r25 * r35);
    r35 = fmaf(r0, r29, r1 * r66);
    r29 = fmaf(r2, r29, r3 * r66);
    r36 = fmaf(r29, r36, r35 * r65);
    r50 = r36 * r50;
    r50 = r6 <= r38 ? r36 : r50;
    r53 = r36 * r53;
    r50 = r26 < r39 ? r53 : r50;
    r43 = r36 * r43;
    r50 = r26 < r42 ? r43 : r50;
    r50 = r26 < r37 ? r36 : r50;
    r26 = r37 * r50;
    r26 = fmaf(r45, r26, r36 * r44);
    r26 = r37 * r26;
    r26 = r26 * r46;
    r26 = r6 <= r31 ? r5 : r26;
    r54 = fmaf(r13, r26, r54);
    r54 = fmaf(r47, r35, r54);
    r26 = fmaf(r7, r26, r32);
    r26 = fmaf(r47, r29, r26);
    WriteIdx2<1024, float, float, float2>(out_point_jac,
                                          4 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r54,
                                          r26);
    r29 = r7 * r51;
    r29 = fmaf(r59, r29, r63 * r58);
    r32 = r7 * r28;
    r32 = fmaf(r59, r32, r64 * r58);
    r35 = r7 * r26;
    r35 = fmaf(r59, r35, r54 * r58);
    WriteSum3<float, float>((float*)inout_shared, r29, r32, r35);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = fmaf(r63, r63, r51 * r51);
    r32 = fmaf(r28, r28, r64 * r64);
    r29 = fmaf(r54, r54, r26 * r26);
    WriteSum3<float, float>((float*)inout_shared, r35, r32, r29);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r29 = fmaf(r51, r28, r63 * r64);
    r63 = fmaf(r63, r54, r51 * r26);
    r54 = fmaf(r28, r26, r64 * r54);
    WriteSum3<float, float>((float*)inout_shared, r29, r63, r54);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void PinholeSplitFixedRotationFixedPrincipalPointResJacFirst(
    float* rotation,
    unsigned int rotation_num_alloc,
    float* translation,
    unsigned int translation_num_alloc,
    SharedIndex* translation_indices,
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
    float* const out_rTr,
    float* out_translation_jac,
    unsigned int out_translation_jac_num_alloc,
    float* const out_translation_njtr,
    unsigned int out_translation_njtr_num_alloc,
    float* const out_translation_precond_diag,
    unsigned int out_translation_precond_diag_num_alloc,
    float* const out_translation_precond_tril,
    unsigned int out_translation_precond_tril_num_alloc,
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
  PinholeSplitFixedRotationFixedPrincipalPointResJacFirstKernel<<<n_blocks,
                                                                  1024>>>(
      rotation,
      rotation_num_alloc,
      translation,
      translation_num_alloc,
      translation_indices,
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
      out_rTr,
      out_translation_jac,
      out_translation_jac_num_alloc,
      out_translation_njtr,
      out_translation_njtr_num_alloc,
      out_translation_precond_diag,
      out_translation_precond_diag_num_alloc,
      out_translation_precond_tril,
      out_translation_precond_tril_num_alloc,
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