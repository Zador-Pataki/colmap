#include "kernel_pinhole_fixed_rotation_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeFixedRotationResJacFirstKernel(
        float* rotation,
        unsigned int rotation_num_alloc,
        float* translation,
        unsigned int translation_num_alloc,
        SharedIndex* translation_indices,
        float* calib,
        unsigned int calib_num_alloc,
        SharedIndex* calib_indices,
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* weight_loss,
        unsigned int weight_loss_num_alloc,
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

  __shared__ SharedIndex translation_indices_loc[1024];
  translation_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? translation_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
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

  __shared__ float out_rTr_local[1];

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
  LoadShared<3, float, float>(translation,
                              0 * translation_num_alloc,
                              translation_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       translation_indices_loc[threadIdx.x].target,
                       r6,
                       r11,
                       r12);
  };
  __syncthreads();
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
    r16 = -2.00000000000000000e+00;
    ReadIdx4<1024, float, float, float4>(rotation,
                                         0 * rotation_num_alloc,
                                         global_thread_idx,
                                         r17,
                                         r18,
                                         r19,
                                         r20);
    r21 = r19 * r19;
    r21 = r16 * r21;
    r22 = 1.00000000000000000e+00;
    r23 = r18 * r18;
    r23 = fmaf(r16, r23, r22);
    r24 = r21 + r23;
    r6 = fmaf(r13, r24, r6);
    r25 = r17 * r19;
    r26 = 2.00000000000000000e+00;
    r25 = r25 * r26;
    r27 = r18 * r20;
    r27 = fmaf(r26, r27, r25);
    r28 = r17 * r18;
    r28 = r28 * r26;
    r29 = r20 * r16;
    r30 = fmaf(r19, r29, r28);
    r6 = fmaf(r15, r27, r6);
    r6 = fmaf(r14, r30, r6);
    r31 = 9.99999999999999955e-07;
    r32 = r17 * r17;
    r32 = r16 * r32;
    r23 = r32 + r23;
    r12 = fmaf(r15, r23, r12);
    r25 = fmaf(r18, r29, r25);
    r16 = r17 * r20;
    r33 = r18 * r19;
    r33 = r33 * r26;
    r16 = fmaf(r26, r16, r33);
    r12 = fmaf(r13, r25, r12);
    r12 = fmaf(r14, r16, r12);
    r34 = copysign(1.0, r12);
    r34 = fmaf(r31, r34, r12);
    r12 = 1.0 / r34;
    r35 = r4 * r12;
    r8 = fmaf(r6, r35, r8);
    r9 = fmaf(r9, r10, r7);
    r21 = r22 + r21;
    r21 = r21 + r32;
    r14 = fmaf(r14, r21, r11);
    r29 = fmaf(r17, r29, r33);
    r33 = r19 * r20;
    r33 = fmaf(r26, r33, r28);
    r14 = fmaf(r15, r29, r14);
    r14 = fmaf(r13, r33, r14);
    r13 = r5 * r14;
    r9 = fmaf(r12, r13, r9);
    r15 = fmaf(r1, r9, r0 * r8);
    ReadIdx3<1024, float, float, float4>(weight_loss,
                                         4 * weight_loss_num_alloc,
                                         global_thread_idx,
                                         r28,
                                         r11,
                                         r32);
    r7 = 0.00000000000000000e+00;
    r32 = fmaxf(r32, r7);
    r36 = sqrtf(r32);
    r9 = fmaf(r3, r9, r2 * r8);
    r8 = fmaf(r15, r15, r9 * r9);
    r37 = 5.00000000000000000e-01;
    r11 = fmaxf(r11, r31);
    r38 = r11 * r11;
    r39 = r26 * r11;
    r40 = fmaxf(r31, r8);
    r41 = sqrtf(r40);
    r42 = r10 * r11;
    r42 = fmaf(r11, r42, r41 * r39);
    r42 = r8 <= r38 ? r8 : r42;
    r39 = 2.50000000000000000e+00;
    r41 = r11 * r11;
    r43 = 1.0 / r38;
    r43 = fmaf(r8, r43, r22);
    r44 = logf(r43);
    r41 = r41 * r44;
    r42 = r28 < r39 ? r41 : r42;
    r41 = 1.50000000000000000e+00;
    r44 = r26 * r11;
    r45 = sqrtf(r43);
    r45 = r10 + r45;
    r44 = r44 * r11;
    r44 = r44 * r45;
    r42 = r28 < r41 ? r44 : r42;
    r42 = r28 < r37 ? r8 : r42;
    r44 = fmaxf(r7, r42);
    r45 = 1.0 / r40;
    r45 = r32 * r45;
    r46 = r44 * r45;
    r47 = sqrtf(r46);
    r47 = r8 <= r31 ? r36 : r47;
    r36 = r15 * r47;
    r48 = r9 * r47;
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r36, r48);
    r48 = r15 * r15;
    r48 = r48 * r47;
    r36 = r9 * r9;
    r36 = r36 * r47;
    r36 = fmaf(r47, r36, r47 * r48);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r36);
  if (global_thread_idx < problem_size) {
    r36 = r10 * r15;
    r48 = 2.50000000000000000e-01;
    r49 = r8 <= r38 ? r7 : r7;
    r49 = r28 < r39 ? r7 : r49;
    r49 = r28 < r41 ? r7 : r49;
    r49 = r28 < r37 ? r7 : r49;
    r49 = r48 * r49;
    r46 = rsqrtf(r46);
    r42 = copysign(1.0, r42);
    r42 = r22 + r42;
    r45 = r42 * r45;
    r49 = r49 * r46;
    r49 = r49 * r45;
    r49 = r8 <= r31 ? r7 : r49;
    r36 = r36 * r49;
    r42 = r26 * r15;
    r48 = r0 * r35;
    r50 = r2 * r26;
    r50 = r50 * r9;
    r50 = fmaf(r35, r50, r42 * r48);
    r51 = r37 * r11;
    r52 = -9.99999999999999955e-07;
    r52 = r52 + r8;
    r52 = copysign(1.0, r52);
    r52 = r22 + r52;
    r22 = rsqrtf(r40);
    r51 = r51 * r52;
    r51 = r51 * r22;
    r22 = r50 * r51;
    r22 = r8 <= r38 ? r50 : r22;
    r53 = 1.0 / r43;
    r54 = r50 * r53;
    r22 = r28 < r39 ? r54 : r22;
    r43 = rsqrtf(r43);
    r54 = r50 * r43;
    r22 = r28 < r41 ? r54 : r22;
    r22 = r28 < r37 ? r50 : r22;
    r54 = r37 * r22;
    r44 = r32 * r44;
    r32 = -5.00000000000000000e-01;
    r40 = r40 * r40;
    r40 = 1.0 / r40;
    r44 = r44 * r32;
    r44 = r44 * r52;
    r44 = r44 * r40;
    r50 = fmaf(r50, r44, r45 * r54);
    r50 = r37 * r50;
    r50 = r50 * r46;
    r50 = r8 <= r31 ? r7 : r50;
    r54 = fmaf(r15, r50, r36);
    r54 = fmaf(r47, r48, r54);
    r48 = r10 * r9;
    r48 = r48 * r49;
    r50 = fmaf(r9, r50, r48);
    r49 = r2 * r47;
    r50 = fmaf(r35, r49, r50);
    r49 = r5 * r3;
    r49 = r49 * r26;
    r49 = r49 * r9;
    r40 = r1 * r26;
    r40 = r40 * r15;
    r52 = r12 * r40;
    r49 = fmaf(r5, r52, r12 * r49);
    r32 = r49 * r51;
    r32 = r8 <= r38 ? r49 : r32;
    r55 = r49 * r53;
    r32 = r28 < r39 ? r55 : r32;
    r55 = r49 * r43;
    r32 = r28 < r41 ? r55 : r32;
    r32 = r28 < r37 ? r49 : r32;
    r55 = r37 * r32;
    r55 = fmaf(r45, r55, r49 * r44);
    r55 = r37 * r55;
    r55 = r55 * r46;
    r55 = r8 <= r31 ? r7 : r55;
    r49 = fmaf(r15, r55, r36);
    r56 = r5 * r1;
    r56 = r56 * r47;
    r49 = fmaf(r12, r56, r49);
    r55 = fmaf(r9, r55, r48);
    r56 = r5 * r3;
    r56 = r56 * r47;
    r55 = fmaf(r12, r56, r55);
    WriteIdx4<1024, float, float, float4>(out_translation_jac,
                                          0 * out_translation_jac_num_alloc,
                                          global_thread_idx,
                                          r54,
                                          r50,
                                          r49,
                                          r55);
    r56 = r1 * r10;
    r34 = r34 * r34;
    r34 = 1.0 / r34;
    r56 = r56 * r34;
    r4 = r4 * r10;
    r4 = r4 * r6;
    r4 = r4 * r34;
    r56 = fmaf(r0, r4, r13 * r56);
    r57 = fmaf(r47, r56, r36);
    r58 = r26 * r9;
    r59 = r3 * r10;
    r59 = r59 * r34;
    r59 = fmaf(r2, r4, r13 * r59);
    r56 = fmaf(r56, r42, r59 * r58);
    r58 = r56 * r51;
    r58 = r8 <= r38 ? r56 : r58;
    r60 = r56 * r53;
    r58 = r28 < r39 ? r60 : r58;
    r60 = r56 * r43;
    r58 = r28 < r41 ? r60 : r58;
    r58 = r28 < r37 ? r56 : r58;
    r60 = r37 * r58;
    r60 = fmaf(r45, r60, r56 * r44);
    r60 = r37 * r60;
    r60 = r60 * r46;
    r60 = r8 <= r31 ? r7 : r60;
    r57 = fmaf(r15, r60, r57);
    r60 = fmaf(r9, r60, r48);
    r60 = fmaf(r47, r59, r60);
    WriteIdx2<1024, float, float, float2>(out_translation_jac,
                                          4 * out_translation_jac_num_alloc,
                                          global_thread_idx,
                                          r57,
                                          r60);
    r59 = r15 * r54;
    r56 = r10 * r47;
    r61 = r9 * r56;
    r59 = fmaf(r50, r61, r56 * r59);
    r62 = r15 * r49;
    r62 = fmaf(r56, r62, r55 * r61);
    r63 = r15 * r57;
    r63 = fmaf(r56, r63, r60 * r61);
    WriteSum3<float, float>((float*)inout_shared, r59, r62, r63);
  };
  FlushSumShared<3, float>(out_translation_njtr,
                           0 * out_translation_njtr_num_alloc,
                           translation_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r63 = fmaf(r54, r54, r50 * r50);
    r62 = fmaf(r49, r49, r55 * r55);
    r59 = fmaf(r60, r60, r57 * r57);
    WriteSum3<float, float>((float*)inout_shared, r63, r62, r59);
  };
  FlushSumShared<3, float>(out_translation_precond_diag,
                           0 * out_translation_precond_diag_num_alloc,
                           translation_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r59 = fmaf(r54, r49, r50 * r55);
    r50 = fmaf(r54, r57, r50 * r60);
    r60 = fmaf(r49, r57, r55 * r60);
    WriteSum3<float, float>((float*)inout_shared, r59, r50, r60);
  };
  FlushSumShared<3, float>(out_translation_precond_tril,
                           0 * out_translation_precond_tril_num_alloc,
                           translation_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r60 = r2 * r26;
    r60 = r60 * r6;
    r60 = r60 * r9;
    r50 = r0 * r6;
    r50 = r50 * r12;
    r50 = fmaf(r42, r50, r12 * r60);
    r60 = r50 * r51;
    r60 = r8 <= r38 ? r50 : r60;
    r59 = r50 * r53;
    r60 = r28 < r39 ? r59 : r60;
    r59 = r50 * r43;
    r60 = r28 < r41 ? r59 : r60;
    r60 = r28 < r37 ? r50 : r60;
    r59 = r37 * r60;
    r59 = fmaf(r45, r59, r50 * r44);
    r59 = r37 * r59;
    r59 = r59 * r46;
    r59 = r8 <= r31 ? r7 : r59;
    r50 = fmaf(r15, r59, r36);
    r55 = r0 * r6;
    r55 = r55 * r47;
    r50 = fmaf(r12, r55, r50);
    r59 = fmaf(r9, r59, r48);
    r55 = r2 * r6;
    r55 = r55 * r47;
    r59 = fmaf(r12, r55, r59);
    r55 = r3 * r26;
    r55 = r55 * r14;
    r55 = r55 * r9;
    r55 = fmaf(r12, r55, r14 * r52);
    r52 = r55 * r51;
    r52 = r8 <= r38 ? r55 : r52;
    r62 = r55 * r53;
    r52 = r28 < r39 ? r62 : r52;
    r62 = r55 * r43;
    r52 = r28 < r41 ? r62 : r52;
    r52 = r28 < r37 ? r55 : r52;
    r62 = r37 * r52;
    r55 = fmaf(r55, r44, r45 * r62);
    r55 = r37 * r55;
    r55 = r55 * r46;
    r55 = r8 <= r31 ? r7 : r55;
    r62 = fmaf(r15, r55, r36);
    r63 = r1 * r14;
    r63 = r63 * r47;
    r62 = fmaf(r12, r63, r62);
    r55 = fmaf(r9, r55, r48);
    r63 = r3 * r14;
    r63 = r63 * r47;
    r55 = fmaf(r12, r63, r55);
    WriteIdx4<1024, float, float, float4>(out_calib_jac,
                                          0 * out_calib_jac_num_alloc,
                                          global_thread_idx,
                                          r50,
                                          r59,
                                          r62,
                                          r55);
    r63 = r2 * r26;
    r63 = fmaf(r0, r42, r9 * r63);
    r64 = r63 * r51;
    r64 = r8 <= r38 ? r63 : r64;
    r65 = r63 * r53;
    r64 = r28 < r39 ? r65 : r64;
    r65 = r63 * r43;
    r64 = r28 < r41 ? r65 : r64;
    r64 = r28 < r37 ? r63 : r64;
    r65 = r37 * r64;
    r63 = fmaf(r63, r44, r45 * r65);
    r63 = r37 * r63;
    r63 = r63 * r46;
    r63 = r8 <= r31 ? r7 : r63;
    r65 = fmaf(r15, r63, r36);
    r65 = fmaf(r0, r47, r65);
    r63 = fmaf(r9, r63, r48);
    r63 = fmaf(r2, r47, r63);
    r66 = fmaf(r1, r47, r36);
    r67 = r3 * r26;
    r67 = fmaf(r9, r67, r40);
    r40 = r67 * r51;
    r40 = r8 <= r38 ? r67 : r40;
    r68 = r67 * r53;
    r40 = r28 < r39 ? r68 : r40;
    r68 = r67 * r43;
    r40 = r28 < r41 ? r68 : r40;
    r40 = r28 < r37 ? r67 : r40;
    r68 = r37 * r40;
    r68 = fmaf(r45, r68, r67 * r44);
    r68 = r37 * r68;
    r68 = r68 * r46;
    r68 = r8 <= r31 ? r7 : r68;
    r66 = fmaf(r15, r68, r66);
    r67 = fmaf(r3, r47, r48);
    r67 = fmaf(r9, r68, r67);
    WriteIdx4<1024, float, float, float4>(out_calib_jac,
                                          4 * out_calib_jac_num_alloc,
                                          global_thread_idx,
                                          r65,
                                          r63,
                                          r66,
                                          r67);
    r68 = r15 * r50;
    r68 = fmaf(r59, r61, r56 * r68);
    r69 = r15 * r62;
    r69 = fmaf(r55, r61, r56 * r69);
    r70 = r15 * r65;
    r70 = fmaf(r56, r70, r63 * r61);
    r71 = r15 * r66;
    r71 = fmaf(r56, r71, r67 * r61);
    WriteSum4<float, float>((float*)inout_shared, r68, r69, r70, r71);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           0 * out_calib_njtr_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r71 = fmaf(r50, r50, r59 * r59);
    r70 = fmaf(r55, r55, r62 * r62);
    r69 = fmaf(r65, r65, r63 * r63);
    r68 = fmaf(r67, r67, r66 * r66);
    WriteSum4<float, float>((float*)inout_shared, r71, r70, r69, r68);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           0 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r68 = fmaf(r59, r55, r50 * r62);
    r69 = fmaf(r50, r65, r59 * r63);
    r59 = fmaf(r59, r67, r50 * r66);
    r70 = fmaf(r55, r63, r62 * r65);
    WriteSum4<float, float>((float*)inout_shared, r68, r69, r59, r70);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           0 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r55 = fmaf(r55, r67, r62 * r66);
    r67 = fmaf(r63, r67, r65 * r66);
    WriteSum2<float, float>((float*)inout_shared, r55, r67);
  };
  FlushSumShared<2, float>(out_calib_precond_tril,
                           4 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r67 = r26 * r9;
    r55 = r10 * r25;
    r55 = r55 * r34;
    r63 = r5 * r33;
    r63 = fmaf(r12, r63, r13 * r55);
    r24 = fmaf(r24, r35, r25 * r4);
    r55 = fmaf(r2, r24, r3 * r63);
    r24 = fmaf(r0, r24, r1 * r63);
    r67 = fmaf(r24, r42, r55 * r67);
    r63 = r67 * r51;
    r63 = r8 <= r38 ? r67 : r63;
    r70 = r67 * r53;
    r63 = r28 < r39 ? r70 : r63;
    r70 = r67 * r43;
    r63 = r28 < r41 ? r70 : r63;
    r63 = r28 < r37 ? r67 : r63;
    r70 = r37 * r63;
    r70 = fmaf(r45, r70, r67 * r44);
    r70 = r37 * r70;
    r70 = r70 * r46;
    r70 = r8 <= r31 ? r7 : r70;
    r67 = fmaf(r15, r70, r36);
    r67 = fmaf(r47, r24, r67);
    r70 = fmaf(r9, r70, r48);
    r70 = fmaf(r47, r55, r70);
    r55 = r26 * r9;
    r24 = r10 * r16;
    r24 = r24 * r34;
    r59 = r5 * r21;
    r59 = fmaf(r12, r59, r13 * r24);
    r30 = fmaf(r30, r35, r16 * r4);
    r24 = fmaf(r2, r30, r3 * r59);
    r30 = fmaf(r0, r30, r1 * r59);
    r55 = fmaf(r30, r42, r24 * r55);
    r59 = r55 * r51;
    r59 = r8 <= r38 ? r55 : r59;
    r69 = r55 * r53;
    r59 = r28 < r39 ? r69 : r59;
    r69 = r55 * r43;
    r59 = r28 < r41 ? r69 : r59;
    r59 = r28 < r37 ? r55 : r59;
    r69 = r37 * r59;
    r55 = fmaf(r55, r44, r45 * r69);
    r55 = r37 * r55;
    r55 = r55 * r46;
    r55 = r8 <= r31 ? r7 : r55;
    r69 = fmaf(r15, r55, r36);
    r69 = fmaf(r47, r30, r69);
    r55 = fmaf(r9, r55, r48);
    r55 = fmaf(r47, r24, r55);
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r67,
                                          r70,
                                          r69,
                                          r55);
    r24 = r26 * r9;
    r30 = r5 * r29;
    r68 = r10 * r23;
    r68 = r68 * r34;
    r68 = fmaf(r13, r68, r12 * r30);
    r4 = fmaf(r23, r4, r27 * r35);
    r35 = fmaf(r2, r4, r3 * r68);
    r4 = fmaf(r0, r4, r1 * r68);
    r42 = fmaf(r4, r42, r35 * r24);
    r51 = r42 * r51;
    r51 = r8 <= r38 ? r42 : r51;
    r53 = r42 * r53;
    r51 = r28 < r39 ? r53 : r51;
    r43 = r42 * r43;
    r51 = r28 < r41 ? r43 : r51;
    r51 = r28 < r37 ? r42 : r51;
    r28 = r37 * r51;
    r28 = fmaf(r45, r28, r42 * r44);
    r28 = r37 * r28;
    r28 = r28 * r46;
    r28 = r8 <= r31 ? r7 : r28;
    r36 = fmaf(r15, r28, r36);
    r36 = fmaf(r47, r4, r36);
    r28 = fmaf(r9, r28, r48);
    r28 = fmaf(r47, r35, r28);
    WriteIdx2<1024, float, float, float2>(out_point_jac,
                                          4 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r36,
                                          r28);
    r35 = r15 * r67;
    r35 = fmaf(r70, r61, r56 * r35);
    r48 = r15 * r69;
    r48 = fmaf(r55, r61, r56 * r48);
    r4 = r15 * r36;
    r61 = fmaf(r28, r61, r56 * r4);
    WriteSum3<float, float>((float*)inout_shared, r35, r48, r61);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r61 = fmaf(r70, r70, r67 * r67);
    r48 = fmaf(r55, r55, r69 * r69);
    r35 = fmaf(r28, r28, r36 * r36);
    WriteSum3<float, float>((float*)inout_shared, r61, r48, r35);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = fmaf(r70, r55, r67 * r69);
    r70 = fmaf(r70, r28, r67 * r36);
    r28 = fmaf(r55, r28, r69 * r36);
    WriteSum3<float, float>((float*)inout_shared, r35, r70, r28);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void PinholeFixedRotationResJacFirst(
    float* rotation,
    unsigned int rotation_num_alloc,
    float* translation,
    unsigned int translation_num_alloc,
    SharedIndex* translation_indices,
    float* calib,
    unsigned int calib_num_alloc,
    SharedIndex* calib_indices,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* weight_loss,
    unsigned int weight_loss_num_alloc,
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
  PinholeFixedRotationResJacFirstKernel<<<n_blocks, 1024>>>(
      rotation,
      rotation_num_alloc,
      translation,
      translation_num_alloc,
      translation_indices,
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