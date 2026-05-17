#include "kernel_pinhole_fixed_rotation_fixed_calib_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeFixedRotationFixedCalibResJacFirstKernel(
        float* rotation,
        unsigned int rotation_num_alloc,
        float* translation,
        unsigned int translation_num_alloc,
        SharedIndex* translation_indices,
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* weight_loss,
        unsigned int weight_loss_num_alloc,
        float* calib,
        unsigned int calib_num_alloc,
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
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(weight_loss,
                                         0 * weight_loss_num_alloc,
                                         global_thread_idx,
                                         r0,
                                         r1,
                                         r2,
                                         r3);
    ReadIdx4<1024, float, float, float4>(
        calib, 0 * calib_num_alloc, global_thread_idx, r4, r5, r6, r7);
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
    r14 = r5 * r14;
    r9 = fmaf(r12, r14, r9);
    r13 = fmaf(r1, r9, r0 * r8);
    ReadIdx3<1024, float, float, float4>(weight_loss,
                                         4 * weight_loss_num_alloc,
                                         global_thread_idx,
                                         r15,
                                         r28,
                                         r11);
    r32 = 0.00000000000000000e+00;
    r11 = fmaxf(r11, r32);
    r7 = sqrtf(r11);
    r9 = fmaf(r3, r9, r2 * r8);
    r8 = fmaf(r13, r13, r9 * r9);
    r36 = 5.00000000000000000e-01;
    r28 = fmaxf(r28, r31);
    r37 = r28 * r28;
    r38 = r26 * r28;
    r39 = fmaxf(r31, r8);
    r40 = sqrtf(r39);
    r41 = r10 * r28;
    r41 = fmaf(r28, r41, r40 * r38);
    r41 = r8 <= r37 ? r8 : r41;
    r38 = 2.50000000000000000e+00;
    r40 = r28 * r28;
    r42 = 1.0 / r37;
    r42 = fmaf(r8, r42, r22);
    r43 = logf(r42);
    r40 = r40 * r43;
    r41 = r15 < r38 ? r40 : r41;
    r40 = 1.50000000000000000e+00;
    r43 = r26 * r28;
    r44 = sqrtf(r42);
    r44 = r10 + r44;
    r43 = r43 * r28;
    r43 = r43 * r44;
    r41 = r15 < r40 ? r43 : r41;
    r41 = r15 < r36 ? r8 : r41;
    r43 = fmaxf(r32, r41);
    r44 = 1.0 / r39;
    r44 = r11 * r44;
    r45 = r43 * r44;
    r46 = sqrtf(r45);
    r46 = r8 <= r31 ? r7 : r46;
    r7 = r13 * r46;
    r47 = r9 * r46;
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r7, r47);
    r47 = r13 * r13;
    r47 = r47 * r46;
    r7 = r9 * r9;
    r7 = r7 * r46;
    r7 = fmaf(r46, r7, r46 * r47);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r7);
  if (global_thread_idx < problem_size) {
    r7 = r10 * r13;
    r47 = 2.50000000000000000e-01;
    r48 = r8 <= r37 ? r32 : r32;
    r48 = r15 < r38 ? r32 : r48;
    r48 = r15 < r40 ? r32 : r48;
    r48 = r15 < r36 ? r32 : r48;
    r48 = r47 * r48;
    r45 = rsqrtf(r45);
    r41 = copysign(1.0, r41);
    r41 = r22 + r41;
    r44 = r41 * r44;
    r48 = r48 * r45;
    r48 = r48 * r44;
    r48 = r8 <= r31 ? r32 : r48;
    r7 = r7 * r48;
    r41 = r26 * r13;
    r47 = r0 * r35;
    r49 = r2 * r26;
    r49 = r49 * r9;
    r49 = fmaf(r35, r49, r41 * r47);
    r50 = r36 * r28;
    r51 = -9.99999999999999955e-07;
    r51 = r51 + r8;
    r51 = copysign(1.0, r51);
    r51 = r22 + r51;
    r22 = rsqrtf(r39);
    r50 = r50 * r51;
    r50 = r50 * r22;
    r22 = r49 * r50;
    r22 = r8 <= r37 ? r49 : r22;
    r52 = 1.0 / r42;
    r53 = r49 * r52;
    r22 = r15 < r38 ? r53 : r22;
    r42 = rsqrtf(r42);
    r53 = r49 * r42;
    r22 = r15 < r40 ? r53 : r22;
    r22 = r15 < r36 ? r49 : r22;
    r53 = r36 * r22;
    r43 = r11 * r43;
    r11 = -5.00000000000000000e-01;
    r39 = r39 * r39;
    r39 = 1.0 / r39;
    r43 = r43 * r11;
    r43 = r43 * r51;
    r43 = r43 * r39;
    r49 = fmaf(r49, r43, r44 * r53);
    r49 = r36 * r49;
    r49 = r49 * r45;
    r49 = r8 <= r31 ? r32 : r49;
    r53 = fmaf(r13, r49, r7);
    r53 = fmaf(r46, r47, r53);
    r47 = r10 * r9;
    r47 = r47 * r48;
    r49 = fmaf(r9, r49, r47);
    r48 = r2 * r46;
    r49 = fmaf(r35, r48, r49);
    r48 = r3 * r5;
    r48 = r48 * r26;
    r48 = r48 * r9;
    r39 = r1 * r5;
    r39 = r39 * r12;
    r39 = fmaf(r41, r39, r12 * r48);
    r48 = r39 * r50;
    r48 = r8 <= r37 ? r39 : r48;
    r51 = r39 * r52;
    r48 = r15 < r38 ? r51 : r48;
    r51 = r39 * r42;
    r48 = r15 < r40 ? r51 : r48;
    r48 = r15 < r36 ? r39 : r48;
    r51 = r36 * r48;
    r51 = fmaf(r44, r51, r39 * r43);
    r51 = r36 * r51;
    r51 = r51 * r45;
    r51 = r8 <= r31 ? r32 : r51;
    r39 = fmaf(r13, r51, r7);
    r11 = r1 * r5;
    r11 = r11 * r46;
    r39 = fmaf(r12, r11, r39);
    r51 = fmaf(r9, r51, r47);
    r11 = r3 * r5;
    r11 = r11 * r46;
    r51 = fmaf(r12, r11, r51);
    WriteIdx4<1024, float, float, float4>(out_translation_jac,
                                          0 * out_translation_jac_num_alloc,
                                          global_thread_idx,
                                          r53,
                                          r49,
                                          r39,
                                          r51);
    r11 = r1 * r10;
    r34 = r34 * r34;
    r34 = 1.0 / r34;
    r11 = r11 * r34;
    r4 = r4 * r10;
    r4 = r4 * r6;
    r4 = r4 * r34;
    r11 = fmaf(r0, r4, r14 * r11);
    r6 = fmaf(r46, r11, r7);
    r54 = r26 * r9;
    r55 = r3 * r10;
    r55 = r55 * r34;
    r55 = fmaf(r2, r4, r14 * r55);
    r11 = fmaf(r11, r41, r55 * r54);
    r54 = r11 * r50;
    r54 = r8 <= r37 ? r11 : r54;
    r56 = r11 * r52;
    r54 = r15 < r38 ? r56 : r54;
    r56 = r11 * r42;
    r54 = r15 < r40 ? r56 : r54;
    r54 = r15 < r36 ? r11 : r54;
    r56 = r36 * r54;
    r56 = fmaf(r44, r56, r11 * r43);
    r56 = r36 * r56;
    r56 = r56 * r45;
    r56 = r8 <= r31 ? r32 : r56;
    r6 = fmaf(r13, r56, r6);
    r56 = fmaf(r9, r56, r47);
    r56 = fmaf(r46, r55, r56);
    WriteIdx2<1024, float, float, float2>(out_translation_jac,
                                          4 * out_translation_jac_num_alloc,
                                          global_thread_idx,
                                          r6,
                                          r56);
    r55 = r13 * r53;
    r11 = r10 * r46;
    r57 = r9 * r11;
    r55 = fmaf(r49, r57, r11 * r55);
    r58 = r13 * r39;
    r58 = fmaf(r11, r58, r51 * r57);
    r59 = r13 * r6;
    r59 = fmaf(r11, r59, r56 * r57);
    WriteSum3<float, float>((float*)inout_shared, r55, r58, r59);
  };
  FlushSumShared<3, float>(out_translation_njtr,
                           0 * out_translation_njtr_num_alloc,
                           translation_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r59 = fmaf(r53, r53, r49 * r49);
    r58 = fmaf(r39, r39, r51 * r51);
    r55 = fmaf(r56, r56, r6 * r6);
    WriteSum3<float, float>((float*)inout_shared, r59, r58, r55);
  };
  FlushSumShared<3, float>(out_translation_precond_diag,
                           0 * out_translation_precond_diag_num_alloc,
                           translation_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r55 = fmaf(r53, r39, r49 * r51);
    r49 = fmaf(r53, r6, r49 * r56);
    r56 = fmaf(r39, r6, r51 * r56);
    WriteSum3<float, float>((float*)inout_shared, r55, r49, r56);
  };
  FlushSumShared<3, float>(out_translation_precond_tril,
                           0 * out_translation_precond_tril_num_alloc,
                           translation_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r56 = r26 * r9;
    r49 = r10 * r25;
    r49 = r49 * r34;
    r55 = r5 * r33;
    r55 = fmaf(r12, r55, r14 * r49);
    r24 = fmaf(r24, r35, r25 * r4);
    r49 = fmaf(r2, r24, r3 * r55);
    r24 = fmaf(r0, r24, r1 * r55);
    r56 = fmaf(r24, r41, r49 * r56);
    r55 = r56 * r50;
    r55 = r8 <= r37 ? r56 : r55;
    r51 = r56 * r52;
    r55 = r15 < r38 ? r51 : r55;
    r51 = r56 * r42;
    r55 = r15 < r40 ? r51 : r55;
    r55 = r15 < r36 ? r56 : r55;
    r51 = r36 * r55;
    r51 = fmaf(r44, r51, r56 * r43);
    r51 = r36 * r51;
    r51 = r51 * r45;
    r51 = r8 <= r31 ? r32 : r51;
    r56 = fmaf(r13, r51, r7);
    r56 = fmaf(r46, r24, r56);
    r51 = fmaf(r9, r51, r47);
    r51 = fmaf(r46, r49, r51);
    r49 = r26 * r9;
    r24 = r10 * r16;
    r24 = r24 * r34;
    r58 = r5 * r21;
    r58 = fmaf(r12, r58, r14 * r24);
    r30 = fmaf(r30, r35, r16 * r4);
    r24 = fmaf(r2, r30, r3 * r58);
    r30 = fmaf(r0, r30, r1 * r58);
    r49 = fmaf(r30, r41, r24 * r49);
    r58 = r49 * r50;
    r58 = r8 <= r37 ? r49 : r58;
    r59 = r49 * r52;
    r58 = r15 < r38 ? r59 : r58;
    r59 = r49 * r42;
    r58 = r15 < r40 ? r59 : r58;
    r58 = r15 < r36 ? r49 : r58;
    r59 = r36 * r58;
    r49 = fmaf(r49, r43, r44 * r59);
    r49 = r36 * r49;
    r49 = r49 * r45;
    r49 = r8 <= r31 ? r32 : r49;
    r59 = fmaf(r13, r49, r7);
    r59 = fmaf(r46, r30, r59);
    r49 = fmaf(r9, r49, r47);
    r49 = fmaf(r46, r24, r49);
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r56,
                                          r51,
                                          r59,
                                          r49);
    r24 = r26 * r9;
    r30 = r5 * r29;
    r60 = r10 * r23;
    r60 = r60 * r34;
    r60 = fmaf(r14, r60, r12 * r30);
    r4 = fmaf(r23, r4, r27 * r35);
    r35 = fmaf(r2, r4, r3 * r60);
    r4 = fmaf(r0, r4, r1 * r60);
    r41 = fmaf(r4, r41, r35 * r24);
    r50 = r41 * r50;
    r50 = r8 <= r37 ? r41 : r50;
    r52 = r41 * r52;
    r50 = r15 < r38 ? r52 : r50;
    r42 = r41 * r42;
    r50 = r15 < r40 ? r42 : r50;
    r50 = r15 < r36 ? r41 : r50;
    r15 = r36 * r50;
    r15 = fmaf(r44, r15, r41 * r43);
    r15 = r36 * r15;
    r15 = r15 * r45;
    r15 = r8 <= r31 ? r32 : r15;
    r7 = fmaf(r13, r15, r7);
    r7 = fmaf(r46, r4, r7);
    r15 = fmaf(r9, r15, r47);
    r15 = fmaf(r46, r35, r15);
    WriteIdx2<1024, float, float, float2>(
        out_point_jac, 4 * out_point_jac_num_alloc, global_thread_idx, r7, r15);
    r35 = r13 * r56;
    r35 = fmaf(r51, r57, r11 * r35);
    r47 = r13 * r59;
    r47 = fmaf(r49, r57, r11 * r47);
    r4 = r13 * r7;
    r57 = fmaf(r15, r57, r11 * r4);
    WriteSum3<float, float>((float*)inout_shared, r35, r47, r57);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r57 = fmaf(r51, r51, r56 * r56);
    r47 = fmaf(r49, r49, r59 * r59);
    r35 = fmaf(r15, r15, r7 * r7);
    WriteSum3<float, float>((float*)inout_shared, r57, r47, r35);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = fmaf(r51, r49, r56 * r59);
    r51 = fmaf(r51, r15, r56 * r7);
    r15 = fmaf(r49, r15, r59 * r7);
    WriteSum3<float, float>((float*)inout_shared, r35, r51, r15);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void PinholeFixedRotationFixedCalibResJacFirst(
    float* rotation,
    unsigned int rotation_num_alloc,
    float* translation,
    unsigned int translation_num_alloc,
    SharedIndex* translation_indices,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* weight_loss,
    unsigned int weight_loss_num_alloc,
    float* calib,
    unsigned int calib_num_alloc,
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
  PinholeFixedRotationFixedCalibResJacFirstKernel<<<n_blocks, 1024>>>(
      rotation,
      rotation_num_alloc,
      translation,
      translation_num_alloc,
      translation_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      weight_loss,
      weight_loss_num_alloc,
      calib,
      calib_num_alloc,
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