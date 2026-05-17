#include "kernel_pinhole_split_fixed_rotation_fixed_focal_fixed_principal_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedRotationFixedFocalFixedPrincipalPointResJacKernel(
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
        float* focal,
        unsigned int focal_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
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
    ReadIdx2<1024, float, float, float2>(
        focal, 0 * focal_num_alloc, global_thread_idx, r29, r30);
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
    r12 = r30 * r12;
    r7 = fmaf(r10, r12, r7);
    r11 = fmaf(r1, r7, r0 * r6);
    ReadIdx3<1024, float, float, float4>(weight_loss,
                                         4 * weight_loss_num_alloc,
                                         global_thread_idx,
                                         r13,
                                         r26,
                                         r9);
    r32 = 0.00000000000000000e+00;
    r9 = fmaxf(r9, r32);
    r5 = sqrtf(r9);
    r7 = fmaf(r3, r7, r2 * r6);
    r6 = fmaf(r7, r7, r11 * r11);
    r36 = 5.00000000000000000e-01;
    r26 = fmaxf(r26, r31);
    r37 = r26 * r26;
    r38 = r8 * r26;
    r39 = r24 * r26;
    r40 = fmaxf(r31, r6);
    r41 = sqrtf(r40);
    r39 = fmaf(r41, r39, r26 * r38);
    r39 = r6 <= r37 ? r6 : r39;
    r38 = 2.50000000000000000e+00;
    r41 = r26 * r26;
    r42 = 1.0 / r37;
    r42 = fmaf(r6, r42, r20);
    r43 = logf(r42);
    r41 = r41 * r43;
    r39 = r13 < r38 ? r41 : r39;
    r41 = 1.50000000000000000e+00;
    r43 = r24 * r26;
    r44 = sqrtf(r42);
    r44 = r8 + r44;
    r43 = r43 * r26;
    r43 = r43 * r44;
    r39 = r13 < r41 ? r43 : r39;
    r39 = r13 < r36 ? r6 : r39;
    r43 = fmaxf(r32, r39);
    r44 = 1.0 / r40;
    r44 = r9 * r44;
    r45 = r43 * r44;
    r46 = sqrtf(r45);
    r46 = r6 <= r31 ? r5 : r46;
    r5 = r11 * r46;
    r47 = r7 * r46;
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r5, r47);
    r47 = r24 * r7;
    r5 = r2 * r35;
    r48 = r0 * r24;
    r48 = r48 * r11;
    r48 = fmaf(r35, r48, r47 * r5);
    r49 = r36 * r26;
    r50 = -9.99999999999999955e-07;
    r50 = r50 + r6;
    r50 = copysign(1.0, r50);
    r50 = r20 + r50;
    r51 = rsqrtf(r40);
    r49 = r49 * r50;
    r49 = r49 * r51;
    r51 = r48 * r49;
    r51 = r6 <= r37 ? r48 : r51;
    r52 = 1.0 / r42;
    r53 = r48 * r52;
    r51 = r13 < r38 ? r53 : r51;
    r42 = rsqrtf(r42);
    r53 = r48 * r42;
    r51 = r13 < r41 ? r53 : r51;
    r51 = r13 < r36 ? r48 : r51;
    r53 = r36 * r51;
    r39 = copysign(1.0, r39);
    r39 = r20 + r39;
    r44 = r39 * r44;
    r43 = r9 * r43;
    r9 = -5.00000000000000000e-01;
    r40 = r40 * r40;
    r40 = 1.0 / r40;
    r43 = r43 * r9;
    r43 = r43 * r50;
    r43 = r43 * r40;
    r48 = fmaf(r48, r43, r44 * r53);
    r48 = r36 * r48;
    r45 = rsqrtf(r45);
    r48 = r48 * r45;
    r48 = r6 <= r31 ? r32 : r48;
    r53 = r8 * r11;
    r40 = 2.50000000000000000e-01;
    r50 = r6 <= r37 ? r32 : r32;
    r50 = r13 < r38 ? r32 : r50;
    r50 = r13 < r41 ? r32 : r50;
    r50 = r13 < r36 ? r32 : r50;
    r50 = r40 * r50;
    r50 = r50 * r45;
    r50 = r50 * r44;
    r50 = r6 <= r31 ? r32 : r50;
    r53 = r53 * r50;
    r40 = fmaf(r11, r48, r53);
    r9 = r0 * r46;
    r40 = fmaf(r35, r9, r40);
    r9 = r8 * r7;
    r9 = r9 * r50;
    r48 = fmaf(r7, r48, r9);
    r48 = fmaf(r46, r5, r48);
    r5 = r3 * r30;
    r5 = r5 * r10;
    r50 = r1 * r30;
    r50 = r50 * r24;
    r50 = r50 * r11;
    r50 = fmaf(r10, r50, r47 * r5);
    r5 = r50 * r49;
    r5 = r6 <= r37 ? r50 : r5;
    r39 = r50 * r52;
    r5 = r13 < r38 ? r39 : r5;
    r39 = r50 * r42;
    r5 = r13 < r41 ? r39 : r5;
    r5 = r13 < r36 ? r50 : r5;
    r39 = r36 * r5;
    r50 = fmaf(r50, r43, r44 * r39);
    r50 = r36 * r50;
    r50 = r50 * r45;
    r50 = r6 <= r31 ? r32 : r50;
    r39 = fmaf(r11, r50, r53);
    r20 = r1 * r30;
    r20 = r20 * r46;
    r39 = fmaf(r10, r20, r39);
    r50 = fmaf(r7, r50, r9);
    r20 = r3 * r30;
    r20 = r20 * r46;
    r50 = fmaf(r10, r20, r50);
    WriteIdx4<1024, float, float, float4>(out_translation_jac,
                                          0 * out_translation_jac_num_alloc,
                                          global_thread_idx,
                                          r40,
                                          r48,
                                          r39,
                                          r50);
    r29 = r29 * r8;
    r34 = r34 * r34;
    r34 = 1.0 / r34;
    r29 = r29 * r4;
    r29 = r29 * r34;
    r4 = r1 * r8;
    r4 = r4 * r34;
    r4 = fmaf(r12, r4, r0 * r29);
    r20 = fmaf(r46, r4, r53);
    r54 = r24 * r11;
    r55 = r3 * r8;
    r55 = r55 * r34;
    r55 = fmaf(r12, r55, r2 * r29);
    r54 = fmaf(r55, r47, r4 * r54);
    r4 = r54 * r49;
    r4 = r6 <= r37 ? r54 : r4;
    r56 = r54 * r52;
    r4 = r13 < r38 ? r56 : r4;
    r56 = r54 * r42;
    r4 = r13 < r41 ? r56 : r4;
    r4 = r13 < r36 ? r54 : r4;
    r56 = r36 * r4;
    r56 = fmaf(r44, r56, r54 * r43);
    r56 = r36 * r56;
    r56 = r56 * r45;
    r56 = r6 <= r31 ? r32 : r56;
    r20 = fmaf(r11, r56, r20);
    r56 = fmaf(r7, r56, r9);
    r56 = fmaf(r46, r55, r56);
    WriteIdx2<1024, float, float, float2>(out_translation_jac,
                                          4 * out_translation_jac_num_alloc,
                                          global_thread_idx,
                                          r20,
                                          r56);
    r55 = r8 * r46;
    r54 = r11 * r55;
    r57 = r7 * r48;
    r57 = fmaf(r55, r57, r40 * r54);
    r58 = r7 * r50;
    r58 = fmaf(r39, r54, r55 * r58);
    r59 = r7 * r56;
    r59 = fmaf(r20, r54, r55 * r59);
    WriteSum3<float, float>((float*)inout_shared, r57, r58, r59);
  };
  FlushSumShared<3, float>(out_translation_njtr,
                           0 * out_translation_njtr_num_alloc,
                           translation_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r59 = fmaf(r40, r40, r48 * r48);
    r58 = fmaf(r39, r39, r50 * r50);
    r57 = fmaf(r56, r56, r20 * r20);
    WriteSum3<float, float>((float*)inout_shared, r59, r58, r57);
  };
  FlushSumShared<3, float>(out_translation_precond_diag,
                           0 * out_translation_precond_diag_num_alloc,
                           translation_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r57 = fmaf(r40, r39, r48 * r50);
    r40 = fmaf(r48, r56, r40 * r20);
    r20 = fmaf(r39, r20, r50 * r56);
    WriteSum3<float, float>((float*)inout_shared, r57, r40, r20);
  };
  FlushSumShared<3, float>(out_translation_precond_tril,
                           0 * out_translation_precond_tril_num_alloc,
                           translation_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r20 = r24 * r11;
    r40 = r30 * r33;
    r57 = r23 * r8;
    r57 = r57 * r34;
    r57 = fmaf(r12, r57, r10 * r40);
    r22 = fmaf(r23, r29, r22 * r35);
    r40 = fmaf(r0, r22, r1 * r57);
    r22 = fmaf(r2, r22, r3 * r57);
    r20 = fmaf(r22, r47, r40 * r20);
    r57 = r20 * r49;
    r57 = r6 <= r37 ? r20 : r57;
    r39 = r20 * r52;
    r57 = r13 < r38 ? r39 : r57;
    r39 = r20 * r42;
    r57 = r13 < r41 ? r39 : r57;
    r57 = r13 < r36 ? r20 : r57;
    r39 = r36 * r57;
    r20 = fmaf(r20, r43, r44 * r39);
    r20 = r36 * r20;
    r20 = r20 * r45;
    r20 = r6 <= r31 ? r32 : r20;
    r39 = fmaf(r11, r20, r53);
    r39 = fmaf(r46, r40, r39);
    r20 = fmaf(r7, r20, r9);
    r20 = fmaf(r46, r22, r20);
    r22 = r30 * r19;
    r40 = r14 * r8;
    r40 = r40 * r34;
    r40 = fmaf(r12, r40, r10 * r22);
    r28 = fmaf(r14, r29, r28 * r35);
    r22 = fmaf(r0, r28, r1 * r40);
    r58 = fmaf(r46, r22, r53);
    r59 = r24 * r11;
    r28 = fmaf(r2, r28, r3 * r40);
    r59 = fmaf(r28, r47, r22 * r59);
    r22 = r59 * r49;
    r22 = r6 <= r37 ? r59 : r22;
    r40 = r59 * r52;
    r22 = r13 < r38 ? r40 : r22;
    r40 = r59 * r42;
    r22 = r13 < r41 ? r40 : r22;
    r22 = r13 < r36 ? r59 : r22;
    r40 = r36 * r22;
    r59 = fmaf(r59, r43, r44 * r40);
    r59 = r36 * r59;
    r59 = r59 * r45;
    r59 = r6 <= r31 ? r32 : r59;
    r58 = fmaf(r11, r59, r58);
    r28 = fmaf(r46, r28, r9);
    r28 = fmaf(r7, r59, r28);
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r39,
                                          r20,
                                          r58,
                                          r28);
    r59 = r24 * r11;
    r40 = r30 * r27;
    r60 = r21 * r8;
    r60 = r60 * r34;
    r60 = fmaf(r12, r60, r10 * r40);
    r29 = fmaf(r21, r29, r25 * r35);
    r35 = fmaf(r0, r29, r1 * r60);
    r29 = fmaf(r2, r29, r3 * r60);
    r47 = fmaf(r29, r47, r35 * r59);
    r49 = r47 * r49;
    r49 = r6 <= r37 ? r47 : r49;
    r52 = r47 * r52;
    r49 = r13 < r38 ? r52 : r49;
    r42 = r47 * r42;
    r49 = r13 < r41 ? r42 : r49;
    r49 = r13 < r36 ? r47 : r49;
    r13 = r36 * r49;
    r13 = fmaf(r44, r13, r47 * r43);
    r13 = r36 * r13;
    r13 = r13 * r45;
    r13 = r6 <= r31 ? r32 : r13;
    r53 = fmaf(r11, r13, r53);
    r53 = fmaf(r46, r35, r53);
    r13 = fmaf(r7, r13, r9);
    r13 = fmaf(r46, r29, r13);
    WriteIdx2<1024, float, float, float2>(out_point_jac,
                                          4 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r53,
                                          r13);
    r29 = r7 * r20;
    r29 = fmaf(r55, r29, r39 * r54);
    r9 = r7 * r28;
    r9 = fmaf(r55, r9, r58 * r54);
    r35 = r7 * r13;
    r35 = fmaf(r55, r35, r53 * r54);
    WriteSum3<float, float>((float*)inout_shared, r29, r9, r35);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = fmaf(r39, r39, r20 * r20);
    r9 = fmaf(r28, r28, r58 * r58);
    r29 = fmaf(r53, r53, r13 * r13);
    WriteSum3<float, float>((float*)inout_shared, r35, r9, r29);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r29 = fmaf(r20, r28, r39 * r58);
    r39 = fmaf(r39, r53, r20 * r13);
    r53 = fmaf(r28, r13, r58 * r53);
    WriteSum3<float, float>((float*)inout_shared, r29, r39, r53);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
}

void PinholeSplitFixedRotationFixedFocalFixedPrincipalPointResJac(
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
    float* focal,
    unsigned int focal_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
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
  PinholeSplitFixedRotationFixedFocalFixedPrincipalPointResJacKernel<<<n_blocks,
                                                                       1024>>>(
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
      focal,
      focal_num_alloc,
      principal_point,
      principal_point_num_alloc,
      out_res,
      out_res_num_alloc,
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