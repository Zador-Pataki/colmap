#include "kernel_pinhole_split_fixed_rotation_fixed_principal_point_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedRotationFixedPrincipalPointFixedPointResJacKernel(
        float* rotation,
        unsigned int rotation_num_alloc,
        float* translation,
        unsigned int translation_num_alloc,
        SharedIndex* translation_indices,
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

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52;

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
    r4 = 9.99999999999999955e-07;
  };
  LoadShared<3, float, float>(translation,
                              0 * translation_num_alloc,
                              translation_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       translation_indices_loc[threadIdx.x].target,
                       r9,
                       r10,
                       r11);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r12, r13, r14);
    r15 = -2.00000000000000000e+00;
    ReadIdx4<1024, float, float, float4>(rotation,
                                         0 * rotation_num_alloc,
                                         global_thread_idx,
                                         r16,
                                         r17,
                                         r18,
                                         r19);
    r20 = r16 * r16;
    r20 = r15 * r20;
    r21 = 1.00000000000000000e+00;
    r22 = r17 * r17;
    r22 = fmaf(r15, r22, r21);
    r23 = r20 + r22;
    r23 = fmaf(r14, r23, r11);
    r11 = r16 * r18;
    r24 = 2.00000000000000000e+00;
    r11 = r11 * r24;
    r25 = r19 * r15;
    r26 = fmaf(r17, r25, r11);
    r27 = r16 * r19;
    r28 = r17 * r18;
    r28 = r28 * r24;
    r27 = fmaf(r24, r27, r28);
    r23 = fmaf(r12, r26, r23);
    r23 = fmaf(r13, r27, r23);
    r27 = copysign(1.0, r23);
    r27 = fmaf(r4, r27, r23);
    r23 = 1.0 / r27;
  };
  LoadShared<2, float, float>(
      focal, 0 * focal_num_alloc, focal_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>(
        (float*)inout_shared, focal_indices_loc[threadIdx.x].target, r26, r29);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r30 = r18 * r18;
    r30 = r15 * r30;
    r22 = r30 + r22;
    r22 = fmaf(r12, r22, r9);
    r9 = r17 * r19;
    r9 = fmaf(r24, r9, r11);
    r11 = r16 * r17;
    r11 = r11 * r24;
    r15 = fmaf(r18, r25, r11);
    r22 = fmaf(r14, r9, r22);
    r22 = fmaf(r13, r15, r22);
    r15 = r26 * r22;
    r6 = fmaf(r23, r15, r6);
    r7 = fmaf(r7, r8, r5);
    r30 = r21 + r30;
    r30 = r30 + r20;
    r30 = fmaf(r13, r30, r10);
    r25 = fmaf(r16, r25, r28);
    r28 = r18 * r19;
    r28 = fmaf(r24, r28, r11);
    r30 = fmaf(r14, r25, r30);
    r30 = fmaf(r12, r28, r30);
    r28 = r29 * r30;
    r7 = fmaf(r23, r28, r7);
    r12 = fmaf(r1, r7, r0 * r6);
    ReadIdx3<1024, float, float, float4>(weight_loss,
                                         4 * weight_loss_num_alloc,
                                         global_thread_idx,
                                         r25,
                                         r14,
                                         r11);
    r13 = 0.00000000000000000e+00;
    r11 = fmaxf(r11, r13);
    r10 = sqrtf(r11);
    r7 = fmaf(r3, r7, r2 * r6);
    r6 = fmaf(r7, r7, r12 * r12);
    r20 = 5.00000000000000000e-01;
    r14 = fmaxf(r14, r4);
    r5 = r14 * r14;
    r9 = r8 * r14;
    r31 = r24 * r14;
    r32 = fmaxf(r4, r6);
    r33 = sqrtf(r32);
    r31 = fmaf(r33, r31, r14 * r9);
    r31 = r6 <= r5 ? r6 : r31;
    r9 = 2.50000000000000000e+00;
    r33 = r14 * r14;
    r34 = 1.0 / r5;
    r34 = fmaf(r6, r34, r21);
    r35 = logf(r34);
    r33 = r33 * r35;
    r31 = r25 < r9 ? r33 : r31;
    r33 = 1.50000000000000000e+00;
    r35 = r24 * r14;
    r36 = sqrtf(r34);
    r36 = r8 + r36;
    r35 = r35 * r14;
    r35 = r35 * r36;
    r31 = r25 < r33 ? r35 : r31;
    r31 = r25 < r20 ? r6 : r31;
    r35 = fmaxf(r13, r31);
    r36 = 1.0 / r32;
    r36 = r11 * r36;
    r37 = r35 * r36;
    r38 = sqrtf(r37);
    r38 = r6 <= r4 ? r10 : r38;
    r10 = r12 * r38;
    r39 = r7 * r38;
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r10, r39);
    r39 = r26 * r2;
    r10 = r24 * r23;
    r39 = r39 * r7;
    r40 = r26 * r0;
    r41 = r12 * r10;
    r40 = fmaf(r41, r40, r10 * r39);
    r39 = r20 * r14;
    r42 = -9.99999999999999955e-07;
    r42 = r42 + r6;
    r42 = copysign(1.0, r42);
    r42 = r21 + r42;
    r43 = rsqrtf(r32);
    r39 = r39 * r42;
    r39 = r39 * r43;
    r43 = r40 * r39;
    r43 = r6 <= r5 ? r40 : r43;
    r44 = 1.0 / r34;
    r45 = r40 * r44;
    r43 = r25 < r9 ? r45 : r43;
    r34 = rsqrtf(r34);
    r45 = r40 * r34;
    r43 = r25 < r33 ? r45 : r43;
    r43 = r25 < r20 ? r40 : r43;
    r45 = r20 * r43;
    r31 = copysign(1.0, r31);
    r31 = r21 + r31;
    r36 = r31 * r36;
    r35 = r11 * r35;
    r11 = -5.00000000000000000e-01;
    r32 = r32 * r32;
    r32 = 1.0 / r32;
    r35 = r35 * r11;
    r35 = r35 * r42;
    r35 = r35 * r32;
    r40 = fmaf(r40, r35, r36 * r45);
    r40 = r20 * r40;
    r37 = rsqrtf(r37);
    r40 = r40 * r37;
    r40 = r6 <= r4 ? r13 : r40;
    r45 = r8 * r12;
    r32 = 2.50000000000000000e-01;
    r42 = r6 <= r5 ? r13 : r13;
    r42 = r25 < r9 ? r13 : r42;
    r42 = r25 < r33 ? r13 : r42;
    r42 = r25 < r20 ? r13 : r42;
    r42 = r32 * r42;
    r42 = r42 * r37;
    r42 = r42 * r36;
    r42 = r6 <= r4 ? r13 : r42;
    r45 = r45 * r42;
    r32 = fmaf(r12, r40, r45);
    r11 = r26 * r0;
    r11 = r11 * r38;
    r32 = fmaf(r23, r11, r32);
    r11 = r8 * r7;
    r11 = r11 * r42;
    r40 = fmaf(r7, r40, r11);
    r42 = r26 * r2;
    r42 = r42 * r38;
    r40 = fmaf(r23, r42, r40);
    r42 = r29 * r3;
    r42 = r42 * r7;
    r31 = r1 * r41;
    r42 = fmaf(r29, r31, r10 * r42);
    r21 = r42 * r39;
    r21 = r6 <= r5 ? r42 : r21;
    r46 = r42 * r44;
    r21 = r25 < r9 ? r46 : r21;
    r46 = r42 * r34;
    r21 = r25 < r33 ? r46 : r21;
    r21 = r25 < r20 ? r42 : r21;
    r46 = r20 * r21;
    r42 = fmaf(r42, r35, r36 * r46);
    r42 = r20 * r42;
    r42 = r42 * r37;
    r42 = r6 <= r4 ? r13 : r42;
    r46 = fmaf(r12, r42, r45);
    r47 = r29 * r1;
    r47 = r47 * r38;
    r46 = fmaf(r23, r47, r46);
    r42 = fmaf(r7, r42, r11);
    r47 = r29 * r3;
    r47 = r47 * r38;
    r42 = fmaf(r23, r47, r42);
    WriteIdx4<1024, float, float, float4>(out_translation_jac,
                                          0 * out_translation_jac_num_alloc,
                                          global_thread_idx,
                                          r32,
                                          r40,
                                          r46,
                                          r42);
    r27 = r27 * r27;
    r27 = 1.0 / r27;
    r27 = r8 * r27;
    r15 = r15 * r27;
    r47 = r1 * r28;
    r47 = fmaf(r27, r47, r0 * r15);
    r48 = fmaf(r38, r47, r45);
    r49 = r24 * r12;
    r50 = r24 * r7;
    r51 = r3 * r28;
    r51 = fmaf(r27, r51, r2 * r15);
    r50 = fmaf(r51, r50, r47 * r49);
    r49 = r50 * r39;
    r49 = r6 <= r5 ? r50 : r49;
    r47 = r50 * r44;
    r49 = r25 < r9 ? r47 : r49;
    r47 = r50 * r34;
    r49 = r25 < r33 ? r47 : r49;
    r49 = r25 < r20 ? r50 : r49;
    r47 = r20 * r49;
    r47 = fmaf(r36, r47, r50 * r35);
    r47 = r20 * r47;
    r47 = r47 * r37;
    r47 = r6 <= r4 ? r13 : r47;
    r48 = fmaf(r12, r47, r48);
    r47 = fmaf(r7, r47, r11);
    r47 = fmaf(r38, r51, r47);
    WriteIdx2<1024, float, float, float2>(out_translation_jac,
                                          4 * out_translation_jac_num_alloc,
                                          global_thread_idx,
                                          r48,
                                          r47);
    r51 = r12 * r32;
    r50 = r8 * r38;
    r15 = r7 * r50;
    r51 = fmaf(r40, r15, r50 * r51);
    r27 = r12 * r46;
    r27 = fmaf(r50, r27, r42 * r15);
    r52 = r12 * r48;
    r52 = fmaf(r50, r52, r47 * r15);
    WriteSum3<float, float>((float*)inout_shared, r51, r27, r52);
  };
  FlushSumShared<3, float>(out_translation_njtr,
                           0 * out_translation_njtr_num_alloc,
                           translation_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r52 = fmaf(r32, r32, r40 * r40);
    r27 = fmaf(r46, r46, r42 * r42);
    r51 = fmaf(r47, r47, r48 * r48);
    WriteSum3<float, float>((float*)inout_shared, r52, r27, r51);
  };
  FlushSumShared<3, float>(out_translation_precond_diag,
                           0 * out_translation_precond_diag_num_alloc,
                           translation_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r51 = fmaf(r32, r46, r40 * r42);
    r40 = fmaf(r40, r47, r32 * r48);
    r47 = fmaf(r46, r48, r42 * r47);
    WriteSum3<float, float>((float*)inout_shared, r51, r40, r47);
  };
  FlushSumShared<3, float>(out_translation_precond_tril,
                           0 * out_translation_precond_tril_num_alloc,
                           translation_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = r0 * r22;
    r40 = r2 * r22;
    r40 = r40 * r7;
    r40 = fmaf(r10, r40, r41 * r47);
    r47 = r40 * r39;
    r47 = r6 <= r5 ? r40 : r47;
    r41 = r40 * r44;
    r47 = r25 < r9 ? r41 : r47;
    r41 = r40 * r34;
    r47 = r25 < r33 ? r41 : r47;
    r47 = r25 < r20 ? r40 : r47;
    r41 = r20 * r47;
    r41 = fmaf(r36, r41, r40 * r35);
    r41 = r20 * r41;
    r41 = r41 * r37;
    r41 = r6 <= r4 ? r13 : r41;
    r40 = fmaf(r12, r41, r45);
    r51 = r0 * r22;
    r51 = r51 * r38;
    r40 = fmaf(r23, r51, r40);
    r41 = fmaf(r7, r41, r11);
    r51 = r2 * r22;
    r51 = r51 * r38;
    r41 = fmaf(r23, r51, r41);
    r51 = r3 * r30;
    r51 = r51 * r7;
    r51 = fmaf(r10, r51, r30 * r31);
    r39 = r51 * r39;
    r39 = r6 <= r5 ? r51 : r39;
    r44 = r51 * r44;
    r39 = r25 < r9 ? r44 : r39;
    r34 = r51 * r34;
    r39 = r25 < r33 ? r34 : r39;
    r39 = r25 < r20 ? r51 : r39;
    r25 = r20 * r39;
    r25 = fmaf(r36, r25, r51 * r35);
    r25 = r20 * r25;
    r25 = r25 * r37;
    r25 = r6 <= r4 ? r13 : r25;
    r45 = fmaf(r12, r25, r45);
    r13 = r1 * r30;
    r13 = r13 * r38;
    r45 = fmaf(r23, r13, r45);
    r25 = fmaf(r7, r25, r11);
    r11 = r3 * r30;
    r11 = r11 * r38;
    r25 = fmaf(r23, r11, r25);
    WriteIdx4<1024, float, float, float4>(out_focal_jac,
                                          0 * out_focal_jac_num_alloc,
                                          global_thread_idx,
                                          r40,
                                          r41,
                                          r45,
                                          r25);
    r11 = r12 * r40;
    r11 = fmaf(r50, r11, r41 * r15);
    r23 = r12 * r45;
    r15 = fmaf(r25, r15, r50 * r23);
    WriteSum2<float, float>((float*)inout_shared, r11, r15);
  };
  FlushSumShared<2, float>(out_focal_njtr,
                           0 * out_focal_njtr_num_alloc,
                           focal_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r15 = fmaf(r41, r41, r40 * r40);
    r11 = fmaf(r25, r25, r45 * r45);
    WriteSum2<float, float>((float*)inout_shared, r15, r11);
  };
  FlushSumShared<2, float>(out_focal_precond_diag,
                           0 * out_focal_precond_diag_num_alloc,
                           focal_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = fmaf(r41, r25, r40 * r45);
    WriteSum1<float, float>((float*)inout_shared, r25);
  };
  FlushSumShared<1, float>(out_focal_precond_tril,
                           0 * out_focal_precond_tril_num_alloc,
                           focal_indices_loc,
                           (float*)inout_shared);
}

void PinholeSplitFixedRotationFixedPrincipalPointFixedPointResJac(
    float* rotation,
    unsigned int rotation_num_alloc,
    float* translation,
    unsigned int translation_num_alloc,
    SharedIndex* translation_indices,
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
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitFixedRotationFixedPrincipalPointFixedPointResJacKernel<<<n_blocks,
                                                                       1024>>>(
      rotation,
      rotation_num_alloc,
      translation,
      translation_num_alloc,
      translation_indices,
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
      problem_size);
}

}  // namespace caspar