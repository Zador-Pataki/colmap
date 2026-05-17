#include "kernel_pinhole_fixed_rotation_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeFixedRotationFixedPointResJacKernel(
        float* rotation,
        unsigned int rotation_num_alloc,
        float* translation,
        unsigned int translation_num_alloc,
        SharedIndex* translation_indices,
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

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53;

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
    r6 = 9.99999999999999955e-07;
  };
  LoadShared<3, float, float>(translation,
                              0 * translation_num_alloc,
                              translation_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       translation_indices_loc[threadIdx.x].target,
                       r11,
                       r12,
                       r13);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r14, r15, r16);
    r17 = -2.00000000000000000e+00;
    ReadIdx4<1024, float, float, float4>(rotation,
                                         0 * rotation_num_alloc,
                                         global_thread_idx,
                                         r18,
                                         r19,
                                         r20,
                                         r21);
    r22 = r18 * r18;
    r22 = r17 * r22;
    r23 = 1.00000000000000000e+00;
    r24 = r19 * r19;
    r24 = fmaf(r17, r24, r23);
    r25 = r22 + r24;
    r25 = fmaf(r16, r25, r13);
    r13 = r18 * r20;
    r26 = 2.00000000000000000e+00;
    r13 = r13 * r26;
    r27 = r21 * r17;
    r28 = fmaf(r19, r27, r13);
    r29 = r18 * r21;
    r30 = r19 * r20;
    r30 = r30 * r26;
    r29 = fmaf(r26, r29, r30);
    r25 = fmaf(r14, r28, r25);
    r25 = fmaf(r15, r29, r25);
    r29 = copysign(1.0, r25);
    r29 = fmaf(r6, r29, r25);
    r25 = 1.0 / r29;
    r28 = r20 * r20;
    r28 = r17 * r28;
    r24 = r28 + r24;
    r24 = fmaf(r14, r24, r11);
    r11 = r19 * r21;
    r11 = fmaf(r26, r11, r13);
    r13 = r18 * r19;
    r13 = r13 * r26;
    r17 = fmaf(r20, r27, r13);
    r24 = fmaf(r16, r11, r24);
    r24 = fmaf(r15, r17, r24);
    r17 = r4 * r24;
    r8 = fmaf(r25, r17, r8);
    r9 = fmaf(r9, r10, r7);
    r28 = r23 + r28;
    r28 = r28 + r22;
    r28 = fmaf(r15, r28, r12);
    r27 = fmaf(r18, r27, r30);
    r30 = r20 * r21;
    r30 = fmaf(r26, r30, r13);
    r28 = fmaf(r16, r27, r28);
    r28 = fmaf(r14, r30, r28);
    r30 = r5 * r28;
    r9 = fmaf(r25, r30, r9);
    r14 = fmaf(r1, r9, r0 * r8);
    ReadIdx3<1024, float, float, float4>(weight_loss,
                                         4 * weight_loss_num_alloc,
                                         global_thread_idx,
                                         r27,
                                         r16,
                                         r13);
    r15 = 0.00000000000000000e+00;
    r13 = fmaxf(r13, r15);
    r12 = sqrtf(r13);
    r9 = fmaf(r3, r9, r2 * r8);
    r8 = fmaf(r14, r14, r9 * r9);
    r22 = 5.00000000000000000e-01;
    r16 = fmaxf(r16, r6);
    r7 = r16 * r16;
    r11 = r26 * r16;
    r31 = fmaxf(r6, r8);
    r32 = sqrtf(r31);
    r33 = r10 * r16;
    r33 = fmaf(r16, r33, r32 * r11);
    r33 = r8 <= r7 ? r8 : r33;
    r11 = 2.50000000000000000e+00;
    r32 = r16 * r16;
    r34 = 1.0 / r7;
    r34 = fmaf(r8, r34, r23);
    r35 = logf(r34);
    r32 = r32 * r35;
    r33 = r27 < r11 ? r32 : r33;
    r32 = 1.50000000000000000e+00;
    r35 = r26 * r16;
    r36 = sqrtf(r34);
    r36 = r10 + r36;
    r35 = r35 * r16;
    r35 = r35 * r36;
    r33 = r27 < r32 ? r35 : r33;
    r33 = r27 < r22 ? r8 : r33;
    r35 = fmaxf(r15, r33);
    r36 = 1.0 / r31;
    r36 = r13 * r36;
    r37 = r35 * r36;
    r38 = sqrtf(r37);
    r38 = r8 <= r6 ? r12 : r38;
    r12 = r14 * r38;
    r39 = r9 * r38;
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r12, r39);
    r39 = r10 * r14;
    r12 = 2.50000000000000000e-01;
    r40 = r8 <= r7 ? r15 : r15;
    r40 = r27 < r11 ? r15 : r40;
    r40 = r27 < r32 ? r15 : r40;
    r40 = r27 < r22 ? r15 : r40;
    r40 = r12 * r40;
    r37 = rsqrtf(r37);
    r33 = copysign(1.0, r33);
    r33 = r23 + r33;
    r36 = r33 * r36;
    r40 = r40 * r37;
    r40 = r40 * r36;
    r40 = r8 <= r6 ? r15 : r40;
    r39 = r39 * r40;
    r33 = r4 * r0;
    r12 = r26 * r25;
    r33 = r33 * r14;
    r41 = r4 * r2;
    r42 = r9 * r12;
    r41 = fmaf(r42, r41, r12 * r33);
    r33 = r22 * r16;
    r43 = -9.99999999999999955e-07;
    r43 = r43 + r8;
    r43 = copysign(1.0, r43);
    r43 = r23 + r43;
    r23 = rsqrtf(r31);
    r33 = r33 * r43;
    r33 = r33 * r23;
    r23 = r41 * r33;
    r23 = r8 <= r7 ? r41 : r23;
    r44 = 1.0 / r34;
    r45 = r41 * r44;
    r23 = r27 < r11 ? r45 : r23;
    r34 = rsqrtf(r34);
    r45 = r41 * r34;
    r23 = r27 < r32 ? r45 : r23;
    r23 = r27 < r22 ? r41 : r23;
    r45 = r22 * r23;
    r35 = r13 * r35;
    r13 = -5.00000000000000000e-01;
    r31 = r31 * r31;
    r31 = 1.0 / r31;
    r35 = r35 * r13;
    r35 = r35 * r43;
    r35 = r35 * r31;
    r41 = fmaf(r41, r35, r36 * r45);
    r41 = r22 * r41;
    r41 = r41 * r37;
    r41 = r8 <= r6 ? r15 : r41;
    r45 = fmaf(r14, r41, r39);
    r31 = r4 * r0;
    r31 = r31 * r38;
    r45 = fmaf(r25, r31, r45);
    r31 = r10 * r9;
    r31 = r31 * r40;
    r41 = fmaf(r9, r41, r31);
    r40 = r4 * r2;
    r40 = r40 * r38;
    r41 = fmaf(r25, r40, r41);
    r40 = r3 * r42;
    r43 = r5 * r1;
    r43 = r43 * r14;
    r43 = fmaf(r12, r43, r5 * r40);
    r13 = r43 * r33;
    r13 = r8 <= r7 ? r43 : r13;
    r46 = r43 * r44;
    r13 = r27 < r11 ? r46 : r13;
    r46 = r43 * r34;
    r13 = r27 < r32 ? r46 : r13;
    r13 = r27 < r22 ? r43 : r13;
    r46 = r22 * r13;
    r46 = fmaf(r36, r46, r43 * r35);
    r46 = r22 * r46;
    r46 = r46 * r37;
    r46 = r8 <= r6 ? r15 : r46;
    r43 = fmaf(r14, r46, r39);
    r47 = r5 * r1;
    r47 = r47 * r38;
    r43 = fmaf(r25, r47, r43);
    r46 = fmaf(r9, r46, r31);
    r47 = r5 * r3;
    r47 = r47 * r38;
    r46 = fmaf(r25, r47, r46);
    WriteIdx4<1024, float, float, float4>(out_translation_jac,
                                          0 * out_translation_jac_num_alloc,
                                          global_thread_idx,
                                          r45,
                                          r41,
                                          r43,
                                          r46);
    r29 = r29 * r29;
    r29 = 1.0 / r29;
    r29 = r10 * r29;
    r30 = r30 * r29;
    r47 = r0 * r17;
    r47 = fmaf(r29, r47, r1 * r30);
    r48 = fmaf(r38, r47, r39);
    r49 = r26 * r14;
    r50 = r26 * r9;
    r51 = r2 * r17;
    r51 = fmaf(r29, r51, r3 * r30);
    r50 = fmaf(r51, r50, r47 * r49);
    r49 = r50 * r33;
    r49 = r8 <= r7 ? r50 : r49;
    r47 = r50 * r44;
    r49 = r27 < r11 ? r47 : r49;
    r47 = r50 * r34;
    r49 = r27 < r32 ? r47 : r49;
    r49 = r27 < r22 ? r50 : r49;
    r47 = r22 * r49;
    r47 = fmaf(r36, r47, r50 * r35);
    r47 = r22 * r47;
    r47 = r47 * r37;
    r47 = r8 <= r6 ? r15 : r47;
    r48 = fmaf(r14, r47, r48);
    r47 = fmaf(r9, r47, r31);
    r47 = fmaf(r38, r51, r47);
    WriteIdx2<1024, float, float, float2>(out_translation_jac,
                                          4 * out_translation_jac_num_alloc,
                                          global_thread_idx,
                                          r48,
                                          r47);
    r51 = r10 * r38;
    r50 = r14 * r51;
    r30 = r9 * r41;
    r30 = fmaf(r51, r30, r45 * r50);
    r29 = r9 * r46;
    r29 = fmaf(r43, r50, r51 * r29);
    r52 = r9 * r47;
    r52 = fmaf(r48, r50, r51 * r52);
    WriteSum3<float, float>((float*)inout_shared, r30, r29, r52);
  };
  FlushSumShared<3, float>(out_translation_njtr,
                           0 * out_translation_njtr_num_alloc,
                           translation_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r52 = fmaf(r45, r45, r41 * r41);
    r29 = fmaf(r43, r43, r46 * r46);
    r30 = fmaf(r47, r47, r48 * r48);
    WriteSum3<float, float>((float*)inout_shared, r52, r29, r30);
  };
  FlushSumShared<3, float>(out_translation_precond_diag,
                           0 * out_translation_precond_diag_num_alloc,
                           translation_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = fmaf(r45, r43, r41 * r46);
    r45 = fmaf(r45, r48, r41 * r47);
    r48 = fmaf(r43, r48, r46 * r47);
    WriteSum3<float, float>((float*)inout_shared, r30, r45, r48);
  };
  FlushSumShared<3, float>(out_translation_precond_tril,
                           0 * out_translation_precond_tril_num_alloc,
                           translation_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = r2 * r24;
    r45 = r0 * r24;
    r45 = r45 * r14;
    r45 = fmaf(r12, r45, r42 * r48);
    r48 = r45 * r33;
    r48 = r8 <= r7 ? r45 : r48;
    r42 = r45 * r44;
    r48 = r27 < r11 ? r42 : r48;
    r42 = r45 * r34;
    r48 = r27 < r32 ? r42 : r48;
    r48 = r27 < r22 ? r45 : r48;
    r42 = r22 * r48;
    r42 = fmaf(r36, r42, r45 * r35);
    r42 = r22 * r42;
    r42 = r42 * r37;
    r42 = r8 <= r6 ? r15 : r42;
    r45 = fmaf(r14, r42, r39);
    r30 = r0 * r24;
    r30 = r30 * r38;
    r45 = fmaf(r25, r30, r45);
    r42 = fmaf(r9, r42, r31);
    r30 = r2 * r24;
    r30 = r30 * r38;
    r42 = fmaf(r25, r30, r42);
    r30 = r1 * r28;
    r30 = r30 * r14;
    r40 = fmaf(r28, r40, r12 * r30);
    r30 = r40 * r33;
    r30 = r8 <= r7 ? r40 : r30;
    r12 = r40 * r44;
    r30 = r27 < r11 ? r12 : r30;
    r12 = r40 * r34;
    r30 = r27 < r32 ? r12 : r30;
    r30 = r27 < r22 ? r40 : r30;
    r12 = r22 * r30;
    r40 = fmaf(r40, r35, r36 * r12);
    r40 = r22 * r40;
    r40 = r40 * r37;
    r40 = r8 <= r6 ? r15 : r40;
    r12 = fmaf(r14, r40, r39);
    r43 = r1 * r28;
    r43 = r43 * r38;
    r12 = fmaf(r25, r43, r12);
    r40 = fmaf(r9, r40, r31);
    r43 = r3 * r28;
    r43 = r43 * r38;
    r40 = fmaf(r25, r43, r40);
    WriteIdx4<1024, float, float, float4>(out_calib_jac,
                                          0 * out_calib_jac_num_alloc,
                                          global_thread_idx,
                                          r45,
                                          r42,
                                          r12,
                                          r40);
    r43 = r0 * r26;
    r25 = r2 * r26;
    r25 = fmaf(r9, r25, r14 * r43);
    r43 = r25 * r33;
    r43 = r8 <= r7 ? r25 : r43;
    r29 = r25 * r44;
    r43 = r27 < r11 ? r29 : r43;
    r29 = r25 * r34;
    r43 = r27 < r32 ? r29 : r43;
    r43 = r27 < r22 ? r25 : r43;
    r29 = r22 * r43;
    r25 = fmaf(r25, r35, r36 * r29);
    r25 = r22 * r25;
    r25 = r25 * r37;
    r25 = r8 <= r6 ? r15 : r25;
    r29 = fmaf(r14, r25, r39);
    r29 = fmaf(r0, r38, r29);
    r25 = fmaf(r9, r25, r31);
    r25 = fmaf(r2, r38, r25);
    r39 = fmaf(r1, r38, r39);
    r52 = r1 * r26;
    r53 = r3 * r26;
    r53 = fmaf(r9, r53, r14 * r52);
    r33 = r53 * r33;
    r33 = r8 <= r7 ? r53 : r33;
    r44 = r53 * r44;
    r33 = r27 < r11 ? r44 : r33;
    r34 = r53 * r34;
    r33 = r27 < r32 ? r34 : r33;
    r33 = r27 < r22 ? r53 : r33;
    r27 = r22 * r33;
    r27 = fmaf(r36, r27, r53 * r35);
    r27 = r22 * r27;
    r27 = r27 * r37;
    r27 = r8 <= r6 ? r15 : r27;
    r39 = fmaf(r14, r27, r39);
    r38 = fmaf(r3, r38, r31);
    r38 = fmaf(r9, r27, r38);
    WriteIdx4<1024, float, float, float4>(out_calib_jac,
                                          4 * out_calib_jac_num_alloc,
                                          global_thread_idx,
                                          r29,
                                          r25,
                                          r39,
                                          r38);
    r27 = r9 * r42;
    r27 = fmaf(r51, r27, r45 * r50);
    r31 = r9 * r40;
    r31 = fmaf(r51, r31, r12 * r50);
    r15 = r9 * r25;
    r15 = fmaf(r29, r50, r51 * r15);
    r6 = r9 * r38;
    r50 = fmaf(r39, r50, r51 * r6);
    WriteSum4<float, float>((float*)inout_shared, r27, r31, r15, r50);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           0 * out_calib_njtr_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = fmaf(r45, r45, r42 * r42);
    r15 = fmaf(r40, r40, r12 * r12);
    r31 = fmaf(r29, r29, r25 * r25);
    r27 = fmaf(r38, r38, r39 * r39);
    WriteSum4<float, float>((float*)inout_shared, r50, r15, r31, r27);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           0 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fmaf(r42, r40, r45 * r12);
    r31 = fmaf(r45, r29, r42 * r25);
    r45 = fmaf(r42, r38, r45 * r39);
    r15 = fmaf(r40, r25, r12 * r29);
    WriteSum4<float, float>((float*)inout_shared, r27, r31, r45, r15);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           0 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r12 = fmaf(r40, r38, r12 * r39);
    r39 = fmaf(r25, r38, r29 * r39);
    WriteSum2<float, float>((float*)inout_shared, r12, r39);
  };
  FlushSumShared<2, float>(out_calib_precond_tril,
                           4 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
}

void PinholeFixedRotationFixedPointResJac(
    float* rotation,
    unsigned int rotation_num_alloc,
    float* translation,
    unsigned int translation_num_alloc,
    SharedIndex* translation_indices,
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
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeFixedRotationFixedPointResJacKernel<<<n_blocks, 1024>>>(
      rotation,
      rotation_num_alloc,
      translation,
      translation_num_alloc,
      translation_indices,
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
      problem_size);
}

}  // namespace caspar