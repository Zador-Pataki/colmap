#include "kernel_pinhole_split_fixed_rotation_fixed_focal_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedRotationFixedFocalFixedPointResJacFirstKernel(
        float* rotation,
        unsigned int rotation_num_alloc,
        float* translation,
        unsigned int translation_num_alloc,
        SharedIndex* translation_indices,
        float* principal_point,
        unsigned int principal_point_num_alloc,
        SharedIndex* principal_point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* weight_loss,
        unsigned int weight_loss_num_alloc,
        float* focal,
        unsigned int focal_num_alloc,
        float* point,
        unsigned int point_num_alloc,
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
        float* out_principal_point_jac,
        unsigned int out_principal_point_jac_num_alloc,
        float* const out_principal_point_njtr,
        unsigned int out_principal_point_njtr_num_alloc,
        float* const out_principal_point_precond_diag,
        unsigned int out_principal_point_precond_diag_num_alloc,
        float* const out_principal_point_precond_tril,
        unsigned int out_principal_point_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex translation_indices_loc[1024];
  translation_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? translation_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex principal_point_indices_loc[1024];
  principal_point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? principal_point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(weight_loss,
                                         0 * weight_loss_num_alloc,
                                         global_thread_idx,
                                         r0,
                                         r1,
                                         r2,
                                         r3);
  };
  LoadShared<2, float, float>(principal_point,
                              0 * principal_point_num_alloc,
                              principal_point_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared,
                       principal_point_indices_loc[threadIdx.x].target,
                       r4,
                       r5);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
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
  if (global_thread_idx < problem_size) {
    ReadIdx3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r11, r12, r13);
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
    r22 = fmaf(r11, r22, r4);
    r4 = r15 * r17;
    r23 = 2.00000000000000000e+00;
    r4 = r4 * r23;
    r24 = r16 * r18;
    r24 = fmaf(r23, r24, r4);
    r25 = r15 * r16;
    r25 = r25 * r23;
    r26 = r18 * r14;
    r27 = fmaf(r17, r26, r25);
    r22 = fmaf(r13, r24, r22);
    r22 = fmaf(r12, r27, r22);
    ReadIdx2<1024, float, float, float2>(
        focal, 0 * focal_num_alloc, global_thread_idx, r27, r24);
    r28 = 9.99999999999999955e-07;
    r29 = r15 * r15;
    r29 = r14 * r29;
    r21 = r29 + r21;
    r21 = fmaf(r13, r21, r10);
    r4 = fmaf(r16, r26, r4);
    r10 = r15 * r18;
    r14 = r16 * r17;
    r14 = r14 * r23;
    r10 = fmaf(r23, r10, r14);
    r21 = fmaf(r11, r4, r21);
    r21 = fmaf(r12, r10, r21);
    r10 = copysign(1.0, r21);
    r10 = fmaf(r28, r10, r21);
    r21 = 1.0 / r10;
    r4 = r27 * r21;
    r6 = fmaf(r22, r4, r6);
    r7 = fmaf(r7, r8, r5);
    r19 = r20 + r19;
    r19 = r19 + r29;
    r19 = fmaf(r12, r19, r9);
    r26 = fmaf(r15, r26, r14);
    r14 = r17 * r18;
    r14 = fmaf(r23, r14, r25);
    r19 = fmaf(r13, r26, r19);
    r19 = fmaf(r11, r14, r19);
    r19 = r24 * r19;
    r7 = fmaf(r21, r19, r7);
    r14 = fmaf(r1, r7, r0 * r6);
    ReadIdx3<1024, float, float, float4>(weight_loss,
                                         4 * weight_loss_num_alloc,
                                         global_thread_idx,
                                         r11,
                                         r26,
                                         r13);
    r25 = 0.00000000000000000e+00;
    r13 = fmaxf(r13, r25);
    r12 = sqrtf(r13);
    r7 = fmaf(r3, r7, r2 * r6);
    r6 = fmaf(r7, r7, r14 * r14);
    r9 = 5.00000000000000000e-01;
    r26 = fmaxf(r26, r28);
    r29 = r26 * r26;
    r5 = r8 * r26;
    r30 = r23 * r26;
    r31 = fmaxf(r28, r6);
    r32 = sqrtf(r31);
    r30 = fmaf(r32, r30, r26 * r5);
    r30 = r6 <= r29 ? r6 : r30;
    r5 = 2.50000000000000000e+00;
    r32 = r26 * r26;
    r33 = 1.0 / r29;
    r33 = fmaf(r6, r33, r20);
    r34 = logf(r33);
    r32 = r32 * r34;
    r30 = r11 < r5 ? r32 : r30;
    r32 = 1.50000000000000000e+00;
    r34 = r23 * r26;
    r35 = sqrtf(r33);
    r35 = r8 + r35;
    r34 = r34 * r26;
    r34 = r34 * r35;
    r30 = r11 < r32 ? r34 : r30;
    r30 = r11 < r9 ? r6 : r30;
    r34 = fmaxf(r25, r30);
    r35 = 1.0 / r31;
    r35 = r13 * r35;
    r36 = r34 * r35;
    r37 = sqrtf(r36);
    r37 = r6 <= r28 ? r12 : r37;
    r12 = r14 * r37;
    r38 = r7 * r37;
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r12, r38);
    r38 = r14 * r14;
    r38 = r38 * r37;
    r12 = r7 * r7;
    r12 = r12 * r37;
    r12 = fmaf(r37, r12, r37 * r38);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r12);
  if (global_thread_idx < problem_size) {
    r12 = r23 * r7;
    r38 = r2 * r4;
    r39 = r0 * r23;
    r39 = r39 * r14;
    r39 = fmaf(r4, r39, r12 * r38);
    r40 = r9 * r26;
    r41 = -9.99999999999999955e-07;
    r41 = r41 + r6;
    r41 = copysign(1.0, r41);
    r41 = r20 + r41;
    r42 = rsqrtf(r31);
    r40 = r40 * r41;
    r40 = r40 * r42;
    r42 = r39 * r40;
    r42 = r6 <= r29 ? r39 : r42;
    r43 = 1.0 / r33;
    r44 = r39 * r43;
    r42 = r11 < r5 ? r44 : r42;
    r33 = rsqrtf(r33);
    r44 = r39 * r33;
    r42 = r11 < r32 ? r44 : r42;
    r42 = r11 < r9 ? r39 : r42;
    r44 = r9 * r42;
    r30 = copysign(1.0, r30);
    r30 = r20 + r30;
    r35 = r30 * r35;
    r34 = r13 * r34;
    r13 = -5.00000000000000000e-01;
    r31 = r31 * r31;
    r31 = 1.0 / r31;
    r34 = r34 * r13;
    r34 = r34 * r41;
    r34 = r34 * r31;
    r39 = fmaf(r39, r34, r35 * r44);
    r39 = r9 * r39;
    r36 = rsqrtf(r36);
    r39 = r39 * r36;
    r39 = r6 <= r28 ? r25 : r39;
    r44 = r8 * r14;
    r31 = 2.50000000000000000e-01;
    r41 = r6 <= r29 ? r25 : r25;
    r41 = r11 < r5 ? r25 : r41;
    r41 = r11 < r32 ? r25 : r41;
    r41 = r11 < r9 ? r25 : r41;
    r41 = r31 * r41;
    r41 = r41 * r36;
    r41 = r41 * r35;
    r41 = r6 <= r28 ? r25 : r41;
    r44 = r44 * r41;
    r31 = fmaf(r14, r39, r44);
    r13 = r0 * r37;
    r31 = fmaf(r4, r13, r31);
    r13 = r8 * r7;
    r13 = r13 * r41;
    r39 = fmaf(r7, r39, r13);
    r39 = fmaf(r37, r38, r39);
    r38 = r24 * r21;
    r41 = r3 * r23;
    r41 = r41 * r7;
    r4 = r1 * r24;
    r4 = r4 * r23;
    r4 = r4 * r14;
    r4 = fmaf(r21, r4, r41 * r38);
    r38 = r4 * r40;
    r38 = r6 <= r29 ? r4 : r38;
    r30 = r4 * r43;
    r38 = r11 < r5 ? r30 : r38;
    r30 = r4 * r33;
    r38 = r11 < r32 ? r30 : r38;
    r38 = r11 < r9 ? r4 : r38;
    r30 = r9 * r38;
    r4 = fmaf(r4, r34, r35 * r30);
    r4 = r9 * r4;
    r4 = r4 * r36;
    r4 = r6 <= r28 ? r25 : r4;
    r30 = fmaf(r14, r4, r44);
    r20 = r1 * r24;
    r20 = r20 * r37;
    r30 = fmaf(r21, r20, r30);
    r4 = fmaf(r7, r4, r13);
    r20 = r3 * r24;
    r20 = r20 * r37;
    r4 = fmaf(r21, r20, r4);
    WriteIdx4<1024, float, float, float4>(out_translation_jac,
                                          0 * out_translation_jac_num_alloc,
                                          global_thread_idx,
                                          r31,
                                          r39,
                                          r30,
                                          r4);
    r27 = r27 * r8;
    r10 = r10 * r10;
    r10 = 1.0 / r10;
    r27 = r27 * r22;
    r27 = r27 * r10;
    r22 = r1 * r8;
    r22 = r22 * r10;
    r22 = fmaf(r19, r22, r0 * r27);
    r20 = fmaf(r37, r22, r44);
    r45 = r23 * r14;
    r46 = r3 * r8;
    r46 = r46 * r10;
    r46 = fmaf(r19, r46, r2 * r27);
    r45 = fmaf(r46, r12, r22 * r45);
    r22 = r45 * r40;
    r22 = r6 <= r29 ? r45 : r22;
    r27 = r45 * r43;
    r22 = r11 < r5 ? r27 : r22;
    r27 = r45 * r33;
    r22 = r11 < r32 ? r27 : r22;
    r22 = r11 < r9 ? r45 : r22;
    r27 = r9 * r22;
    r27 = fmaf(r35, r27, r45 * r34);
    r27 = r9 * r27;
    r27 = r27 * r36;
    r27 = r6 <= r28 ? r25 : r27;
    r20 = fmaf(r14, r27, r20);
    r27 = fmaf(r7, r27, r13);
    r27 = fmaf(r37, r46, r27);
    WriteIdx2<1024, float, float, float2>(out_translation_jac,
                                          4 * out_translation_jac_num_alloc,
                                          global_thread_idx,
                                          r20,
                                          r27);
    r46 = r8 * r37;
    r45 = r14 * r46;
    r19 = r7 * r39;
    r19 = fmaf(r46, r19, r31 * r45);
    r10 = r7 * r4;
    r10 = fmaf(r30, r45, r46 * r10);
    r47 = r7 * r27;
    r47 = fmaf(r20, r45, r46 * r47);
    WriteSum3<float, float>((float*)inout_shared, r19, r10, r47);
  };
  FlushSumShared<3, float>(out_translation_njtr,
                           0 * out_translation_njtr_num_alloc,
                           translation_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = fmaf(r31, r31, r39 * r39);
    r10 = fmaf(r30, r30, r4 * r4);
    r19 = fmaf(r27, r27, r20 * r20);
    WriteSum3<float, float>((float*)inout_shared, r47, r10, r19);
  };
  FlushSumShared<3, float>(out_translation_precond_diag,
                           0 * out_translation_precond_diag_num_alloc,
                           translation_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = fmaf(r31, r30, r39 * r4);
    r31 = fmaf(r39, r27, r31 * r20);
    r20 = fmaf(r30, r20, r4 * r27);
    WriteSum3<float, float>((float*)inout_shared, r19, r31, r20);
  };
  FlushSumShared<3, float>(out_translation_precond_tril,
                           0 * out_translation_precond_tril_num_alloc,
                           translation_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r20 = fmaf(r0, r37, r44);
    r31 = r0 * r23;
    r12 = fmaf(r2, r12, r14 * r31);
    r31 = r12 * r40;
    r31 = r6 <= r29 ? r12 : r31;
    r19 = r12 * r43;
    r31 = r11 < r5 ? r19 : r31;
    r19 = r12 * r33;
    r31 = r11 < r32 ? r19 : r31;
    r31 = r11 < r9 ? r12 : r31;
    r19 = r9 * r31;
    r12 = fmaf(r12, r34, r35 * r19);
    r12 = r9 * r12;
    r12 = r12 * r36;
    r12 = r6 <= r28 ? r25 : r12;
    r20 = fmaf(r14, r12, r20);
    r2 = fmaf(r2, r37, r13);
    r2 = fmaf(r7, r12, r2);
    r44 = fmaf(r1, r37, r44);
    r12 = r1 * r23;
    r12 = fmaf(r14, r12, r41);
    r40 = r12 * r40;
    r40 = r6 <= r29 ? r12 : r40;
    r43 = r12 * r43;
    r40 = r11 < r5 ? r43 : r40;
    r33 = r12 * r33;
    r40 = r11 < r32 ? r33 : r40;
    r40 = r11 < r9 ? r12 : r40;
    r11 = r9 * r40;
    r11 = fmaf(r35, r11, r12 * r34);
    r11 = r9 * r11;
    r11 = r11 * r36;
    r11 = r6 <= r28 ? r25 : r11;
    r44 = fmaf(r14, r11, r44);
    r13 = fmaf(r3, r37, r13);
    r13 = fmaf(r7, r11, r13);
    WriteIdx4<1024, float, float, float4>(out_principal_point_jac,
                                          0 * out_principal_point_jac_num_alloc,
                                          global_thread_idx,
                                          r20,
                                          r2,
                                          r44,
                                          r13);
    r11 = r7 * r2;
    r11 = fmaf(r20, r45, r46 * r11);
    r25 = r7 * r13;
    r45 = fmaf(r44, r45, r46 * r25);
    WriteSum2<float, float>((float*)inout_shared, r11, r45);
  };
  FlushSumShared<2, float>(out_principal_point_njtr,
                           0 * out_principal_point_njtr_num_alloc,
                           principal_point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r45 = fmaf(r2, r2, r20 * r20);
    r11 = fmaf(r13, r13, r44 * r44);
    WriteSum2<float, float>((float*)inout_shared, r45, r11);
  };
  FlushSumShared<2, float>(out_principal_point_precond_diag,
                           0 * out_principal_point_precond_diag_num_alloc,
                           principal_point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r44 = fmaf(r20, r44, r2 * r13);
    WriteSum1<float, float>((float*)inout_shared, r44);
  };
  FlushSumShared<1, float>(out_principal_point_precond_tril,
                           0 * out_principal_point_precond_tril_num_alloc,
                           principal_point_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void PinholeSplitFixedRotationFixedFocalFixedPointResJacFirst(
    float* rotation,
    unsigned int rotation_num_alloc,
    float* translation,
    unsigned int translation_num_alloc,
    SharedIndex* translation_indices,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    SharedIndex* principal_point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* weight_loss,
    unsigned int weight_loss_num_alloc,
    float* focal,
    unsigned int focal_num_alloc,
    float* point,
    unsigned int point_num_alloc,
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
    float* out_principal_point_jac,
    unsigned int out_principal_point_jac_num_alloc,
    float* const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    float* const out_principal_point_precond_diag,
    unsigned int out_principal_point_precond_diag_num_alloc,
    float* const out_principal_point_precond_tril,
    unsigned int out_principal_point_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitFixedRotationFixedFocalFixedPointResJacFirstKernel<<<n_blocks,
                                                                   1024>>>(
      rotation,
      rotation_num_alloc,
      translation,
      translation_num_alloc,
      translation_indices,
      principal_point,
      principal_point_num_alloc,
      principal_point_indices,
      pixel,
      pixel_num_alloc,
      weight_loss,
      weight_loss_num_alloc,
      focal,
      focal_num_alloc,
      point,
      point_num_alloc,
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
      out_principal_point_jac,
      out_principal_point_jac_num_alloc,
      out_principal_point_njtr,
      out_principal_point_njtr_num_alloc,
      out_principal_point_precond_diag,
      out_principal_point_precond_diag_num_alloc,
      out_principal_point_precond_tril,
      out_principal_point_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar