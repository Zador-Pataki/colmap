#include "kernel_pinhole_split_fixed_rotation_fixed_focal_fixed_principal_point_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedRotationFixedFocalFixedPrincipalPointFixedPointResJacKernel(
        float* rotation,
        unsigned int rotation_num_alloc,
        float* translation,
        unsigned int translation_num_alloc,
        SharedIndex* translation_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* weight_loss,
        unsigned int weight_loss_num_alloc,
        float* focal,
        unsigned int focal_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
        float* point,
        unsigned int point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_translation_njtr,
        unsigned int out_translation_njtr_num_alloc,
        float* const out_translation_precond_diag,
        unsigned int out_translation_precond_diag_num_alloc,
        float* const out_translation_precond_tril,
        unsigned int out_translation_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex translation_indices_loc[1024];
  translation_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? translation_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46;

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
    r4 = r16 * r18;
    r23 = 2.00000000000000000e+00;
    r24 = r15 * r23;
    r25 = r17 * r24;
    r4 = fmaf(r23, r4, r25);
    r26 = r18 * r14;
    r27 = r16 * r24;
    r28 = fmaf(r17, r26, r27);
    r22 = fmaf(r13, r4, r22);
    r22 = fmaf(r12, r28, r22);
    ReadIdx2<1024, float, float, float2>(
        focal, 0 * focal_num_alloc, global_thread_idx, r28, r4);
    r29 = 9.99999999999999955e-07;
    r30 = r15 * r15;
    r30 = r30 * r14;
    r21 = r30 + r21;
    r21 = fmaf(r13, r21, r10);
    r25 = fmaf(r16, r26, r25);
    r10 = r16 * r17;
    r10 = r10 * r23;
    r24 = fmaf(r18, r24, r10);
    r21 = fmaf(r11, r25, r21);
    r21 = fmaf(r12, r24, r21);
    r24 = copysign(1.0, r21);
    r24 = fmaf(r29, r24, r21);
    r21 = 1.0 / r24;
    r25 = r28 * r21;
    r6 = fmaf(r22, r25, r6);
    r7 = fmaf(r7, r8, r5);
    r19 = r20 + r19;
    r19 = r19 + r30;
    r19 = fmaf(r12, r19, r9);
    r26 = fmaf(r15, r26, r10);
    r15 = r17 * r18;
    r15 = fmaf(r23, r15, r27);
    r19 = fmaf(r13, r26, r19);
    r19 = fmaf(r11, r15, r19);
    r19 = r4 * r19;
    r7 = fmaf(r21, r19, r7);
    r15 = fmaf(r1, r7, r0 * r6);
    ReadIdx3<1024, float, float, float4>(weight_loss,
                                         4 * weight_loss_num_alloc,
                                         global_thread_idx,
                                         r11,
                                         r26,
                                         r13);
    r27 = 0.00000000000000000e+00;
    r13 = fmaxf(r13, r27);
    r10 = sqrtf(r13);
    r7 = fmaf(r3, r7, r2 * r6);
    r6 = fmaf(r7, r7, r15 * r15);
    r12 = 5.00000000000000000e-01;
    r26 = fmaxf(r26, r29);
    r9 = r26 * r26;
    r30 = r23 * r26;
    r5 = fmaxf(r29, r6);
    r14 = sqrtf(r5);
    r30 = fmaf(r14, r30, r8 * r9);
    r30 = r6 <= r9 ? r6 : r30;
    r14 = 2.50000000000000000e+00;
    r31 = 1.0 / r9;
    r31 = fmaf(r6, r31, r20);
    r32 = logf(r31);
    r32 = r32 * r9;
    r30 = r11 < r14 ? r32 : r30;
    r32 = 1.50000000000000000e+00;
    r33 = sqrtf(r31);
    r33 = r8 + r33;
    r33 = r23 * r33;
    r33 = r33 * r9;
    r30 = r11 < r32 ? r33 : r30;
    r30 = r11 < r12 ? r6 : r30;
    r33 = fmaxf(r27, r30);
    r34 = 1.0 / r5;
    r34 = r13 * r34;
    r35 = r33 * r34;
    r36 = sqrtf(r35);
    r36 = r6 <= r29 ? r10 : r36;
    r10 = r15 * r36;
    r37 = r7 * r36;
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r10, r37);
    r37 = r23 * r7;
    r10 = r2 * r25;
    r38 = r0 * r23;
    r38 = r38 * r15;
    r38 = fmaf(r25, r38, r10 * r37);
    r37 = r12 * r26;
    r39 = -9.99999999999999955e-07;
    r39 = r39 + r6;
    r39 = copysign(1.0, r39);
    r39 = r20 + r39;
    r40 = rsqrtf(r5);
    r37 = r37 * r39;
    r37 = r37 * r40;
    r40 = r38 * r37;
    r40 = r6 <= r9 ? r38 : r40;
    r41 = 1.0 / r31;
    r42 = r38 * r41;
    r40 = r11 < r14 ? r42 : r40;
    r31 = rsqrtf(r31);
    r42 = r38 * r31;
    r40 = r11 < r32 ? r42 : r40;
    r40 = r11 < r12 ? r38 : r40;
    r42 = r12 * r40;
    r30 = copysign(1.0, r30);
    r30 = r20 + r30;
    r34 = r30 * r34;
    r33 = r13 * r33;
    r13 = -5.00000000000000000e-01;
    r5 = r5 * r5;
    r5 = 1.0 / r5;
    r33 = r33 * r13;
    r33 = r33 * r39;
    r33 = r33 * r5;
    r38 = fmaf(r38, r33, r34 * r42);
    r38 = r12 * r38;
    r35 = rsqrtf(r35);
    r38 = r38 * r35;
    r38 = r6 <= r29 ? r27 : r38;
    r42 = r8 * r15;
    r5 = 2.50000000000000000e-01;
    r39 = r6 <= r9 ? r27 : r27;
    r39 = r11 < r14 ? r27 : r39;
    r39 = r11 < r32 ? r27 : r39;
    r39 = r11 < r12 ? r27 : r39;
    r39 = r5 * r39;
    r39 = r39 * r35;
    r39 = r39 * r34;
    r39 = r6 <= r29 ? r27 : r39;
    r42 = r42 * r39;
    r5 = fmaf(r15, r38, r42);
    r13 = r0 * r36;
    r5 = fmaf(r25, r13, r5);
    r13 = r15 * r5;
    r25 = r8 * r36;
    r30 = r7 * r25;
    r20 = r8 * r7;
    r20 = r20 * r39;
    r38 = fmaf(r7, r38, r20);
    r38 = fmaf(r36, r10, r38);
    r13 = fmaf(r38, r30, r25 * r13);
    r10 = r3 * r4;
    r10 = r10 * r23;
    r10 = r10 * r7;
    r39 = r1 * r4;
    r39 = r39 * r23;
    r39 = r39 * r15;
    r39 = fmaf(r21, r39, r21 * r10);
    r10 = r39 * r37;
    r10 = r6 <= r9 ? r39 : r10;
    r43 = r39 * r41;
    r10 = r11 < r14 ? r43 : r10;
    r43 = r39 * r31;
    r10 = r11 < r32 ? r43 : r10;
    r10 = r11 < r12 ? r39 : r10;
    r43 = r12 * r10;
    r39 = fmaf(r39, r33, r34 * r43);
    r39 = r12 * r39;
    r39 = r39 * r35;
    r39 = r6 <= r29 ? r27 : r39;
    r43 = fmaf(r7, r39, r20);
    r44 = r3 * r4;
    r44 = r44 * r36;
    r43 = fmaf(r21, r44, r43);
    r39 = fmaf(r15, r39, r42);
    r44 = r1 * r4;
    r44 = r44 * r36;
    r39 = fmaf(r21, r44, r39);
    r44 = r15 * r39;
    r44 = fmaf(r25, r44, r43 * r30);
    r21 = r23 * r15;
    r28 = r28 * r8;
    r24 = r24 * r24;
    r24 = 1.0 / r24;
    r28 = r28 * r22;
    r28 = r28 * r24;
    r22 = r1 * r8;
    r22 = r22 * r24;
    r22 = fmaf(r19, r22, r0 * r28);
    r45 = r23 * r7;
    r46 = r3 * r8;
    r46 = r46 * r24;
    r46 = fmaf(r19, r46, r2 * r28);
    r45 = fmaf(r46, r45, r22 * r21);
    r37 = r45 * r37;
    r37 = r6 <= r9 ? r45 : r37;
    r41 = r45 * r41;
    r37 = r11 < r14 ? r41 : r37;
    r31 = r45 * r31;
    r37 = r11 < r32 ? r31 : r37;
    r37 = r11 < r12 ? r45 : r37;
    r11 = r12 * r37;
    r11 = fmaf(r34, r11, r45 * r33);
    r11 = r12 * r11;
    r11 = r11 * r35;
    r11 = r6 <= r29 ? r27 : r11;
    r20 = fmaf(r7, r11, r20);
    r20 = fmaf(r36, r46, r20);
    r22 = fmaf(r36, r22, r42);
    r22 = fmaf(r15, r11, r22);
    r11 = r15 * r22;
    r11 = fmaf(r25, r11, r20 * r30);
    WriteSum3<float, float>((float*)inout_shared, r13, r44, r11);
  };
  FlushSumShared<3, float>(out_translation_njtr,
                           0 * out_translation_njtr_num_alloc,
                           translation_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = fmaf(r5, r5, r38 * r38);
    r44 = fmaf(r39, r39, r43 * r43);
    r13 = fmaf(r20, r20, r22 * r22);
    WriteSum3<float, float>((float*)inout_shared, r11, r44, r13);
  };
  FlushSumShared<3, float>(out_translation_precond_diag,
                           0 * out_translation_precond_diag_num_alloc,
                           translation_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = fmaf(r5, r39, r38 * r43);
    r38 = fmaf(r38, r20, r5 * r22);
    r20 = fmaf(r39, r22, r43 * r20);
    WriteSum3<float, float>((float*)inout_shared, r13, r38, r20);
  };
  FlushSumShared<3, float>(out_translation_precond_tril,
                           0 * out_translation_precond_tril_num_alloc,
                           translation_indices_loc,
                           (float*)inout_shared);
}

void PinholeSplitFixedRotationFixedFocalFixedPrincipalPointFixedPointResJac(
    float* rotation,
    unsigned int rotation_num_alloc,
    float* translation,
    unsigned int translation_num_alloc,
    SharedIndex* translation_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* weight_loss,
    unsigned int weight_loss_num_alloc,
    float* focal,
    unsigned int focal_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_translation_njtr,
    unsigned int out_translation_njtr_num_alloc,
    float* const out_translation_precond_diag,
    unsigned int out_translation_precond_diag_num_alloc,
    float* const out_translation_precond_tril,
    unsigned int out_translation_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitFixedRotationFixedFocalFixedPrincipalPointFixedPointResJacKernel<<<
      n_blocks,
      1024>>>(rotation,
              rotation_num_alloc,
              translation,
              translation_num_alloc,
              translation_indices,
              pixel,
              pixel_num_alloc,
              weight_loss,
              weight_loss_num_alloc,
              focal,
              focal_num_alloc,
              principal_point,
              principal_point_num_alloc,
              point,
              point_num_alloc,
              out_res,
              out_res_num_alloc,
              out_translation_njtr,
              out_translation_njtr_num_alloc,
              out_translation_precond_diag,
              out_translation_precond_diag_num_alloc,
              out_translation_precond_tril,
              out_translation_precond_tril_num_alloc,
              problem_size);
}

}  // namespace caspar