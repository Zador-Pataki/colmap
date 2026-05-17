#include "kernel_pinhole_fixed_rotation_fixed_calib_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeFixedRotationFixedCalibFixedPointResJacKernel(
        float* rotation,
        unsigned int rotation_num_alloc,
        float* translation,
        unsigned int translation_num_alloc,
        SharedIndex* translation_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* weight_loss,
        unsigned int weight_loss_num_alloc,
        float* calib,
        unsigned int calib_num_alloc,
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
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45;

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
  if (global_thread_idx < problem_size) {
    ReadIdx3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r13, r14, r15);
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
    r24 = fmaf(r13, r24, r6);
    r6 = r18 * r20;
    r25 = 2.00000000000000000e+00;
    r26 = r17 * r25;
    r27 = r19 * r26;
    r6 = fmaf(r25, r6, r27);
    r28 = r20 * r16;
    r29 = r18 * r26;
    r30 = fmaf(r19, r28, r29);
    r24 = fmaf(r15, r6, r24);
    r24 = fmaf(r14, r30, r24);
    r30 = 9.99999999999999955e-07;
    r6 = r17 * r17;
    r6 = r6 * r16;
    r23 = r6 + r23;
    r23 = fmaf(r15, r23, r12);
    r27 = fmaf(r18, r28, r27);
    r12 = r18 * r19;
    r12 = r12 * r25;
    r26 = fmaf(r20, r26, r12);
    r23 = fmaf(r13, r27, r23);
    r23 = fmaf(r14, r26, r23);
    r26 = copysign(1.0, r23);
    r26 = fmaf(r30, r26, r23);
    r23 = 1.0 / r26;
    r27 = r4 * r23;
    r8 = fmaf(r24, r27, r8);
    r9 = fmaf(r9, r10, r7);
    r21 = r22 + r21;
    r21 = r21 + r6;
    r21 = fmaf(r14, r21, r11);
    r28 = fmaf(r17, r28, r12);
    r17 = r19 * r20;
    r17 = fmaf(r25, r17, r29);
    r21 = fmaf(r15, r28, r21);
    r21 = fmaf(r13, r17, r21);
    r21 = r5 * r21;
    r9 = fmaf(r23, r21, r9);
    r17 = fmaf(r1, r9, r0 * r8);
    ReadIdx3<1024, float, float, float4>(weight_loss,
                                         4 * weight_loss_num_alloc,
                                         global_thread_idx,
                                         r13,
                                         r28,
                                         r15);
    r29 = 0.00000000000000000e+00;
    r15 = fmaxf(r15, r29);
    r12 = sqrtf(r15);
    r9 = fmaf(r3, r9, r2 * r8);
    r8 = fmaf(r17, r17, r9 * r9);
    r14 = 5.00000000000000000e-01;
    r28 = fmaxf(r28, r30);
    r11 = r28 * r28;
    r6 = r25 * r28;
    r7 = fmaxf(r30, r8);
    r16 = sqrtf(r7);
    r6 = fmaf(r10, r11, r16 * r6);
    r6 = r8 <= r11 ? r8 : r6;
    r16 = 2.50000000000000000e+00;
    r31 = 1.0 / r11;
    r31 = fmaf(r8, r31, r22);
    r32 = logf(r31);
    r32 = r32 * r11;
    r6 = r13 < r16 ? r32 : r6;
    r32 = 1.50000000000000000e+00;
    r33 = sqrtf(r31);
    r33 = r10 + r33;
    r33 = r25 * r33;
    r33 = r33 * r11;
    r6 = r13 < r32 ? r33 : r6;
    r6 = r13 < r14 ? r8 : r6;
    r33 = fmaxf(r29, r6);
    r34 = 1.0 / r7;
    r34 = r15 * r34;
    r35 = r33 * r34;
    r36 = sqrtf(r35);
    r36 = r8 <= r30 ? r12 : r36;
    r12 = r17 * r36;
    r37 = r9 * r36;
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r12, r37);
    r37 = r10 * r17;
    r12 = 2.50000000000000000e-01;
    r38 = r8 <= r11 ? r29 : r29;
    r38 = r13 < r16 ? r29 : r38;
    r38 = r13 < r32 ? r29 : r38;
    r38 = r13 < r14 ? r29 : r38;
    r38 = r12 * r38;
    r35 = rsqrtf(r35);
    r6 = copysign(1.0, r6);
    r6 = r22 + r6;
    r34 = r6 * r34;
    r38 = r38 * r35;
    r38 = r38 * r34;
    r38 = r8 <= r30 ? r29 : r38;
    r37 = r37 * r38;
    r6 = r25 * r17;
    r12 = r0 * r27;
    r39 = r2 * r25;
    r39 = r39 * r9;
    r39 = fmaf(r27, r39, r12 * r6);
    r6 = r14 * r28;
    r40 = -9.99999999999999955e-07;
    r40 = r40 + r8;
    r40 = copysign(1.0, r40);
    r40 = r22 + r40;
    r22 = rsqrtf(r7);
    r6 = r6 * r40;
    r6 = r6 * r22;
    r22 = r39 * r6;
    r22 = r8 <= r11 ? r39 : r22;
    r41 = 1.0 / r31;
    r42 = r39 * r41;
    r22 = r13 < r16 ? r42 : r22;
    r31 = rsqrtf(r31);
    r42 = r39 * r31;
    r22 = r13 < r32 ? r42 : r22;
    r22 = r13 < r14 ? r39 : r22;
    r42 = r14 * r22;
    r33 = r15 * r33;
    r15 = -5.00000000000000000e-01;
    r7 = r7 * r7;
    r7 = 1.0 / r7;
    r33 = r33 * r15;
    r33 = r33 * r40;
    r33 = r33 * r7;
    r39 = fmaf(r39, r33, r34 * r42);
    r39 = r14 * r39;
    r39 = r39 * r35;
    r39 = r8 <= r30 ? r29 : r39;
    r42 = fmaf(r17, r39, r37);
    r42 = fmaf(r36, r12, r42);
    r12 = r17 * r42;
    r7 = r10 * r36;
    r40 = r9 * r7;
    r15 = r10 * r9;
    r15 = r15 * r38;
    r39 = fmaf(r9, r39, r15);
    r38 = r2 * r36;
    r39 = fmaf(r27, r38, r39);
    r12 = fmaf(r39, r40, r7 * r12);
    r38 = r3 * r5;
    r38 = r38 * r25;
    r38 = r38 * r9;
    r27 = r1 * r5;
    r27 = r27 * r25;
    r27 = r27 * r17;
    r27 = fmaf(r23, r27, r23 * r38);
    r38 = r27 * r6;
    r38 = r8 <= r11 ? r27 : r38;
    r43 = r27 * r41;
    r38 = r13 < r16 ? r43 : r38;
    r43 = r27 * r31;
    r38 = r13 < r32 ? r43 : r38;
    r38 = r13 < r14 ? r27 : r38;
    r43 = r14 * r38;
    r43 = fmaf(r34, r43, r27 * r33);
    r43 = r14 * r43;
    r43 = r43 * r35;
    r43 = r8 <= r30 ? r29 : r43;
    r27 = fmaf(r9, r43, r15);
    r44 = r3 * r5;
    r44 = r44 * r36;
    r27 = fmaf(r23, r44, r27);
    r43 = fmaf(r17, r43, r37);
    r44 = r1 * r5;
    r44 = r44 * r36;
    r43 = fmaf(r23, r44, r43);
    r44 = r17 * r43;
    r44 = fmaf(r7, r44, r27 * r40);
    r23 = r25 * r17;
    r45 = r1 * r10;
    r26 = r26 * r26;
    r26 = 1.0 / r26;
    r45 = r45 * r26;
    r4 = r4 * r10;
    r4 = r4 * r24;
    r4 = r4 * r26;
    r0 = fmaf(r0, r4, r21 * r45);
    r45 = r25 * r9;
    r24 = r3 * r10;
    r24 = r24 * r26;
    r4 = fmaf(r2, r4, r21 * r24);
    r45 = fmaf(r4, r45, r0 * r23);
    r6 = r45 * r6;
    r6 = r8 <= r11 ? r45 : r6;
    r41 = r45 * r41;
    r6 = r13 < r16 ? r41 : r6;
    r31 = r45 * r31;
    r6 = r13 < r32 ? r31 : r6;
    r6 = r13 < r14 ? r45 : r6;
    r13 = r14 * r6;
    r13 = fmaf(r34, r13, r45 * r33);
    r13 = r14 * r13;
    r13 = r13 * r35;
    r13 = r8 <= r30 ? r29 : r13;
    r15 = fmaf(r9, r13, r15);
    r15 = fmaf(r36, r4, r15);
    r0 = fmaf(r36, r0, r37);
    r0 = fmaf(r17, r13, r0);
    r13 = r17 * r0;
    r13 = fmaf(r7, r13, r15 * r40);
    WriteSum3<float, float>((float*)inout_shared, r12, r44, r13);
  };
  FlushSumShared<3, float>(out_translation_njtr,
                           0 * out_translation_njtr_num_alloc,
                           translation_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = fmaf(r42, r42, r39 * r39);
    r44 = fmaf(r43, r43, r27 * r27);
    r12 = fmaf(r15, r15, r0 * r0);
    WriteSum3<float, float>((float*)inout_shared, r13, r44, r12);
  };
  FlushSumShared<3, float>(out_translation_precond_diag,
                           0 * out_translation_precond_diag_num_alloc,
                           translation_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r12 = fmaf(r42, r43, r39 * r27);
    r39 = fmaf(r42, r0, r39 * r15);
    r15 = fmaf(r43, r0, r27 * r15);
    WriteSum3<float, float>((float*)inout_shared, r12, r39, r15);
  };
  FlushSumShared<3, float>(out_translation_precond_tril,
                           0 * out_translation_precond_tril_num_alloc,
                           translation_indices_loc,
                           (float*)inout_shared);
}

void PinholeFixedRotationFixedCalibFixedPointResJac(
    float* rotation,
    unsigned int rotation_num_alloc,
    float* translation,
    unsigned int translation_num_alloc,
    SharedIndex* translation_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* weight_loss,
    unsigned int weight_loss_num_alloc,
    float* calib,
    unsigned int calib_num_alloc,
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
  PinholeFixedRotationFixedCalibFixedPointResJacKernel<<<n_blocks, 1024>>>(
      rotation,
      rotation_num_alloc,
      translation,
      translation_num_alloc,
      translation_indices,
      pixel,
      pixel_num_alloc,
      weight_loss,
      weight_loss_num_alloc,
      calib,
      calib_num_alloc,
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