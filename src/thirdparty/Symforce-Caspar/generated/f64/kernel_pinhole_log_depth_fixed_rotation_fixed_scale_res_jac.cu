#include "kernel_pinhole_log_depth_fixed_rotation_fixed_scale_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeLogDepthFixedRotationFixedScaleResJacKernel(
        double* rotation,
        unsigned int rotation_num_alloc,
        double* translation,
        unsigned int translation_num_alloc,
        SharedIndex* translation_indices,
        double* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        double* log_depth,
        unsigned int log_depth_num_alloc,
        double* loss,
        unsigned int loss_num_alloc,
        double* scale,
        unsigned int scale_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* out_translation_jac,
        unsigned int out_translation_jac_num_alloc,
        double* const out_translation_njtr,
        unsigned int out_translation_njtr_num_alloc,
        double* const out_translation_precond_diag,
        unsigned int out_translation_precond_diag_num_alloc,
        double* const out_translation_precond_tril,
        unsigned int out_translation_precond_tril_num_alloc,
        double* out_point_jac,
        unsigned int out_point_jac_num_alloc,
        double* const out_point_njtr,
        unsigned int out_point_njtr_num_alloc,
        double* const out_point_precond_diag,
        unsigned int out_point_precond_diag_num_alloc,
        double* const out_point_precond_tril,
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

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35;

  if (global_thread_idx < problem_size) {
    r0 = 0.00000000000000000e+00;
  };
  LoadShared<1, double, double>(translation,
                                2 * translation_num_alloc,
                                translation_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, translation_indices_loc[threadIdx.x].target, r1);
  };
  __syncthreads();
  LoadShared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r3 = 1.00000000000000000e+00;
    r4 = -2.00000000000000000e+00;
    ReadIdx2<1024, double, double, double2>(
        rotation, 0 * rotation_num_alloc, global_thread_idx, r5, r6);
    r7 = r6 * r6;
    r7 = fma(r4, r7, r3);
    r8 = r5 * r5;
    r7 = fma(r4, r8, r7);
    r2 = fma(r2, r7, r1);
  };
  LoadShared<2, double, double>(
      point, 0 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r1, r8);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        rotation, 2 * rotation_num_alloc, global_thread_idx, r9, r10);
    r11 = r5 * r9;
    r12 = 2.00000000000000000e+00;
    r13 = r6 * r10;
    r13 = fma(r4, r13, r12 * r11);
    r11 = r5 * r10;
    r4 = r6 * r9;
    r4 = fma(r12, r4, r12 * r11);
    r2 = fma(r1, r13, r2);
    r2 = fma(r8, r4, r2);
    r8 = 1.00000000000000008e-15;
    r1 = fmax(r8, r2);
    r11 = log(r1);
    ReadIdx1<1024, double, double, double>(
        scale, 0 * scale_num_alloc, global_thread_idx, r14);
    r15 = -1.00000000000000000e+00;
    r14 = fma(r14, r15, r11);
    ReadIdx1<1024, double, double, double>(
        log_depth, 0 * log_depth_num_alloc, global_thread_idx, r11);
    r14 = fma(r11, r15, r14);
    r14 = r0 < r2 ? r14 : r0;
    ReadIdx1<1024, double, double, double>(
        loss, 2 * loss_num_alloc, global_thread_idx, r11);
    r11 = fmax(r11, r0);
    r16 = sqrt(r11);
    r17 = r14 * r14;
    r18 = fmax(r8, r17);
    r19 = 1.0 / r18;
    ReadIdx2<1024, double, double, double2>(
        loss, 0 * loss_num_alloc, global_thread_idx, r20, r21);
    r22 = 5.00000000000000000e-01;
    r21 = fmax(r21, r8);
    r23 = r21 * r21;
    r24 = r15 * r21;
    r25 = r12 * r21;
    r26 = sqrt(r18);
    r25 = fma(r26, r25, r21 * r24);
    r25 = r17 <= r23 ? r17 : r25;
    r24 = 2.50000000000000000e+00;
    r26 = r21 * r21;
    r27 = r14 * r14;
    r28 = 1.0 / r23;
    r27 = fma(r28, r27, r3);
    r28 = log(r27);
    r26 = r26 * r28;
    r25 = r20 < r24 ? r26 : r25;
    r26 = 1.50000000000000000e+00;
    r28 = r12 * r21;
    r29 = sqrt(r27);
    r29 = r15 + r29;
    r28 = r28 * r21;
    r28 = r28 * r29;
    r25 = r20 < r26 ? r28 : r25;
    r25 = r20 < r22 ? r17 : r25;
    r28 = fmax(r0, r25);
    r28 = r11 * r28;
    r29 = r19 * r28;
    r30 = sqrt(r29);
    r30 = r17 <= r8 ? r16 : r30;
    r16 = r14 * r30;
    WriteIdx1<1024, double, double, double>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r16);
    r16 = -1.00000000000000008e-15;
    r31 = r16 + r2;
    r31 = copysign(1.0, r31);
    r31 = r3 + r31;
    r31 = r22 * r31;
    r1 = 1.0 / r1;
    r31 = r31 * r1;
    r1 = r0 < r2 ? r31 : r0;
    r32 = r15 * r14;
    r16 = r16 + r17;
    r16 = copysign(1.0, r16);
    r16 = r3 + r16;
    r33 = r18 * r18;
    r33 = 1.0 / r33;
    r32 = r32 * r16;
    r32 = r32 * r33;
    r32 = r32 * r28;
    r28 = r12 * r14;
    r33 = r1 * r28;
    r34 = r14 * r21;
    r18 = rsqrt(r18);
    r34 = r34 * r16;
    r34 = r34 * r18;
    r18 = r1 * r34;
    r18 = r17 <= r23 ? r33 : r18;
    r16 = 1.0 / r27;
    r35 = r1 * r16;
    r35 = r35 * r28;
    r18 = r20 < r24 ? r35 : r18;
    r27 = rsqrt(r27);
    r27 = r27 * r28;
    r35 = r1 * r27;
    r18 = r20 < r26 ? r35 : r18;
    r18 = r20 < r22 ? r33 : r18;
    r11 = r11 * r22;
    r25 = copysign(1.0, r25);
    r25 = r3 + r25;
    r11 = r11 * r25;
    r11 = r11 * r19;
    r18 = fma(r18, r11, r1 * r32);
    r18 = r22 * r18;
    r29 = rsqrt(r29);
    r18 = r18 * r29;
    r18 = r17 <= r8 ? r0 : r18;
    r1 = fma(r30, r1, r14 * r18);
    r18 = r15 * r14;
    r19 = r0 < r2 ? r0 : r0;
    r25 = r19 * r28;
    r3 = r19 * r34;
    r3 = r17 <= r23 ? r25 : r3;
    r33 = r19 * r16;
    r33 = r33 * r28;
    r3 = r20 < r24 ? r33 : r3;
    r33 = r19 * r27;
    r3 = r20 < r26 ? r33 : r3;
    r3 = r20 < r22 ? r25 : r3;
    r3 = fma(r19, r32, r3 * r11);
    r3 = r22 * r3;
    r3 = r3 * r29;
    r3 = r17 <= r8 ? r0 : r3;
    r25 = r15 * r30;
    r25 = fma(r19, r25, r3 * r18);
    r1 = r1 + r25;
    WriteIdx1<1024, double, double, double>(out_translation_jac,
                                            0 * out_translation_jac_num_alloc,
                                            global_thread_idx,
                                            r1);
    r18 = r15 * r14;
    r18 = r18 * r30;
    r18 = r18 * r1;
    WriteSum1<double, double>((double*)inout_shared, r18);
  };
  FlushSumShared<1, double>(out_translation_njtr,
                            2 * out_translation_njtr_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r1 * r1;
    WriteSum1<double, double>((double*)inout_shared, r1);
  };
  FlushSumShared<1, double>(out_translation_precond_diag,
                            2 * out_translation_precond_diag_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = r13 * r31;
    r13 = r0 < r2 ? r13 : r0;
    r1 = r13 * r28;
    r18 = r13 * r34;
    r18 = r17 <= r23 ? r1 : r18;
    r19 = r13 * r16;
    r19 = r19 * r28;
    r18 = r20 < r24 ? r19 : r18;
    r19 = r13 * r27;
    r18 = r20 < r26 ? r19 : r18;
    r18 = r20 < r22 ? r1 : r18;
    r18 = fma(r18, r11, r13 * r32);
    r18 = r22 * r18;
    r18 = r18 * r29;
    r18 = r17 <= r8 ? r0 : r18;
    r18 = fma(r14, r18, r30 * r13);
    r18 = r18 + r25;
    r4 = r4 * r31;
    r4 = r0 < r2 ? r4 : r0;
    r13 = r4 * r28;
    r1 = r4 * r34;
    r1 = r17 <= r23 ? r13 : r1;
    r19 = r4 * r16;
    r19 = r19 * r28;
    r1 = r20 < r24 ? r19 : r1;
    r19 = r4 * r27;
    r1 = r20 < r26 ? r19 : r1;
    r1 = r20 < r22 ? r13 : r1;
    r1 = fma(r4, r32, r1 * r11);
    r1 = r22 * r1;
    r1 = r1 * r29;
    r1 = r17 <= r8 ? r0 : r1;
    r4 = fma(r30, r4, r14 * r1);
    r4 = r4 + r25;
    WriteIdx2<1024, double, double, double2>(
        out_point_jac, 0 * out_point_jac_num_alloc, global_thread_idx, r18, r4);
    r31 = r7 * r31;
    r31 = r0 < r2 ? r31 : r0;
    r2 = r31 * r28;
    r34 = r31 * r34;
    r34 = r17 <= r23 ? r2 : r34;
    r16 = r31 * r16;
    r16 = r16 * r28;
    r34 = r20 < r24 ? r16 : r34;
    r27 = r31 * r27;
    r34 = r20 < r26 ? r27 : r34;
    r34 = r20 < r22 ? r2 : r34;
    r32 = fma(r31, r32, r34 * r11);
    r32 = r22 * r32;
    r32 = r32 * r29;
    r32 = r17 <= r8 ? r0 : r32;
    r32 = fma(r14, r32, r30 * r31);
    r32 = r32 + r25;
    WriteIdx1<1024, double, double, double>(
        out_point_jac, 2 * out_point_jac_num_alloc, global_thread_idx, r32);
    r25 = r15 * r14;
    r25 = r25 * r30;
    r25 = r25 * r18;
    r31 = r15 * r14;
    r31 = r31 * r30;
    r31 = r31 * r4;
    WriteSum2<double, double>((double*)inout_shared, r25, r31);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = r15 * r14;
    r31 = r31 * r30;
    r31 = r31 * r32;
    WriteSum1<double, double>((double*)inout_shared, r31);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = r18 * r18;
    r25 = r4 * r4;
    WriteSum2<double, double>((double*)inout_shared, r31, r25);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = r32 * r32;
    WriteSum1<double, double>((double*)inout_shared, r25);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = r18 * r4;
    r18 = r18 * r32;
    WriteSum2<double, double>((double*)inout_shared, r25, r18);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = r4 * r32;
    WriteSum1<double, double>((double*)inout_shared, r32);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
}

void PinholeLogDepthFixedRotationFixedScaleResJac(
    double* rotation,
    unsigned int rotation_num_alloc,
    double* translation,
    unsigned int translation_num_alloc,
    SharedIndex* translation_indices,
    double* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    double* log_depth,
    unsigned int log_depth_num_alloc,
    double* loss,
    unsigned int loss_num_alloc,
    double* scale,
    unsigned int scale_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* out_translation_jac,
    unsigned int out_translation_jac_num_alloc,
    double* const out_translation_njtr,
    unsigned int out_translation_njtr_num_alloc,
    double* const out_translation_precond_diag,
    unsigned int out_translation_precond_diag_num_alloc,
    double* const out_translation_precond_tril,
    unsigned int out_translation_precond_tril_num_alloc,
    double* out_point_jac,
    unsigned int out_point_jac_num_alloc,
    double* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    double* const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    double* const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeLogDepthFixedRotationFixedScaleResJacKernel<<<n_blocks, 1024>>>(
      rotation,
      rotation_num_alloc,
      translation,
      translation_num_alloc,
      translation_indices,
      point,
      point_num_alloc,
      point_indices,
      log_depth,
      log_depth_num_alloc,
      loss,
      loss_num_alloc,
      scale,
      scale_num_alloc,
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