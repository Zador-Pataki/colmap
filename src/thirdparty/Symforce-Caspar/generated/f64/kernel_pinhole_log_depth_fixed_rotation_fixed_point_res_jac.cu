#include "kernel_pinhole_log_depth_fixed_rotation_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeLogDepthFixedRotationFixedPointResJacKernel(
        double* rotation,
        unsigned int rotation_num_alloc,
        double* translation,
        unsigned int translation_num_alloc,
        SharedIndex* translation_indices,
        double* scale,
        unsigned int scale_num_alloc,
        SharedIndex* scale_indices,
        double* log_depth,
        unsigned int log_depth_num_alloc,
        double* loss,
        unsigned int loss_num_alloc,
        double* point,
        unsigned int point_num_alloc,
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
        double* out_scale_jac,
        unsigned int out_scale_jac_num_alloc,
        double* const out_scale_njtr,
        unsigned int out_scale_njtr_num_alloc,
        double* const out_scale_precond_diag,
        unsigned int out_scale_precond_diag_num_alloc,
        double* const out_scale_precond_tril,
        unsigned int out_scale_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex translation_indices_loc[1024];
  translation_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? translation_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex scale_indices_loc[1024];
  scale_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? scale_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32;

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
  if (global_thread_idx < problem_size) {
    ReadIdx1<1024, double, double, double>(
        point, 2 * point_num_alloc, global_thread_idx, r2);
    r3 = 1.00000000000000000e+00;
    r4 = -2.00000000000000000e+00;
    ReadIdx2<1024, double, double, double2>(
        rotation, 0 * rotation_num_alloc, global_thread_idx, r5, r6);
    r7 = r6 * r6;
    r7 = fma(r4, r7, r3);
    r8 = r5 * r5;
    r7 = fma(r4, r8, r7);
    r7 = fma(r2, r7, r1);
    ReadIdx2<1024, double, double, double2>(
        point, 0 * point_num_alloc, global_thread_idx, r2, r1);
    ReadIdx2<1024, double, double, double2>(
        rotation, 2 * rotation_num_alloc, global_thread_idx, r8, r9);
    r10 = r5 * r8;
    r11 = 2.00000000000000000e+00;
    r12 = r6 * r9;
    r12 = fma(r4, r12, r11 * r10);
    r10 = r5 * r9;
    r4 = r6 * r8;
    r4 = fma(r11, r4, r11 * r10);
    r7 = fma(r2, r12, r7);
    r7 = fma(r1, r4, r7);
    r4 = 1.00000000000000008e-15;
    r1 = fmax(r4, r7);
    r12 = log(r1);
  };
  LoadShared<1, double, double>(
      scale, 0 * scale_num_alloc, scale_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, scale_indices_loc[threadIdx.x].target, r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r10 = -1.00000000000000000e+00;
    r2 = fma(r2, r10, r12);
    ReadIdx1<1024, double, double, double>(
        log_depth, 0 * log_depth_num_alloc, global_thread_idx, r12);
    r2 = fma(r12, r10, r2);
    r2 = r0 < r7 ? r2 : r0;
    ReadIdx1<1024, double, double, double>(
        loss, 2 * loss_num_alloc, global_thread_idx, r12);
    r12 = fmax(r12, r0);
    r13 = sqrt(r12);
    r14 = r2 * r2;
    r15 = fmax(r4, r14);
    r16 = 1.0 / r15;
    ReadIdx2<1024, double, double, double2>(
        loss, 0 * loss_num_alloc, global_thread_idx, r17, r18);
    r19 = 5.00000000000000000e-01;
    r18 = fmax(r18, r4);
    r20 = r18 * r18;
    r21 = r11 * r18;
    r22 = sqrt(r15);
    r21 = fma(r22, r21, r10 * r20);
    r21 = r14 <= r20 ? r14 : r21;
    r22 = 2.50000000000000000e+00;
    r23 = r2 * r2;
    r24 = 1.0 / r20;
    r23 = fma(r24, r23, r3);
    r24 = log(r23);
    r24 = r24 * r20;
    r21 = r17 < r22 ? r24 : r21;
    r24 = 1.50000000000000000e+00;
    r25 = sqrt(r23);
    r25 = r10 + r25;
    r25 = r11 * r25;
    r25 = r25 * r20;
    r21 = r17 < r24 ? r25 : r21;
    r21 = r17 < r19 ? r14 : r21;
    r25 = fmax(r0, r21);
    r25 = r12 * r25;
    r26 = r16 * r25;
    r27 = sqrt(r26);
    r27 = r14 <= r4 ? r13 : r27;
    r13 = r2 * r27;
    WriteIdx1<1024, double, double, double>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r13);
    r13 = -1.00000000000000008e-15;
    r28 = r13 + r7;
    r28 = copysign(1.0, r28);
    r28 = r3 + r28;
    r28 = r19 * r28;
    r1 = 1.0 / r1;
    r28 = r28 * r1;
    r28 = r0 < r7 ? r28 : r0;
    r1 = r10 * r2;
    r13 = r13 + r14;
    r13 = copysign(1.0, r13);
    r13 = r3 + r13;
    r29 = r15 * r15;
    r29 = 1.0 / r29;
    r1 = r1 * r13;
    r1 = r1 * r29;
    r1 = r1 * r25;
    r25 = r11 * r2;
    r29 = r28 * r25;
    r30 = r2 * r18;
    r15 = rsqrt(r15);
    r30 = r30 * r13;
    r30 = r30 * r28;
    r30 = r30 * r15;
    r30 = r14 <= r20 ? r29 : r30;
    r31 = 1.0 / r23;
    r32 = r28 * r31;
    r32 = r32 * r25;
    r30 = r17 < r22 ? r32 : r30;
    r23 = rsqrt(r23);
    r32 = r28 * r23;
    r32 = r32 * r25;
    r30 = r17 < r24 ? r32 : r30;
    r30 = r17 < r19 ? r29 : r30;
    r12 = r12 * r19;
    r21 = copysign(1.0, r21);
    r21 = r3 + r21;
    r12 = r12 * r21;
    r12 = r12 * r16;
    r30 = fma(r30, r12, r28 * r1);
    r30 = r19 * r30;
    r26 = rsqrt(r26);
    r30 = r30 * r26;
    r30 = r14 <= r4 ? r0 : r30;
    r28 = fma(r27, r28, r2 * r30);
    r30 = r10 * r2;
    r16 = r11 * r2;
    r21 = r0 < r7 ? r0 : r0;
    r16 = r16 * r21;
    r3 = r2 * r18;
    r3 = r3 * r13;
    r3 = r3 * r21;
    r3 = r3 * r15;
    r3 = r14 <= r20 ? r16 : r3;
    r29 = r31 * r16;
    r3 = r17 < r22 ? r29 : r3;
    r29 = r23 * r16;
    r3 = r17 < r24 ? r29 : r3;
    r3 = r17 < r19 ? r16 : r3;
    r3 = fma(r21, r1, r3 * r12);
    r3 = r19 * r3;
    r3 = r3 * r26;
    r3 = r14 <= r4 ? r0 : r3;
    r16 = r10 * r27;
    r16 = fma(r21, r16, r3 * r30);
    r28 = r28 + r16;
    WriteIdx1<1024, double, double, double>(out_translation_jac,
                                            0 * out_translation_jac_num_alloc,
                                            global_thread_idx,
                                            r28);
    r30 = r10 * r2;
    r30 = r30 * r27;
    r30 = r30 * r28;
    WriteSum1<double, double>((double*)inout_shared, r30);
  };
  FlushSumShared<1, double>(out_translation_njtr,
                            2 * out_translation_njtr_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r28 = r28 * r28;
    WriteSum1<double, double>((double*)inout_shared, r28);
  };
  FlushSumShared<1, double>(out_translation_precond_diag,
                            2 * out_translation_precond_diag_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = r0 < r7 ? r10 : r0;
    r28 = r7 * r25;
    r30 = r2 * r18;
    r30 = r30 * r13;
    r30 = r30 * r7;
    r30 = r30 * r15;
    r30 = r14 <= r20 ? r28 : r30;
    r31 = r7 * r31;
    r31 = r31 * r25;
    r30 = r17 < r22 ? r31 : r30;
    r23 = r7 * r23;
    r23 = r23 * r25;
    r30 = r17 < r24 ? r23 : r30;
    r30 = r17 < r19 ? r28 : r30;
    r12 = fma(r30, r12, r7 * r1);
    r12 = r19 * r12;
    r12 = r12 * r26;
    r12 = r14 <= r4 ? r0 : r12;
    r7 = fma(r27, r7, r2 * r12);
    r7 = r7 + r16;
    WriteIdx1<1024, double, double, double>(
        out_scale_jac, 0 * out_scale_jac_num_alloc, global_thread_idx, r7);
    r16 = r10 * r2;
    r16 = r16 * r27;
    r16 = r16 * r7;
    WriteSum1<double, double>((double*)inout_shared, r16);
  };
  FlushSumShared<1, double>(out_scale_njtr,
                            0 * out_scale_njtr_num_alloc,
                            scale_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = r7 * r7;
    WriteSum1<double, double>((double*)inout_shared, r7);
  };
  FlushSumShared<1, double>(out_scale_precond_diag,
                            0 * out_scale_precond_diag_num_alloc,
                            scale_indices_loc,
                            (double*)inout_shared);
}

void PinholeLogDepthFixedRotationFixedPointResJac(
    double* rotation,
    unsigned int rotation_num_alloc,
    double* translation,
    unsigned int translation_num_alloc,
    SharedIndex* translation_indices,
    double* scale,
    unsigned int scale_num_alloc,
    SharedIndex* scale_indices,
    double* log_depth,
    unsigned int log_depth_num_alloc,
    double* loss,
    unsigned int loss_num_alloc,
    double* point,
    unsigned int point_num_alloc,
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
    double* out_scale_jac,
    unsigned int out_scale_jac_num_alloc,
    double* const out_scale_njtr,
    unsigned int out_scale_njtr_num_alloc,
    double* const out_scale_precond_diag,
    unsigned int out_scale_precond_diag_num_alloc,
    double* const out_scale_precond_tril,
    unsigned int out_scale_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeLogDepthFixedRotationFixedPointResJacKernel<<<n_blocks, 1024>>>(
      rotation,
      rotation_num_alloc,
      translation,
      translation_num_alloc,
      translation_indices,
      scale,
      scale_num_alloc,
      scale_indices,
      log_depth,
      log_depth_num_alloc,
      loss,
      loss_num_alloc,
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
      out_scale_jac,
      out_scale_jac_num_alloc,
      out_scale_njtr,
      out_scale_njtr_num_alloc,
      out_scale_precond_diag,
      out_scale_precond_diag_num_alloc,
      out_scale_precond_tril,
      out_scale_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar