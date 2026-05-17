#include "kernel_pinhole_fixed_rotation_fixed_calib_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeFixedRotationFixedCalibFixedPointResJacFirstKernel(
        double* rotation,
        unsigned int rotation_num_alloc,
        double* translation,
        unsigned int translation_num_alloc,
        SharedIndex* translation_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* weight_loss,
        unsigned int weight_loss_num_alloc,
        double* calib,
        unsigned int calib_num_alloc,
        double* point,
        unsigned int point_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* const out_rTr,
        double* const out_translation_njtr,
        unsigned int out_translation_njtr_num_alloc,
        double* const out_translation_precond_diag,
        unsigned int out_translation_precond_diag_num_alloc,
        double* const out_translation_precond_tril,
        unsigned int out_translation_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex translation_indices_loc[1024];
  translation_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? translation_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 0 * weight_loss_num_alloc, global_thread_idx, r0, r1);
    ReadIdx2<1024, double, double, double2>(
        calib, 2 * calib_num_alloc, global_thread_idx, r2, r3);
    ReadIdx2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r4, r5);
    r6 = -1.00000000000000000e+00;
    r5 = fma(r5, r6, r3);
  };
  LoadShared<2, double, double>(translation,
                                0 * translation_num_alloc,
                                translation_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        translation_indices_loc[threadIdx.x].target,
                        r3,
                        r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        point, 0 * point_num_alloc, global_thread_idx, r8, r9);
    r10 = -2.00000000000000000e+00;
    ReadIdx2<1024, double, double, double2>(
        rotation, 2 * rotation_num_alloc, global_thread_idx, r11, r12);
    r13 = r11 * r11;
    r13 = r10 * r13;
    r14 = 1.00000000000000000e+00;
    ReadIdx2<1024, double, double, double2>(
        rotation, 0 * rotation_num_alloc, global_thread_idx, r15, r16);
    r17 = r15 * r15;
    r17 = fma(r10, r17, r14);
    r18 = r13 + r17;
    r18 = fma(r9, r18, r7);
    ReadIdx1<1024, double, double, double>(
        point, 2 * point_num_alloc, global_thread_idx, r7);
    r19 = 2.00000000000000000e+00;
    r20 = r16 * r19;
    r21 = r11 * r20;
    r22 = r12 * r10;
    r23 = fma(r15, r22, r21);
    r24 = r11 * r12;
    r25 = r15 * r20;
    r24 = fma(r19, r24, r25);
    r18 = fma(r7, r23, r18);
    r18 = fma(r8, r24, r18);
    ReadIdx2<1024, double, double, double2>(
        calib, 0 * calib_num_alloc, global_thread_idx, r24, r23);
    r26 = 1.00000000000000008e-15;
  };
  LoadShared<1, double, double>(translation,
                                2 * translation_num_alloc,
                                translation_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared,
                        translation_indices_loc[threadIdx.x].target,
                        r27);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r28 = r16 * r16;
    r28 = r28 * r10;
    r17 = r28 + r17;
    r17 = fma(r7, r17, r27);
    r27 = r15 * r11;
    r27 = r27 * r19;
    r16 = fma(r16, r22, r27);
    r10 = r15 * r12;
    r10 = fma(r19, r10, r21);
    r17 = fma(r8, r16, r17);
    r17 = fma(r9, r10, r17);
    r10 = copysign(1.0, r17);
    r10 = fma(r26, r10, r17);
    r17 = 1.0 / r10;
    r16 = r23 * r17;
    r5 = fma(r18, r16, r5);
    r4 = fma(r4, r6, r2);
    r13 = r14 + r13;
    r13 = r13 + r28;
    r13 = fma(r8, r13, r3);
    r20 = fma(r12, r20, r27);
    r22 = fma(r11, r22, r25);
    r13 = fma(r7, r20, r13);
    r13 = fma(r9, r22, r13);
    r13 = r24 * r13;
    r4 = fma(r17, r13, r4);
    r22 = fma(r0, r4, r1 * r5);
    ReadIdx1<1024, double, double, double>(
        weight_loss, 6 * weight_loss_num_alloc, global_thread_idx, r9);
    r20 = 0.00000000000000000e+00;
    r9 = fmax(r9, r20);
    r7 = sqrt(r9);
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 2 * weight_loss_num_alloc, global_thread_idx, r25, r27);
    r4 = fma(r25, r4, r27 * r5);
    r5 = r22 * r22;
    r8 = fma(r4, r4, r5);
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 4 * weight_loss_num_alloc, global_thread_idx, r3, r28);
    r2 = 5.00000000000000000e-01;
    r28 = fmax(r28, r26);
    r21 = r28 * r28;
    r29 = r19 * r28;
    r30 = fmax(r26, r8);
    r31 = sqrt(r30);
    r29 = fma(r6, r21, r31 * r29);
    r29 = r8 <= r21 ? r8 : r29;
    r31 = 2.50000000000000000e+00;
    r32 = 1.0 / r21;
    r32 = fma(r8, r32, r14);
    r33 = log(r32);
    r33 = r33 * r21;
    r29 = r3 < r31 ? r33 : r29;
    r33 = 1.50000000000000000e+00;
    r34 = sqrt(r32);
    r34 = r6 + r34;
    r34 = r19 * r34;
    r34 = r34 * r21;
    r29 = r3 < r33 ? r34 : r29;
    r29 = r3 < r2 ? r8 : r29;
    r34 = fmax(r20, r29);
    r35 = 1.0 / r30;
    r35 = r9 * r35;
    r36 = r34 * r35;
    r37 = sqrt(r36);
    r37 = r8 <= r26 ? r7 : r37;
    r7 = r22 * r37;
    r38 = r4 * r37;
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r7, r38);
    r38 = r4 * r4;
    r38 = r38 * r37;
    r7 = r37 * r37;
    r7 = fma(r5, r7, r37 * r38);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r7);
  if (global_thread_idx < problem_size) {
    r34 = r9 * r34;
    r9 = -5.00000000000000000e-01;
    r7 = -1.00000000000000008e-15;
    r7 = r7 + r8;
    r7 = copysign(1.0, r7);
    r7 = r14 + r7;
    r38 = r30 * r30;
    r38 = 1.0 / r38;
    r34 = r34 * r9;
    r34 = r34 * r7;
    r34 = r34 * r38;
    r38 = r0 * r24;
    r38 = r38 * r19;
    r38 = r38 * r22;
    r9 = r25 * r24;
    r9 = r9 * r19;
    r9 = r9 * r4;
    r9 = fma(r17, r9, r17 * r38);
    r38 = r2 * r28;
    r30 = rsqrt(r30);
    r38 = r38 * r7;
    r38 = r38 * r30;
    r30 = r9 * r38;
    r30 = r8 <= r21 ? r9 : r30;
    r7 = 1.0 / r32;
    r5 = r9 * r7;
    r30 = r3 < r31 ? r5 : r30;
    r32 = rsqrt(r32);
    r5 = r9 * r32;
    r30 = r3 < r33 ? r5 : r30;
    r30 = r3 < r2 ? r9 : r30;
    r5 = r2 * r30;
    r29 = copysign(1.0, r29);
    r29 = r14 + r29;
    r35 = r29 * r35;
    r5 = fma(r35, r5, r9 * r34);
    r5 = r2 * r5;
    r36 = rsqrt(r36);
    r5 = r5 * r36;
    r5 = r8 <= r26 ? r20 : r5;
    r9 = r6 * r22;
    r29 = 2.50000000000000000e-01;
    r14 = r8 <= r21 ? r20 : r20;
    r14 = r3 < r31 ? r20 : r14;
    r14 = r3 < r33 ? r20 : r14;
    r14 = r3 < r2 ? r20 : r14;
    r14 = r29 * r14;
    r14 = r14 * r36;
    r14 = r14 * r35;
    r14 = r8 <= r26 ? r20 : r14;
    r9 = r9 * r14;
    r29 = fma(r22, r5, r9);
    r39 = r0 * r24;
    r39 = r39 * r37;
    r29 = fma(r17, r39, r29);
    r39 = r22 * r29;
    r40 = r6 * r37;
    r41 = r4 * r40;
    r42 = r6 * r4;
    r42 = r42 * r14;
    r5 = fma(r4, r5, r42);
    r14 = r25 * r24;
    r14 = r14 * r37;
    r5 = fma(r17, r14, r5);
    r39 = fma(r5, r41, r40 * r39);
    r14 = r1 * r19;
    r14 = r14 * r22;
    r17 = r19 * r4;
    r43 = r27 * r16;
    r17 = fma(r43, r17, r16 * r14);
    r14 = r17 * r38;
    r14 = r8 <= r21 ? r17 : r14;
    r44 = r17 * r7;
    r14 = r3 < r31 ? r44 : r14;
    r44 = r17 * r32;
    r14 = r3 < r33 ? r44 : r14;
    r14 = r3 < r2 ? r17 : r14;
    r44 = r2 * r14;
    r17 = fma(r17, r34, r35 * r44);
    r17 = r2 * r17;
    r17 = r17 * r36;
    r17 = r8 <= r26 ? r20 : r17;
    r44 = fma(r4, r17, r42);
    r44 = fma(r37, r43, r44);
    r17 = fma(r22, r17, r9);
    r43 = r1 * r37;
    r17 = fma(r16, r43, r17);
    r43 = r22 * r17;
    r43 = fma(r40, r43, r44 * r41);
    WriteSum2<double, double>((double*)inout_shared, r39, r43);
  };
  FlushSumShared<2, double>(out_translation_njtr,
                            0 * out_translation_njtr_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = r19 * r4;
    r23 = r23 * r6;
    r10 = r10 * r10;
    r10 = 1.0 / r10;
    r23 = r23 * r18;
    r23 = r23 * r10;
    r18 = r25 * r6;
    r18 = r18 * r10;
    r18 = fma(r13, r18, r27 * r23);
    r27 = r19 * r22;
    r39 = r0 * r6;
    r39 = r39 * r10;
    r39 = fma(r13, r39, r1 * r23);
    r27 = fma(r39, r27, r18 * r43);
    r38 = r27 * r38;
    r38 = r8 <= r21 ? r27 : r38;
    r7 = r27 * r7;
    r38 = r3 < r31 ? r7 : r38;
    r32 = r27 * r32;
    r38 = r3 < r33 ? r32 : r38;
    r38 = r3 < r2 ? r27 : r38;
    r3 = r2 * r38;
    r3 = fma(r35, r3, r27 * r34);
    r3 = r2 * r3;
    r3 = r3 * r36;
    r3 = r8 <= r26 ? r20 : r3;
    r42 = fma(r4, r3, r42);
    r42 = fma(r37, r18, r42);
    r3 = fma(r22, r3, r9);
    r3 = fma(r37, r39, r3);
    r39 = r22 * r3;
    r39 = fma(r40, r39, r42 * r41);
    WriteSum1<double, double>((double*)inout_shared, r39);
  };
  FlushSumShared<1, double>(out_translation_njtr,
                            2 * out_translation_njtr_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = fma(r5, r5, r29 * r29);
    r41 = fma(r17, r17, r44 * r44);
    WriteSum2<double, double>((double*)inout_shared, r39, r41);
  };
  FlushSumShared<2, double>(out_translation_precond_diag,
                            0 * out_translation_precond_diag_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = fma(r3, r3, r42 * r42);
    WriteSum1<double, double>((double*)inout_shared, r41);
  };
  FlushSumShared<1, double>(out_translation_precond_diag,
                            2 * out_translation_precond_diag_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = fma(r29, r17, r5 * r44);
    r5 = fma(r5, r42, r29 * r3);
    WriteSum2<double, double>((double*)inout_shared, r41, r5);
  };
  FlushSumShared<2, double>(out_translation_precond_tril,
                            0 * out_translation_precond_tril_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = fma(r44, r42, r17 * r3);
    WriteSum1<double, double>((double*)inout_shared, r42);
  };
  FlushSumShared<1, double>(out_translation_precond_tril,
                            2 * out_translation_precond_tril_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void PinholeFixedRotationFixedCalibFixedPointResJacFirst(
    double* rotation,
    unsigned int rotation_num_alloc,
    double* translation,
    unsigned int translation_num_alloc,
    SharedIndex* translation_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* weight_loss,
    unsigned int weight_loss_num_alloc,
    double* calib,
    unsigned int calib_num_alloc,
    double* point,
    unsigned int point_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_rTr,
    double* const out_translation_njtr,
    unsigned int out_translation_njtr_num_alloc,
    double* const out_translation_precond_diag,
    unsigned int out_translation_precond_diag_num_alloc,
    double* const out_translation_precond_tril,
    unsigned int out_translation_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeFixedRotationFixedCalibFixedPointResJacFirstKernel<<<n_blocks, 1024>>>(
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
      out_rTr,
      out_translation_njtr,
      out_translation_njtr_num_alloc,
      out_translation_precond_diag,
      out_translation_precond_diag_num_alloc,
      out_translation_precond_tril,
      out_translation_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar