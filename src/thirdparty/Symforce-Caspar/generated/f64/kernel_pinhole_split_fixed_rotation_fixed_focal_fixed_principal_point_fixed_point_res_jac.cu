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
        double* rotation,
        unsigned int rotation_num_alloc,
        double* translation,
        unsigned int translation_num_alloc,
        SharedIndex* translation_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* weight_loss,
        unsigned int weight_loss_num_alloc,
        double* focal,
        unsigned int focal_num_alloc,
        double* principal_point,
        unsigned int principal_point_num_alloc,
        double* point,
        unsigned int point_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
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

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 0 * weight_loss_num_alloc, global_thread_idx, r0, r1);
    ReadIdx2<1024, double, double, double2>(principal_point,
                                            0 * principal_point_num_alloc,
                                            global_thread_idx,
                                            r2,
                                            r3);
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
        focal, 0 * focal_num_alloc, global_thread_idx, r24, r23);
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
    r5 = fma(r4, r4, r22 * r22);
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 4 * weight_loss_num_alloc, global_thread_idx, r8, r3);
    r28 = 5.00000000000000000e-01;
    r3 = fmax(r3, r26);
    r2 = r3 * r3;
    r21 = r19 * r3;
    r29 = fmax(r26, r5);
    r30 = sqrt(r29);
    r21 = fma(r30, r21, r6 * r2);
    r21 = r5 <= r2 ? r5 : r21;
    r30 = 2.50000000000000000e+00;
    r31 = 1.0 / r2;
    r31 = fma(r5, r31, r14);
    r32 = log(r31);
    r32 = r32 * r2;
    r21 = r8 < r30 ? r32 : r21;
    r32 = 1.50000000000000000e+00;
    r33 = sqrt(r31);
    r33 = r6 + r33;
    r33 = r19 * r33;
    r33 = r33 * r2;
    r21 = r8 < r32 ? r33 : r21;
    r21 = r8 < r28 ? r5 : r21;
    r33 = fmax(r20, r21);
    r34 = 1.0 / r29;
    r34 = r9 * r34;
    r35 = r33 * r34;
    r36 = sqrt(r35);
    r36 = r5 <= r26 ? r7 : r36;
    r7 = r22 * r36;
    r37 = r4 * r36;
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r7, r37);
    r37 = r6 * r4;
    r7 = 2.50000000000000000e-01;
    r38 = r5 <= r2 ? r20 : r20;
    r38 = r8 < r30 ? r20 : r38;
    r38 = r8 < r32 ? r20 : r38;
    r38 = r8 < r28 ? r20 : r38;
    r38 = r7 * r38;
    r35 = rsqrt(r35);
    r21 = copysign(1.0, r21);
    r21 = r14 + r21;
    r34 = r21 * r34;
    r38 = r38 * r35;
    r38 = r38 * r34;
    r38 = r5 <= r26 ? r20 : r38;
    r37 = r37 * r38;
    r33 = r9 * r33;
    r9 = -5.00000000000000000e-01;
    r21 = -1.00000000000000008e-15;
    r21 = r21 + r5;
    r21 = copysign(1.0, r21);
    r21 = r14 + r21;
    r14 = r29 * r29;
    r14 = 1.0 / r14;
    r33 = r33 * r9;
    r33 = r33 * r21;
    r33 = r33 * r14;
    r14 = r25 * r24;
    r14 = r14 * r19;
    r14 = r14 * r4;
    r9 = r0 * r24;
    r9 = r9 * r19;
    r9 = r9 * r22;
    r9 = fma(r17, r9, r17 * r14);
    r14 = r28 * r3;
    r29 = rsqrt(r29);
    r14 = r14 * r21;
    r14 = r14 * r29;
    r29 = r9 * r14;
    r29 = r5 <= r2 ? r9 : r29;
    r21 = 1.0 / r31;
    r7 = r9 * r21;
    r29 = r8 < r30 ? r7 : r29;
    r31 = rsqrt(r31);
    r7 = r9 * r31;
    r29 = r8 < r32 ? r7 : r29;
    r29 = r8 < r28 ? r9 : r29;
    r7 = r28 * r29;
    r7 = fma(r34, r7, r9 * r33);
    r7 = r28 * r7;
    r7 = r7 * r35;
    r7 = r5 <= r26 ? r20 : r7;
    r9 = fma(r4, r7, r37);
    r39 = r25 * r24;
    r39 = r39 * r36;
    r9 = fma(r17, r39, r9);
    r39 = r4 * r9;
    r40 = r6 * r36;
    r41 = r22 * r40;
    r42 = r6 * r22;
    r42 = r42 * r38;
    r7 = fma(r22, r7, r42);
    r38 = r0 * r24;
    r38 = r38 * r36;
    r7 = fma(r17, r38, r7);
    r39 = fma(r7, r41, r40 * r39);
    r38 = r27 * r19;
    r38 = r38 * r4;
    r17 = r19 * r22;
    r43 = r1 * r16;
    r17 = fma(r43, r17, r16 * r38);
    r38 = r17 * r14;
    r38 = r5 <= r2 ? r17 : r38;
    r44 = r17 * r21;
    r38 = r8 < r30 ? r44 : r38;
    r44 = r17 * r31;
    r38 = r8 < r32 ? r44 : r38;
    r38 = r8 < r28 ? r17 : r38;
    r44 = r28 * r38;
    r44 = fma(r34, r44, r17 * r33);
    r44 = r28 * r44;
    r44 = r44 * r35;
    r44 = r5 <= r26 ? r20 : r44;
    r17 = fma(r22, r44, r42);
    r17 = fma(r36, r43, r17);
    r44 = fma(r4, r44, r37);
    r43 = r27 * r36;
    r44 = fma(r16, r43, r44);
    r43 = r4 * r44;
    r43 = fma(r40, r43, r17 * r41);
    WriteSum2<double, double>((double*)inout_shared, r39, r43);
  };
  FlushSumShared<2, double>(out_translation_njtr,
                            0 * out_translation_njtr_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = r19 * r22;
    r23 = r23 * r6;
    r10 = r10 * r10;
    r10 = 1.0 / r10;
    r23 = r23 * r18;
    r23 = r23 * r10;
    r18 = r0 * r6;
    r18 = r18 * r10;
    r18 = fma(r13, r18, r1 * r23);
    r1 = r19 * r4;
    r39 = r25 * r6;
    r39 = r39 * r10;
    r39 = fma(r13, r39, r27 * r23);
    r1 = fma(r39, r1, r18 * r43);
    r14 = r1 * r14;
    r14 = r5 <= r2 ? r1 : r14;
    r21 = r1 * r21;
    r14 = r8 < r30 ? r21 : r14;
    r31 = r1 * r31;
    r14 = r8 < r32 ? r31 : r14;
    r14 = r8 < r28 ? r1 : r14;
    r8 = r28 * r14;
    r8 = fma(r34, r8, r1 * r33);
    r8 = r28 * r8;
    r8 = r8 * r35;
    r8 = r5 <= r26 ? r20 : r8;
    r42 = fma(r22, r8, r42);
    r42 = fma(r36, r18, r42);
    r39 = fma(r36, r39, r37);
    r39 = fma(r4, r8, r39);
    r8 = r4 * r39;
    r8 = fma(r40, r8, r42 * r41);
    WriteSum1<double, double>((double*)inout_shared, r8);
  };
  FlushSumShared<1, double>(out_translation_njtr,
                            2 * out_translation_njtr_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r8 = fma(r9, r9, r7 * r7);
    r41 = fma(r44, r44, r17 * r17);
    WriteSum2<double, double>((double*)inout_shared, r8, r41);
  };
  FlushSumShared<2, double>(out_translation_precond_diag,
                            0 * out_translation_precond_diag_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = fma(r42, r42, r39 * r39);
    WriteSum1<double, double>((double*)inout_shared, r41);
  };
  FlushSumShared<1, double>(out_translation_precond_diag,
                            2 * out_translation_precond_diag_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = fma(r9, r44, r7 * r17);
    r7 = fma(r9, r39, r7 * r42);
    WriteSum2<double, double>((double*)inout_shared, r41, r7);
  };
  FlushSumShared<2, double>(out_translation_precond_tril,
                            0 * out_translation_precond_tril_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = fma(r44, r39, r17 * r42);
    WriteSum1<double, double>((double*)inout_shared, r42);
  };
  FlushSumShared<1, double>(out_translation_precond_tril,
                            2 * out_translation_precond_tril_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
}

void PinholeSplitFixedRotationFixedFocalFixedPrincipalPointFixedPointResJac(
    double* rotation,
    unsigned int rotation_num_alloc,
    double* translation,
    unsigned int translation_num_alloc,
    SharedIndex* translation_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* weight_loss,
    unsigned int weight_loss_num_alloc,
    double* focal,
    unsigned int focal_num_alloc,
    double* principal_point,
    unsigned int principal_point_num_alloc,
    double* point,
    unsigned int point_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
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