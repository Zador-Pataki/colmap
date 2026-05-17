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
        double* rotation,
        unsigned int rotation_num_alloc,
        double* translation,
        unsigned int translation_num_alloc,
        SharedIndex* translation_indices,
        double* principal_point,
        unsigned int principal_point_num_alloc,
        SharedIndex* principal_point_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* weight_loss,
        unsigned int weight_loss_num_alloc,
        double* focal,
        unsigned int focal_num_alloc,
        double* point,
        unsigned int point_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* const out_rTr,
        double* out_translation_jac,
        unsigned int out_translation_jac_num_alloc,
        double* const out_translation_njtr,
        unsigned int out_translation_njtr_num_alloc,
        double* const out_translation_precond_diag,
        unsigned int out_translation_precond_diag_num_alloc,
        double* const out_translation_precond_tril,
        unsigned int out_translation_precond_tril_num_alloc,
        double* out_principal_point_jac,
        unsigned int out_principal_point_jac_num_alloc,
        double* const out_principal_point_njtr,
        unsigned int out_principal_point_njtr_num_alloc,
        double* const out_principal_point_precond_diag,
        unsigned int out_principal_point_precond_diag_num_alloc,
        double* const out_principal_point_precond_tril,
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

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 0 * weight_loss_num_alloc, global_thread_idx, r0, r1);
  };
  LoadShared<2, double, double>(principal_point,
                                0 * principal_point_num_alloc,
                                principal_point_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        principal_point_indices_loc[threadIdx.x].target,
                        r2,
                        r3);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
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
    r19 = r16 * r11;
    r20 = 2.00000000000000000e+00;
    r19 = r19 * r20;
    r21 = r12 * r10;
    r22 = fma(r15, r21, r19);
    r23 = r11 * r12;
    r24 = r15 * r16;
    r24 = r24 * r20;
    r23 = fma(r20, r23, r24);
    r18 = fma(r7, r22, r18);
    r18 = fma(r8, r23, r18);
    ReadIdx2<1024, double, double, double2>(
        focal, 0 * focal_num_alloc, global_thread_idx, r23, r22);
    r25 = 1.00000000000000008e-15;
  };
  LoadShared<1, double, double>(translation,
                                2 * translation_num_alloc,
                                translation_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared,
                        translation_indices_loc[threadIdx.x].target,
                        r26);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r27 = r16 * r16;
    r27 = r10 * r27;
    r17 = r27 + r17;
    r17 = fma(r7, r17, r26);
    r26 = r15 * r11;
    r26 = r26 * r20;
    r10 = fma(r16, r21, r26);
    r28 = r15 * r12;
    r28 = fma(r20, r28, r19);
    r17 = fma(r8, r10, r17);
    r17 = fma(r9, r28, r17);
    r28 = copysign(1.0, r17);
    r28 = fma(r25, r28, r17);
    r17 = 1.0 / r28;
    r10 = r22 * r17;
    r5 = fma(r18, r10, r5);
    r4 = fma(r4, r6, r2);
    r13 = r14 + r13;
    r13 = r13 + r27;
    r13 = fma(r8, r13, r3);
    r8 = r16 * r12;
    r8 = fma(r20, r8, r26);
    r21 = fma(r11, r21, r24);
    r13 = fma(r7, r8, r13);
    r13 = fma(r9, r21, r13);
    r13 = r23 * r13;
    r4 = fma(r17, r13, r4);
    r21 = fma(r0, r4, r1 * r5);
    ReadIdx1<1024, double, double, double>(
        weight_loss, 6 * weight_loss_num_alloc, global_thread_idx, r9);
    r8 = 0.00000000000000000e+00;
    r9 = fmax(r9, r8);
    r7 = sqrt(r9);
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 2 * weight_loss_num_alloc, global_thread_idx, r24, r26);
    r4 = fma(r24, r4, r26 * r5);
    r5 = fma(r4, r4, r21 * r21);
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 4 * weight_loss_num_alloc, global_thread_idx, r3, r27);
    r2 = 5.00000000000000000e-01;
    r27 = fmax(r27, r25);
    r19 = r27 * r27;
    r29 = r6 * r27;
    r30 = r20 * r27;
    r31 = fmax(r25, r5);
    r32 = sqrt(r31);
    r30 = fma(r32, r30, r27 * r29);
    r30 = r5 <= r19 ? r5 : r30;
    r29 = 2.50000000000000000e+00;
    r32 = r27 * r27;
    r33 = 1.0 / r19;
    r33 = fma(r5, r33, r14);
    r34 = log(r33);
    r32 = r32 * r34;
    r30 = r3 < r29 ? r32 : r30;
    r32 = 1.50000000000000000e+00;
    r34 = r20 * r27;
    r35 = sqrt(r33);
    r35 = r6 + r35;
    r34 = r34 * r27;
    r34 = r34 * r35;
    r30 = r3 < r32 ? r34 : r30;
    r30 = r3 < r2 ? r5 : r30;
    r34 = fmax(r8, r30);
    r35 = 1.0 / r31;
    r35 = r9 * r35;
    r36 = r34 * r35;
    r37 = sqrt(r36);
    r37 = r5 <= r25 ? r7 : r37;
    r7 = r21 * r37;
    r38 = r4 * r37;
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r7, r38);
    r38 = r21 * r21;
    r38 = r38 * r37;
    r7 = r4 * r4;
    r7 = r7 * r37;
    r7 = fma(r37, r7, r37 * r38);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r7);
  if (global_thread_idx < problem_size) {
    r7 = r6 * r21;
    r38 = 2.50000000000000000e-01;
    r39 = r5 <= r19 ? r8 : r8;
    r39 = r3 < r29 ? r8 : r39;
    r39 = r3 < r32 ? r8 : r39;
    r39 = r3 < r2 ? r8 : r39;
    r39 = r38 * r39;
    r36 = rsqrt(r36);
    r30 = copysign(1.0, r30);
    r30 = r14 + r30;
    r35 = r30 * r35;
    r39 = r39 * r36;
    r39 = r39 * r35;
    r39 = r5 <= r25 ? r8 : r39;
    r7 = r7 * r39;
    r34 = r9 * r34;
    r9 = -5.00000000000000000e-01;
    r30 = -1.00000000000000008e-15;
    r30 = r30 + r5;
    r30 = copysign(1.0, r30);
    r30 = r14 + r30;
    r14 = r31 * r31;
    r14 = 1.0 / r14;
    r34 = r34 * r9;
    r34 = r34 * r30;
    r34 = r34 * r14;
    r14 = r23 * r17;
    r9 = r24 * r20;
    r9 = r9 * r4;
    r38 = r0 * r23;
    r38 = r38 * r20;
    r38 = r38 * r21;
    r38 = fma(r17, r38, r9 * r14);
    r14 = r2 * r27;
    r31 = rsqrt(r31);
    r14 = r14 * r30;
    r14 = r14 * r31;
    r31 = r38 * r14;
    r31 = r5 <= r19 ? r38 : r31;
    r30 = 1.0 / r33;
    r40 = r38 * r30;
    r31 = r3 < r29 ? r40 : r31;
    r33 = rsqrt(r33);
    r40 = r38 * r33;
    r31 = r3 < r32 ? r40 : r31;
    r31 = r3 < r2 ? r38 : r31;
    r40 = r2 * r31;
    r40 = fma(r35, r40, r38 * r34);
    r40 = r2 * r40;
    r40 = r40 * r36;
    r40 = r5 <= r25 ? r8 : r40;
    r38 = fma(r21, r40, r7);
    r41 = r0 * r23;
    r41 = r41 * r37;
    r38 = fma(r17, r41, r38);
    r41 = r6 * r4;
    r41 = r41 * r39;
    r40 = fma(r4, r40, r41);
    r39 = r24 * r23;
    r39 = r39 * r37;
    r40 = fma(r17, r39, r40);
    WriteIdx2<1024, double, double, double2>(out_translation_jac,
                                             0 * out_translation_jac_num_alloc,
                                             global_thread_idx,
                                             r38,
                                             r40);
    r39 = r20 * r4;
    r42 = r26 * r10;
    r43 = r1 * r20;
    r43 = r43 * r21;
    r43 = fma(r10, r43, r39 * r42);
    r44 = r43 * r14;
    r44 = r5 <= r19 ? r43 : r44;
    r45 = r43 * r30;
    r44 = r3 < r29 ? r45 : r44;
    r45 = r43 * r33;
    r44 = r3 < r32 ? r45 : r44;
    r44 = r3 < r2 ? r43 : r44;
    r45 = r2 * r44;
    r45 = fma(r35, r45, r43 * r34);
    r45 = r2 * r45;
    r45 = r45 * r36;
    r45 = r5 <= r25 ? r8 : r45;
    r43 = fma(r21, r45, r7);
    r46 = r1 * r37;
    r43 = fma(r10, r46, r43);
    r45 = fma(r4, r45, r41);
    r45 = fma(r37, r42, r45);
    WriteIdx2<1024, double, double, double2>(out_translation_jac,
                                             2 * out_translation_jac_num_alloc,
                                             global_thread_idx,
                                             r43,
                                             r45);
    r42 = r20 * r21;
    r22 = r22 * r6;
    r28 = r28 * r28;
    r28 = 1.0 / r28;
    r22 = r22 * r18;
    r22 = r22 * r28;
    r18 = r0 * r6;
    r18 = r18 * r28;
    r18 = fma(r13, r18, r1 * r22);
    r46 = r24 * r6;
    r46 = r46 * r28;
    r46 = fma(r13, r46, r26 * r22);
    r42 = fma(r46, r39, r18 * r42);
    r22 = r42 * r14;
    r22 = r5 <= r19 ? r42 : r22;
    r13 = r42 * r30;
    r22 = r3 < r29 ? r13 : r22;
    r13 = r42 * r33;
    r22 = r3 < r32 ? r13 : r22;
    r22 = r3 < r2 ? r42 : r22;
    r13 = r2 * r22;
    r13 = fma(r35, r13, r42 * r34);
    r13 = r2 * r13;
    r13 = r13 * r36;
    r13 = r5 <= r25 ? r8 : r13;
    r42 = fma(r21, r13, r7);
    r42 = fma(r37, r18, r42);
    r46 = fma(r37, r46, r41);
    r46 = fma(r4, r13, r46);
    WriteIdx2<1024, double, double, double2>(out_translation_jac,
                                             4 * out_translation_jac_num_alloc,
                                             global_thread_idx,
                                             r42,
                                             r46);
    r13 = r4 * r40;
    r18 = r6 * r37;
    r28 = r21 * r18;
    r13 = fma(r38, r28, r18 * r13);
    r10 = r4 * r45;
    r10 = fma(r18, r10, r43 * r28);
    WriteSum2<double, double>((double*)inout_shared, r13, r10);
  };
  FlushSumShared<2, double>(out_translation_njtr,
                            0 * out_translation_njtr_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r10 = r4 * r46;
    r10 = fma(r18, r10, r42 * r28);
    WriteSum1<double, double>((double*)inout_shared, r10);
  };
  FlushSumShared<1, double>(out_translation_njtr,
                            2 * out_translation_njtr_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r10 = fma(r40, r40, r38 * r38);
    r13 = fma(r45, r45, r43 * r43);
    WriteSum2<double, double>((double*)inout_shared, r10, r13);
  };
  FlushSumShared<2, double>(out_translation_precond_diag,
                            0 * out_translation_precond_diag_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = fma(r42, r42, r46 * r46);
    WriteSum1<double, double>((double*)inout_shared, r13);
  };
  FlushSumShared<1, double>(out_translation_precond_diag,
                            2 * out_translation_precond_diag_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = fma(r40, r45, r38 * r43);
    r38 = fma(r40, r46, r38 * r42);
    WriteSum2<double, double>((double*)inout_shared, r13, r38);
  };
  FlushSumShared<2, double>(out_translation_precond_tril,
                            0 * out_translation_precond_tril_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = fma(r45, r46, r43 * r42);
    WriteSum1<double, double>((double*)inout_shared, r42);
  };
  FlushSumShared<1, double>(out_translation_precond_tril,
                            2 * out_translation_precond_tril_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = fma(r0, r37, r7);
    r43 = r0 * r20;
    r43 = fma(r21, r43, r9);
    r9 = r43 * r14;
    r9 = r5 <= r19 ? r43 : r9;
    r38 = r43 * r30;
    r9 = r3 < r29 ? r38 : r9;
    r38 = r43 * r33;
    r9 = r3 < r32 ? r38 : r9;
    r9 = r3 < r2 ? r43 : r9;
    r38 = r2 * r9;
    r43 = fma(r43, r34, r35 * r38);
    r43 = r2 * r43;
    r43 = r43 * r36;
    r43 = r5 <= r25 ? r8 : r43;
    r42 = fma(r21, r43, r42);
    r38 = fma(r24, r37, r41);
    r38 = fma(r4, r43, r38);
    WriteIdx2<1024, double, double, double2>(
        out_principal_point_jac,
        0 * out_principal_point_jac_num_alloc,
        global_thread_idx,
        r42,
        r38);
    r7 = fma(r1, r37, r7);
    r43 = r1 * r20;
    r39 = fma(r26, r39, r21 * r43);
    r14 = r39 * r14;
    r14 = r5 <= r19 ? r39 : r14;
    r30 = r39 * r30;
    r14 = r3 < r29 ? r30 : r14;
    r33 = r39 * r33;
    r14 = r3 < r32 ? r33 : r14;
    r14 = r3 < r2 ? r39 : r14;
    r3 = r2 * r14;
    r3 = fma(r35, r3, r39 * r34);
    r3 = r2 * r3;
    r3 = r3 * r36;
    r3 = r5 <= r25 ? r8 : r3;
    r7 = fma(r21, r3, r7);
    r26 = fma(r26, r37, r41);
    r26 = fma(r4, r3, r26);
    WriteIdx2<1024, double, double, double2>(
        out_principal_point_jac,
        2 * out_principal_point_jac_num_alloc,
        global_thread_idx,
        r7,
        r26);
    r3 = r4 * r38;
    r3 = fma(r18, r3, r42 * r28);
    r41 = r4 * r26;
    r28 = fma(r7, r28, r18 * r41);
    WriteSum2<double, double>((double*)inout_shared, r3, r28);
  };
  FlushSumShared<2, double>(out_principal_point_njtr,
                            0 * out_principal_point_njtr_num_alloc,
                            principal_point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r28 = fma(r42, r42, r38 * r38);
    r3 = fma(r26, r26, r7 * r7);
    WriteSum2<double, double>((double*)inout_shared, r28, r3);
  };
  FlushSumShared<2, double>(out_principal_point_precond_diag,
                            0 * out_principal_point_precond_diag_num_alloc,
                            principal_point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = fma(r42, r7, r38 * r26);
    WriteSum1<double, double>((double*)inout_shared, r7);
  };
  FlushSumShared<1, double>(out_principal_point_precond_tril,
                            0 * out_principal_point_precond_tril_num_alloc,
                            principal_point_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void PinholeSplitFixedRotationFixedFocalFixedPointResJacFirst(
    double* rotation,
    unsigned int rotation_num_alloc,
    double* translation,
    unsigned int translation_num_alloc,
    SharedIndex* translation_indices,
    double* principal_point,
    unsigned int principal_point_num_alloc,
    SharedIndex* principal_point_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* weight_loss,
    unsigned int weight_loss_num_alloc,
    double* focal,
    unsigned int focal_num_alloc,
    double* point,
    unsigned int point_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_rTr,
    double* out_translation_jac,
    unsigned int out_translation_jac_num_alloc,
    double* const out_translation_njtr,
    unsigned int out_translation_njtr_num_alloc,
    double* const out_translation_precond_diag,
    unsigned int out_translation_precond_diag_num_alloc,
    double* const out_translation_precond_tril,
    unsigned int out_translation_precond_tril_num_alloc,
    double* out_principal_point_jac,
    unsigned int out_principal_point_jac_num_alloc,
    double* const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    double* const out_principal_point_precond_diag,
    unsigned int out_principal_point_precond_diag_num_alloc,
    double* const out_principal_point_precond_tril,
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