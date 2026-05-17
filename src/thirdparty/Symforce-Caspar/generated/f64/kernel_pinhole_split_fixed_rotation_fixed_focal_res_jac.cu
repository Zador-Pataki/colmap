#include "kernel_pinhole_split_fixed_rotation_fixed_focal_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedRotationFixedFocalResJacKernel(
        double* rotation,
        unsigned int rotation_num_alloc,
        double* translation,
        unsigned int translation_num_alloc,
        SharedIndex* translation_indices,
        double* principal_point,
        unsigned int principal_point_num_alloc,
        SharedIndex* principal_point_indices,
        double* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* weight_loss,
        unsigned int weight_loss_num_alloc,
        double* focal,
        unsigned int focal_num_alloc,
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
        double* out_principal_point_jac,
        unsigned int out_principal_point_jac_num_alloc,
        double* const out_principal_point_njtr,
        unsigned int out_principal_point_njtr_num_alloc,
        double* const out_principal_point_precond_diag,
        unsigned int out_principal_point_precond_diag_num_alloc,
        double* const out_principal_point_precond_tril,
        unsigned int out_principal_point_precond_tril_num_alloc,
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
  __shared__ SharedIndex principal_point_indices_loc[1024];
  principal_point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? principal_point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63;

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
  LoadShared<2, double, double>(
      point, 0 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r8, r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
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
    r7 = fma(r9, r18, r7);
  };
  LoadShared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r19);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r20 = r16 * r11;
    r21 = 2.00000000000000000e+00;
    r20 = r20 * r21;
    r22 = r12 * r10;
    r23 = fma(r15, r22, r20);
    r24 = r11 * r12;
    r25 = r15 * r16;
    r25 = r25 * r21;
    r24 = fma(r21, r24, r25);
    r7 = fma(r19, r23, r7);
    r7 = fma(r8, r24, r7);
    ReadIdx2<1024, double, double, double2>(
        focal, 0 * focal_num_alloc, global_thread_idx, r26, r27);
    r28 = 1.00000000000000008e-15;
  };
  LoadShared<1, double, double>(translation,
                                2 * translation_num_alloc,
                                translation_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared,
                        translation_indices_loc[threadIdx.x].target,
                        r29);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r30 = r16 * r16;
    r30 = r10 * r30;
    r17 = r30 + r17;
    r29 = fma(r19, r17, r29);
    r10 = r15 * r11;
    r10 = r10 * r21;
    r31 = fma(r16, r22, r10);
    r32 = r15 * r12;
    r32 = fma(r21, r32, r20);
    r29 = fma(r8, r31, r29);
    r29 = fma(r9, r32, r29);
    r20 = copysign(1.0, r29);
    r20 = fma(r28, r20, r29);
    r29 = 1.0 / r20;
    r33 = r27 * r29;
    r5 = fma(r7, r33, r5);
    r4 = fma(r4, r6, r2);
    r13 = r14 + r13;
    r13 = r13 + r30;
    r8 = fma(r8, r13, r3);
    r3 = r16 * r12;
    r3 = fma(r21, r3, r10);
    r22 = fma(r11, r22, r25);
    r8 = fma(r19, r3, r8);
    r8 = fma(r9, r22, r8);
    r8 = r26 * r8;
    r4 = fma(r29, r8, r4);
    r9 = fma(r0, r4, r1 * r5);
    ReadIdx1<1024, double, double, double>(
        weight_loss, 6 * weight_loss_num_alloc, global_thread_idx, r19);
    r25 = 0.00000000000000000e+00;
    r19 = fmax(r19, r25);
    r10 = sqrt(r19);
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 2 * weight_loss_num_alloc, global_thread_idx, r30, r2);
    r4 = fma(r30, r4, r2 * r5);
    r5 = fma(r4, r4, r9 * r9);
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 4 * weight_loss_num_alloc, global_thread_idx, r34, r35);
    r36 = 5.00000000000000000e-01;
    r35 = fmax(r35, r28);
    r37 = r35 * r35;
    r38 = r6 * r35;
    r39 = r21 * r35;
    r40 = fmax(r28, r5);
    r41 = sqrt(r40);
    r39 = fma(r41, r39, r35 * r38);
    r39 = r5 <= r37 ? r5 : r39;
    r38 = 2.50000000000000000e+00;
    r41 = r35 * r35;
    r42 = 1.0 / r37;
    r42 = fma(r5, r42, r14);
    r43 = log(r42);
    r41 = r41 * r43;
    r39 = r34 < r38 ? r41 : r39;
    r41 = 1.50000000000000000e+00;
    r43 = r21 * r35;
    r44 = sqrt(r42);
    r44 = r6 + r44;
    r43 = r43 * r35;
    r43 = r43 * r44;
    r39 = r34 < r41 ? r43 : r39;
    r39 = r34 < r36 ? r5 : r39;
    r43 = fmax(r25, r39);
    r44 = 1.0 / r40;
    r44 = r19 * r44;
    r45 = r43 * r44;
    r46 = sqrt(r45);
    r46 = r5 <= r28 ? r10 : r46;
    r10 = r9 * r46;
    r47 = r4 * r46;
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r10, r47);
    r47 = r6 * r9;
    r10 = 2.50000000000000000e-01;
    r48 = r5 <= r37 ? r25 : r25;
    r48 = r34 < r38 ? r25 : r48;
    r48 = r34 < r41 ? r25 : r48;
    r48 = r34 < r36 ? r25 : r48;
    r48 = r10 * r48;
    r45 = rsqrt(r45);
    r39 = copysign(1.0, r39);
    r39 = r14 + r39;
    r44 = r39 * r44;
    r48 = r48 * r45;
    r48 = r48 * r44;
    r48 = r5 <= r28 ? r25 : r48;
    r47 = r47 * r48;
    r43 = r19 * r43;
    r19 = -5.00000000000000000e-01;
    r39 = -1.00000000000000008e-15;
    r39 = r39 + r5;
    r39 = copysign(1.0, r39);
    r39 = r14 + r39;
    r14 = r40 * r40;
    r14 = 1.0 / r14;
    r43 = r43 * r19;
    r43 = r43 * r39;
    r43 = r43 * r14;
    r14 = r26 * r29;
    r19 = r30 * r21;
    r19 = r19 * r4;
    r10 = r0 * r26;
    r10 = r10 * r21;
    r10 = r10 * r9;
    r10 = fma(r29, r10, r19 * r14);
    r14 = r36 * r35;
    r40 = rsqrt(r40);
    r14 = r14 * r39;
    r14 = r14 * r40;
    r40 = r10 * r14;
    r40 = r5 <= r37 ? r10 : r40;
    r39 = 1.0 / r42;
    r49 = r10 * r39;
    r40 = r34 < r38 ? r49 : r40;
    r42 = rsqrt(r42);
    r49 = r10 * r42;
    r40 = r34 < r41 ? r49 : r40;
    r40 = r34 < r36 ? r10 : r40;
    r49 = r36 * r40;
    r49 = fma(r44, r49, r10 * r43);
    r49 = r36 * r49;
    r49 = r49 * r45;
    r49 = r5 <= r28 ? r25 : r49;
    r10 = fma(r9, r49, r47);
    r50 = r0 * r26;
    r50 = r50 * r46;
    r10 = fma(r29, r50, r10);
    r50 = r6 * r4;
    r50 = r50 * r48;
    r49 = fma(r4, r49, r50);
    r48 = r30 * r26;
    r48 = r48 * r46;
    r49 = fma(r29, r48, r49);
    WriteIdx2<1024, double, double, double2>(out_translation_jac,
                                             0 * out_translation_jac_num_alloc,
                                             global_thread_idx,
                                             r10,
                                             r49);
    r48 = r21 * r4;
    r51 = r2 * r33;
    r52 = r1 * r21;
    r52 = r52 * r9;
    r52 = fma(r33, r52, r48 * r51);
    r53 = r52 * r14;
    r53 = r5 <= r37 ? r52 : r53;
    r54 = r52 * r39;
    r53 = r34 < r38 ? r54 : r53;
    r54 = r52 * r42;
    r53 = r34 < r41 ? r54 : r53;
    r53 = r34 < r36 ? r52 : r53;
    r54 = r36 * r53;
    r54 = fma(r44, r54, r52 * r43);
    r54 = r36 * r54;
    r54 = r54 * r45;
    r54 = r5 <= r28 ? r25 : r54;
    r52 = fma(r9, r54, r47);
    r55 = r1 * r46;
    r52 = fma(r33, r55, r52);
    r54 = fma(r4, r54, r50);
    r54 = fma(r46, r51, r54);
    WriteIdx2<1024, double, double, double2>(out_translation_jac,
                                             2 * out_translation_jac_num_alloc,
                                             global_thread_idx,
                                             r52,
                                             r54);
    r51 = r21 * r9;
    r27 = r27 * r6;
    r20 = r20 * r20;
    r20 = 1.0 / r20;
    r27 = r27 * r7;
    r27 = r27 * r20;
    r7 = r0 * r6;
    r7 = r7 * r20;
    r7 = fma(r8, r7, r1 * r27);
    r55 = r30 * r6;
    r55 = r55 * r20;
    r55 = fma(r8, r55, r2 * r27);
    r51 = fma(r55, r48, r7 * r51);
    r56 = r51 * r14;
    r56 = r5 <= r37 ? r51 : r56;
    r57 = r51 * r39;
    r56 = r34 < r38 ? r57 : r56;
    r57 = r51 * r42;
    r56 = r34 < r41 ? r57 : r56;
    r56 = r34 < r36 ? r51 : r56;
    r57 = r36 * r56;
    r57 = fma(r44, r57, r51 * r43);
    r57 = r36 * r57;
    r57 = r57 * r45;
    r57 = r5 <= r28 ? r25 : r57;
    r51 = fma(r9, r57, r47);
    r51 = fma(r46, r7, r51);
    r55 = fma(r46, r55, r50);
    r55 = fma(r4, r57, r55);
    WriteIdx2<1024, double, double, double2>(out_translation_jac,
                                             4 * out_translation_jac_num_alloc,
                                             global_thread_idx,
                                             r51,
                                             r55);
    r57 = r4 * r49;
    r7 = r6 * r46;
    r58 = r9 * r7;
    r57 = fma(r10, r58, r7 * r57);
    r59 = r4 * r54;
    r59 = fma(r7, r59, r52 * r58);
    WriteSum2<double, double>((double*)inout_shared, r57, r59);
  };
  FlushSumShared<2, double>(out_translation_njtr,
                            0 * out_translation_njtr_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r59 = r4 * r55;
    r59 = fma(r7, r59, r51 * r58);
    WriteSum1<double, double>((double*)inout_shared, r59);
  };
  FlushSumShared<1, double>(out_translation_njtr,
                            2 * out_translation_njtr_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r59 = fma(r49, r49, r10 * r10);
    r57 = fma(r54, r54, r52 * r52);
    WriteSum2<double, double>((double*)inout_shared, r59, r57);
  };
  FlushSumShared<2, double>(out_translation_precond_diag,
                            0 * out_translation_precond_diag_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r57 = fma(r51, r51, r55 * r55);
    WriteSum1<double, double>((double*)inout_shared, r57);
  };
  FlushSumShared<1, double>(out_translation_precond_diag,
                            2 * out_translation_precond_diag_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r57 = fma(r49, r54, r10 * r52);
    r10 = fma(r49, r55, r10 * r51);
    WriteSum2<double, double>((double*)inout_shared, r57, r10);
  };
  FlushSumShared<2, double>(out_translation_precond_tril,
                            0 * out_translation_precond_tril_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r51 = fma(r54, r55, r52 * r51);
    WriteSum1<double, double>((double*)inout_shared, r51);
  };
  FlushSumShared<1, double>(out_translation_precond_tril,
                            2 * out_translation_precond_tril_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r51 = fma(r0, r46, r47);
    r52 = r0 * r21;
    r52 = fma(r9, r52, r19);
    r19 = r52 * r14;
    r19 = r5 <= r37 ? r52 : r19;
    r10 = r52 * r39;
    r19 = r34 < r38 ? r10 : r19;
    r10 = r52 * r42;
    r19 = r34 < r41 ? r10 : r19;
    r19 = r34 < r36 ? r52 : r19;
    r10 = r36 * r19;
    r52 = fma(r52, r43, r44 * r10);
    r52 = r36 * r52;
    r52 = r52 * r45;
    r52 = r5 <= r28 ? r25 : r52;
    r51 = fma(r9, r52, r51);
    r10 = fma(r30, r46, r50);
    r10 = fma(r4, r52, r10);
    WriteIdx2<1024, double, double, double2>(
        out_principal_point_jac,
        0 * out_principal_point_jac_num_alloc,
        global_thread_idx,
        r51,
        r10);
    r52 = fma(r1, r46, r47);
    r57 = r1 * r21;
    r57 = fma(r2, r48, r9 * r57);
    r59 = r57 * r14;
    r59 = r5 <= r37 ? r57 : r59;
    r60 = r57 * r39;
    r59 = r34 < r38 ? r60 : r59;
    r60 = r57 * r42;
    r59 = r34 < r41 ? r60 : r59;
    r59 = r34 < r36 ? r57 : r59;
    r60 = r36 * r59;
    r60 = fma(r44, r60, r57 * r43);
    r60 = r36 * r60;
    r60 = r60 * r45;
    r60 = r5 <= r28 ? r25 : r60;
    r52 = fma(r9, r60, r52);
    r57 = fma(r2, r46, r50);
    r57 = fma(r4, r60, r57);
    WriteIdx2<1024, double, double, double2>(
        out_principal_point_jac,
        2 * out_principal_point_jac_num_alloc,
        global_thread_idx,
        r52,
        r57);
    r60 = r4 * r10;
    r60 = fma(r7, r60, r51 * r58);
    r61 = r4 * r57;
    r61 = fma(r52, r58, r7 * r61);
    WriteSum2<double, double>((double*)inout_shared, r60, r61);
  };
  FlushSumShared<2, double>(out_principal_point_njtr,
                            0 * out_principal_point_njtr_num_alloc,
                            principal_point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r61 = fma(r51, r51, r10 * r10);
    r60 = fma(r57, r57, r52 * r52);
    WriteSum2<double, double>((double*)inout_shared, r61, r60);
  };
  FlushSumShared<2, double>(out_principal_point_precond_diag,
                            0 * out_principal_point_precond_diag_num_alloc,
                            principal_point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r52 = fma(r51, r52, r10 * r57);
    WriteSum1<double, double>((double*)inout_shared, r52);
  };
  FlushSumShared<1, double>(out_principal_point_precond_tril,
                            0 * out_principal_point_precond_tril_num_alloc,
                            principal_point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r52 = r21 * r9;
    r24 = fma(r31, r27, r24 * r33);
    r51 = r26 * r13;
    r60 = r31 * r6;
    r60 = r60 * r20;
    r60 = fma(r8, r60, r29 * r51);
    r51 = fma(r0, r60, r1 * r24);
    r60 = fma(r30, r60, r2 * r24);
    r52 = fma(r60, r48, r51 * r52);
    r24 = r52 * r14;
    r24 = r5 <= r37 ? r52 : r24;
    r61 = r52 * r39;
    r24 = r34 < r38 ? r61 : r24;
    r61 = r52 * r42;
    r24 = r34 < r41 ? r61 : r24;
    r24 = r34 < r36 ? r52 : r24;
    r61 = r36 * r24;
    r52 = fma(r52, r43, r44 * r61);
    r52 = r36 * r52;
    r52 = r52 * r45;
    r52 = r5 <= r28 ? r25 : r52;
    r61 = fma(r9, r52, r47);
    r61 = fma(r46, r51, r61);
    r52 = fma(r4, r52, r50);
    r52 = fma(r46, r60, r52);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             0 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r61,
                                             r52);
    r60 = r21 * r9;
    r51 = r26 * r22;
    r62 = r32 * r6;
    r62 = r62 * r20;
    r62 = fma(r8, r62, r29 * r51);
    r18 = fma(r32, r27, r18 * r33);
    r51 = fma(r1, r18, r0 * r62);
    r18 = fma(r2, r18, r30 * r62);
    r60 = fma(r18, r48, r51 * r60);
    r62 = r60 * r14;
    r62 = r5 <= r37 ? r60 : r62;
    r63 = r60 * r39;
    r62 = r34 < r38 ? r63 : r62;
    r63 = r60 * r42;
    r62 = r34 < r41 ? r63 : r62;
    r62 = r34 < r36 ? r60 : r62;
    r63 = r36 * r62;
    r63 = fma(r44, r63, r60 * r43);
    r63 = r36 * r63;
    r63 = r63 * r45;
    r63 = r5 <= r28 ? r25 : r63;
    r60 = fma(r9, r63, r47);
    r60 = fma(r46, r51, r60);
    r63 = fma(r4, r63, r50);
    r63 = fma(r46, r18, r63);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             2 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r60,
                                             r63);
    r18 = r26 * r3;
    r51 = r17 * r6;
    r51 = r51 * r20;
    r51 = fma(r8, r51, r29 * r18);
    r33 = fma(r23, r33, r17 * r27);
    r23 = fma(r1, r33, r0 * r51);
    r47 = fma(r46, r23, r47);
    r27 = r21 * r9;
    r33 = fma(r2, r33, r30 * r51);
    r48 = fma(r33, r48, r23 * r27);
    r14 = r48 * r14;
    r14 = r5 <= r37 ? r48 : r14;
    r39 = r48 * r39;
    r14 = r34 < r38 ? r39 : r14;
    r42 = r48 * r42;
    r14 = r34 < r41 ? r42 : r14;
    r14 = r34 < r36 ? r48 : r14;
    r34 = r36 * r14;
    r34 = fma(r44, r34, r48 * r43);
    r34 = r36 * r34;
    r34 = r34 * r45;
    r34 = r5 <= r28 ? r25 : r34;
    r47 = fma(r9, r34, r47);
    r33 = fma(r46, r33, r50);
    r33 = fma(r4, r34, r33);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             4 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r47,
                                             r33);
    r34 = r4 * r52;
    r34 = fma(r61, r58, r7 * r34);
    r50 = r4 * r63;
    r50 = fma(r7, r50, r60 * r58);
    WriteSum2<double, double>((double*)inout_shared, r34, r50);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = r4 * r33;
    r50 = fma(r7, r50, r47 * r58);
    WriteSum1<double, double>((double*)inout_shared, r50);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = fma(r61, r61, r52 * r52);
    r58 = fma(r63, r63, r60 * r60);
    WriteSum2<double, double>((double*)inout_shared, r50, r58);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r58 = fma(r33, r33, r47 * r47);
    WriteSum1<double, double>((double*)inout_shared, r58);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r58 = fma(r52, r63, r61 * r60);
    r61 = fma(r61, r47, r52 * r33);
    WriteSum2<double, double>((double*)inout_shared, r58, r61);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = fma(r63, r33, r60 * r47);
    WriteSum1<double, double>((double*)inout_shared, r47);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
}

void PinholeSplitFixedRotationFixedFocalResJac(
    double* rotation,
    unsigned int rotation_num_alloc,
    double* translation,
    unsigned int translation_num_alloc,
    SharedIndex* translation_indices,
    double* principal_point,
    unsigned int principal_point_num_alloc,
    SharedIndex* principal_point_indices,
    double* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* weight_loss,
    unsigned int weight_loss_num_alloc,
    double* focal,
    unsigned int focal_num_alloc,
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
    double* out_principal_point_jac,
    unsigned int out_principal_point_jac_num_alloc,
    double* const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    double* const out_principal_point_precond_diag,
    unsigned int out_principal_point_precond_diag_num_alloc,
    double* const out_principal_point_precond_tril,
    unsigned int out_principal_point_precond_tril_num_alloc,
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
  PinholeSplitFixedRotationFixedFocalResJacKernel<<<n_blocks, 1024>>>(
      rotation,
      rotation_num_alloc,
      translation,
      translation_num_alloc,
      translation_indices,
      principal_point,
      principal_point_num_alloc,
      principal_point_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      weight_loss,
      weight_loss_num_alloc,
      focal,
      focal_num_alloc,
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
      out_principal_point_jac,
      out_principal_point_jac_num_alloc,
      out_principal_point_njtr,
      out_principal_point_njtr_num_alloc,
      out_principal_point_precond_diag,
      out_principal_point_precond_diag_num_alloc,
      out_principal_point_precond_tril,
      out_principal_point_precond_tril_num_alloc,
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