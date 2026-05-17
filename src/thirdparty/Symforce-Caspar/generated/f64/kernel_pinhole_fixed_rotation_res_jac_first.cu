#include "kernel_pinhole_fixed_rotation_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeFixedRotationResJacFirstKernel(
        double* rotation,
        unsigned int rotation_num_alloc,
        double* translation,
        unsigned int translation_num_alloc,
        SharedIndex* translation_indices,
        double* calib,
        unsigned int calib_num_alloc,
        SharedIndex* calib_indices,
        double* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* weight_loss,
        unsigned int weight_loss_num_alloc,
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
        double* out_calib_jac,
        unsigned int out_calib_jac_num_alloc,
        double* const out_calib_njtr,
        unsigned int out_calib_njtr_num_alloc,
        double* const out_calib_precond_diag,
        unsigned int out_calib_precond_diag_num_alloc,
        double* const out_calib_precond_tril,
        unsigned int out_calib_precond_tril_num_alloc,
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
  __shared__ SharedIndex calib_indices_loc[1024];
  calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 0 * weight_loss_num_alloc, global_thread_idx, r0, r1);
  };
  LoadShared<2, double, double>(
      calib, 2 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r2, r3);
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
  };
  LoadShared<2, double, double>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r26, r27);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
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
    r9 = r26 * r8;
    r4 = fma(r29, r9, r4);
    r19 = fma(r0, r4, r1 * r5);
    ReadIdx1<1024, double, double, double>(
        weight_loss, 6 * weight_loss_num_alloc, global_thread_idx, r25);
    r10 = 0.00000000000000000e+00;
    r25 = fmax(r25, r10);
    r30 = sqrt(r25);
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 2 * weight_loss_num_alloc, global_thread_idx, r2, r34);
    r4 = fma(r2, r4, r34 * r5);
    r5 = fma(r19, r19, r4 * r4);
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 4 * weight_loss_num_alloc, global_thread_idx, r35, r36);
    r37 = 5.00000000000000000e-01;
    r36 = fmax(r36, r28);
    r38 = r36 * r36;
    r39 = r21 * r36;
    r40 = fmax(r28, r5);
    r41 = sqrt(r40);
    r42 = r6 * r36;
    r42 = fma(r36, r42, r41 * r39);
    r42 = r5 <= r38 ? r5 : r42;
    r39 = 2.50000000000000000e+00;
    r41 = r36 * r36;
    r43 = 1.0 / r38;
    r43 = fma(r5, r43, r14);
    r44 = log(r43);
    r41 = r41 * r44;
    r42 = r35 < r39 ? r41 : r42;
    r41 = 1.50000000000000000e+00;
    r44 = r21 * r36;
    r45 = sqrt(r43);
    r45 = r6 + r45;
    r44 = r44 * r36;
    r44 = r44 * r45;
    r42 = r35 < r41 ? r44 : r42;
    r42 = r35 < r37 ? r5 : r42;
    r44 = fmax(r10, r42);
    r45 = 1.0 / r40;
    r45 = r25 * r45;
    r46 = r44 * r45;
    r47 = sqrt(r46);
    r47 = r5 <= r28 ? r30 : r47;
    r30 = r19 * r47;
    r48 = r4 * r47;
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r30, r48);
    r48 = r4 * r4;
    r48 = r48 * r47;
    r30 = r19 * r19;
    r30 = r30 * r47;
    r30 = fma(r47, r30, r47 * r48);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r30);
  if (global_thread_idx < problem_size) {
    r44 = r25 * r44;
    r25 = -5.00000000000000000e-01;
    r30 = -1.00000000000000008e-15;
    r30 = r30 + r5;
    r30 = copysign(1.0, r30);
    r30 = r14 + r30;
    r48 = r40 * r40;
    r48 = 1.0 / r48;
    r44 = r44 * r25;
    r44 = r44 * r30;
    r44 = r44 * r48;
    r48 = r0 * r21;
    r48 = r48 * r19;
    r25 = r29 * r48;
    r49 = r26 * r2;
    r49 = r49 * r21;
    r49 = r49 * r4;
    r49 = fma(r29, r49, r26 * r25);
    r50 = r37 * r36;
    r40 = rsqrt(r40);
    r50 = r50 * r30;
    r50 = r50 * r40;
    r40 = r49 * r50;
    r40 = r5 <= r38 ? r49 : r40;
    r30 = 1.0 / r43;
    r51 = r49 * r30;
    r40 = r35 < r39 ? r51 : r40;
    r43 = rsqrt(r43);
    r51 = r49 * r43;
    r40 = r35 < r41 ? r51 : r40;
    r40 = r35 < r37 ? r49 : r40;
    r51 = r37 * r40;
    r42 = copysign(1.0, r42);
    r42 = r14 + r42;
    r45 = r42 * r45;
    r51 = fma(r45, r51, r49 * r44);
    r51 = r37 * r51;
    r46 = rsqrt(r46);
    r51 = r51 * r46;
    r51 = r5 <= r28 ? r10 : r51;
    r49 = r6 * r19;
    r42 = 2.50000000000000000e-01;
    r14 = r5 <= r38 ? r10 : r10;
    r14 = r35 < r39 ? r10 : r14;
    r14 = r35 < r41 ? r10 : r14;
    r14 = r35 < r37 ? r10 : r14;
    r14 = r42 * r14;
    r14 = r14 * r46;
    r14 = r14 * r45;
    r14 = r5 <= r28 ? r10 : r14;
    r49 = r49 * r14;
    r42 = fma(r19, r51, r49);
    r52 = r26 * r0;
    r52 = r52 * r47;
    r42 = fma(r29, r52, r42);
    r52 = r6 * r4;
    r52 = r52 * r14;
    r51 = fma(r4, r51, r52);
    r14 = r26 * r2;
    r14 = r14 * r47;
    r51 = fma(r29, r14, r51);
    WriteIdx2<1024, double, double, double2>(out_translation_jac,
                                             0 * out_translation_jac_num_alloc,
                                             global_thread_idx,
                                             r42,
                                             r51);
    r14 = r21 * r19;
    r53 = r1 * r14;
    r54 = r21 * r4;
    r55 = r34 * r33;
    r54 = fma(r55, r54, r33 * r53);
    r53 = r54 * r50;
    r53 = r5 <= r38 ? r54 : r53;
    r56 = r54 * r30;
    r53 = r35 < r39 ? r56 : r53;
    r56 = r54 * r43;
    r53 = r35 < r41 ? r56 : r53;
    r53 = r35 < r37 ? r54 : r53;
    r56 = r37 * r53;
    r54 = fma(r54, r44, r45 * r56);
    r54 = r37 * r54;
    r54 = r54 * r46;
    r54 = r5 <= r28 ? r10 : r54;
    r56 = fma(r19, r54, r49);
    r57 = r1 * r47;
    r56 = fma(r33, r57, r56);
    r54 = fma(r4, r54, r52);
    r54 = fma(r47, r55, r54);
    WriteIdx2<1024, double, double, double2>(out_translation_jac,
                                             2 * out_translation_jac_num_alloc,
                                             global_thread_idx,
                                             r56,
                                             r54);
    r55 = r21 * r4;
    r27 = r27 * r6;
    r20 = r20 * r20;
    r20 = 1.0 / r20;
    r27 = r27 * r7;
    r27 = r27 * r20;
    r57 = r2 * r6;
    r57 = r57 * r20;
    r57 = fma(r9, r57, r34 * r27);
    r58 = r0 * r6;
    r58 = r58 * r20;
    r58 = fma(r9, r58, r1 * r27);
    r55 = fma(r58, r14, r57 * r55);
    r59 = r55 * r50;
    r59 = r5 <= r38 ? r55 : r59;
    r60 = r55 * r30;
    r59 = r35 < r39 ? r60 : r59;
    r60 = r55 * r43;
    r59 = r35 < r41 ? r60 : r59;
    r59 = r35 < r37 ? r55 : r59;
    r60 = r37 * r59;
    r60 = fma(r45, r60, r55 * r44);
    r60 = r37 * r60;
    r60 = r60 * r46;
    r60 = r5 <= r28 ? r10 : r60;
    r55 = fma(r19, r60, r49);
    r55 = fma(r47, r58, r55);
    r60 = fma(r4, r60, r52);
    r60 = fma(r47, r57, r60);
    WriteIdx2<1024, double, double, double2>(out_translation_jac,
                                             4 * out_translation_jac_num_alloc,
                                             global_thread_idx,
                                             r55,
                                             r60);
    r57 = r19 * r42;
    r58 = r6 * r47;
    r61 = r4 * r58;
    r57 = fma(r51, r61, r58 * r57);
    r62 = r19 * r56;
    r62 = fma(r58, r62, r54 * r61);
    WriteSum2<double, double>((double*)inout_shared, r57, r62);
  };
  FlushSumShared<2, double>(out_translation_njtr,
                            0 * out_translation_njtr_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r62 = r19 * r55;
    r62 = fma(r58, r62, r60 * r61);
    WriteSum1<double, double>((double*)inout_shared, r62);
  };
  FlushSumShared<1, double>(out_translation_njtr,
                            2 * out_translation_njtr_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r62 = fma(r51, r51, r42 * r42);
    r57 = fma(r56, r56, r54 * r54);
    WriteSum2<double, double>((double*)inout_shared, r62, r57);
  };
  FlushSumShared<2, double>(out_translation_precond_diag,
                            0 * out_translation_precond_diag_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r57 = fma(r55, r55, r60 * r60);
    WriteSum1<double, double>((double*)inout_shared, r57);
  };
  FlushSumShared<1, double>(out_translation_precond_diag,
                            2 * out_translation_precond_diag_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r57 = fma(r42, r56, r51 * r54);
    r51 = fma(r51, r60, r42 * r55);
    WriteSum2<double, double>((double*)inout_shared, r57, r51);
  };
  FlushSumShared<2, double>(out_translation_precond_tril,
                            0 * out_translation_precond_tril_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r60 = fma(r54, r60, r56 * r55);
    WriteSum1<double, double>((double*)inout_shared, r60);
  };
  FlushSumShared<1, double>(out_translation_precond_tril,
                            2 * out_translation_precond_tril_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r60 = r2 * r21;
    r60 = r60 * r8;
    r60 = r60 * r4;
    r25 = fma(r8, r25, r29 * r60);
    r60 = r25 * r50;
    r60 = r5 <= r38 ? r25 : r60;
    r54 = r25 * r30;
    r60 = r35 < r39 ? r54 : r60;
    r54 = r25 * r43;
    r60 = r35 < r41 ? r54 : r60;
    r60 = r35 < r37 ? r25 : r60;
    r54 = r37 * r60;
    r54 = fma(r45, r54, r25 * r44);
    r54 = r37 * r54;
    r54 = r54 * r46;
    r54 = r5 <= r28 ? r10 : r54;
    r25 = fma(r19, r54, r49);
    r51 = r0 * r8;
    r51 = r51 * r47;
    r25 = fma(r29, r51, r25);
    r54 = fma(r4, r54, r52);
    r51 = r2 * r8;
    r51 = r51 * r47;
    r54 = fma(r29, r51, r54);
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             0 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r25,
                                             r54);
    r51 = r34 * r21;
    r51 = r51 * r7;
    r51 = r51 * r4;
    r57 = r1 * r7;
    r57 = r57 * r29;
    r57 = fma(r14, r57, r29 * r51);
    r51 = r57 * r50;
    r51 = r5 <= r38 ? r57 : r51;
    r62 = r57 * r30;
    r51 = r35 < r39 ? r62 : r51;
    r62 = r57 * r43;
    r51 = r35 < r41 ? r62 : r51;
    r51 = r35 < r37 ? r57 : r51;
    r62 = r37 * r51;
    r57 = fma(r57, r44, r45 * r62);
    r57 = r37 * r57;
    r57 = r57 * r46;
    r57 = r5 <= r28 ? r10 : r57;
    r62 = fma(r19, r57, r49);
    r63 = r1 * r7;
    r63 = r63 * r47;
    r62 = fma(r29, r63, r62);
    r57 = fma(r4, r57, r52);
    r63 = r34 * r7;
    r63 = r63 * r47;
    r57 = fma(r29, r63, r57);
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             2 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r62,
                                             r57);
    r63 = fma(r0, r47, r49);
    r64 = r2 * r21;
    r64 = fma(r4, r64, r48);
    r48 = r64 * r50;
    r48 = r5 <= r38 ? r64 : r48;
    r65 = r64 * r30;
    r48 = r35 < r39 ? r65 : r48;
    r65 = r64 * r43;
    r48 = r35 < r41 ? r65 : r48;
    r48 = r35 < r37 ? r64 : r48;
    r65 = r37 * r48;
    r65 = fma(r45, r65, r64 * r44);
    r65 = r37 * r65;
    r65 = r65 * r46;
    r65 = r5 <= r28 ? r10 : r65;
    r63 = fma(r19, r65, r63);
    r64 = fma(r2, r47, r52);
    r64 = fma(r4, r65, r64);
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             4 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r63,
                                             r64);
    r65 = r34 * r21;
    r65 = fma(r1, r14, r4 * r65);
    r66 = r65 * r50;
    r66 = r5 <= r38 ? r65 : r66;
    r67 = r65 * r30;
    r66 = r35 < r39 ? r67 : r66;
    r67 = r65 * r43;
    r66 = r35 < r41 ? r67 : r66;
    r66 = r35 < r37 ? r65 : r66;
    r67 = r37 * r66;
    r67 = fma(r45, r67, r65 * r44);
    r67 = r37 * r67;
    r67 = r67 * r46;
    r67 = r5 <= r28 ? r10 : r67;
    r65 = fma(r19, r67, r49);
    r65 = fma(r1, r47, r65);
    r67 = fma(r4, r67, r52);
    r67 = fma(r34, r47, r67);
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             6 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r65,
                                             r67);
    r68 = r19 * r25;
    r68 = fma(r54, r61, r58 * r68);
    r69 = r19 * r62;
    r69 = fma(r58, r69, r57 * r61);
    WriteSum2<double, double>((double*)inout_shared, r68, r69);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            0 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r69 = r19 * r63;
    r69 = fma(r64, r61, r58 * r69);
    r68 = r19 * r65;
    r68 = fma(r58, r68, r67 * r61);
    WriteSum2<double, double>((double*)inout_shared, r69, r68);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            2 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r68 = fma(r54, r54, r25 * r25);
    r69 = fma(r62, r62, r57 * r57);
    WriteSum2<double, double>((double*)inout_shared, r68, r69);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            0 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r69 = fma(r64, r64, r63 * r63);
    r68 = fma(r65, r65, r67 * r67);
    WriteSum2<double, double>((double*)inout_shared, r69, r68);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            2 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r68 = fma(r54, r57, r25 * r62);
    r69 = fma(r54, r64, r25 * r63);
    WriteSum2<double, double>((double*)inout_shared, r68, r69);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            0 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r54 = fma(r54, r67, r25 * r65);
    r69 = fma(r62, r63, r57 * r64);
    WriteSum2<double, double>((double*)inout_shared, r54, r69);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            2 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r57 = fma(r57, r67, r62 * r65);
    r67 = fma(r63, r65, r64 * r67);
    WriteSum2<double, double>((double*)inout_shared, r57, r67);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            4 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = fma(r24, r33, r31 * r27);
    r67 = r31 * r6;
    r67 = r67 * r20;
    r57 = r26 * r13;
    r57 = fma(r29, r57, r9 * r67);
    r67 = fma(r0, r57, r1 * r24);
    r64 = fma(r47, r67, r49);
    r69 = r21 * r4;
    r57 = fma(r2, r57, r34 * r24);
    r67 = fma(r67, r14, r57 * r69);
    r69 = r67 * r50;
    r69 = r5 <= r38 ? r67 : r69;
    r24 = r67 * r30;
    r69 = r35 < r39 ? r24 : r69;
    r24 = r67 * r43;
    r69 = r35 < r41 ? r24 : r69;
    r69 = r35 < r37 ? r67 : r69;
    r24 = r37 * r69;
    r24 = fma(r45, r24, r67 * r44);
    r24 = r37 * r24;
    r24 = r24 * r46;
    r24 = r5 <= r28 ? r10 : r24;
    r64 = fma(r19, r24, r64);
    r57 = fma(r47, r57, r52);
    r57 = fma(r4, r24, r57);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             0 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r64,
                                             r57);
    r24 = r21 * r4;
    r67 = r32 * r6;
    r67 = r67 * r20;
    r54 = r26 * r22;
    r54 = fma(r29, r54, r9 * r67);
    r18 = fma(r32, r27, r18 * r33);
    r67 = fma(r34, r18, r2 * r54);
    r18 = fma(r1, r18, r0 * r54);
    r24 = fma(r18, r14, r67 * r24);
    r54 = r24 * r50;
    r54 = r5 <= r38 ? r24 : r54;
    r68 = r24 * r30;
    r54 = r35 < r39 ? r68 : r54;
    r68 = r24 * r43;
    r54 = r35 < r41 ? r68 : r54;
    r54 = r35 < r37 ? r24 : r54;
    r68 = r37 * r54;
    r68 = fma(r45, r68, r24 * r44);
    r68 = r37 * r68;
    r68 = r68 * r46;
    r68 = r5 <= r28 ? r10 : r68;
    r24 = fma(r19, r68, r49);
    r24 = fma(r47, r18, r24);
    r68 = fma(r4, r68, r52);
    r68 = fma(r47, r67, r68);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             2 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r24,
                                             r68);
    r67 = r17 * r6;
    r67 = r67 * r20;
    r20 = r26 * r3;
    r20 = fma(r29, r20, r9 * r67);
    r27 = fma(r17, r27, r23 * r33);
    r33 = fma(r1, r27, r0 * r20);
    r49 = fma(r47, r33, r49);
    r23 = r21 * r4;
    r27 = fma(r34, r27, r2 * r20);
    r33 = fma(r33, r14, r27 * r23);
    r50 = r33 * r50;
    r50 = r5 <= r38 ? r33 : r50;
    r30 = r33 * r30;
    r50 = r35 < r39 ? r30 : r50;
    r43 = r33 * r43;
    r50 = r35 < r41 ? r43 : r50;
    r50 = r35 < r37 ? r33 : r50;
    r35 = r37 * r50;
    r35 = fma(r45, r35, r33 * r44);
    r35 = r37 * r35;
    r35 = r35 * r46;
    r35 = r5 <= r28 ? r10 : r35;
    r49 = fma(r19, r35, r49);
    r27 = fma(r47, r27, r52);
    r27 = fma(r4, r35, r27);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             4 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r49,
                                             r27);
    r35 = r19 * r64;
    r35 = fma(r57, r61, r58 * r35);
    r52 = r19 * r24;
    r52 = fma(r58, r52, r68 * r61);
    WriteSum2<double, double>((double*)inout_shared, r35, r52);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r52 = r19 * r49;
    r52 = fma(r58, r52, r27 * r61);
    WriteSum1<double, double>((double*)inout_shared, r52);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r52 = fma(r64, r64, r57 * r57);
    r61 = fma(r24, r24, r68 * r68);
    WriteSum2<double, double>((double*)inout_shared, r52, r61);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r61 = fma(r49, r49, r27 * r27);
    WriteSum1<double, double>((double*)inout_shared, r61);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r61 = fma(r57, r68, r64 * r24);
    r57 = fma(r57, r27, r64 * r49);
    WriteSum2<double, double>((double*)inout_shared, r61, r57);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fma(r68, r27, r24 * r49);
    WriteSum1<double, double>((double*)inout_shared, r27);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void PinholeFixedRotationResJacFirst(
    double* rotation,
    unsigned int rotation_num_alloc,
    double* translation,
    unsigned int translation_num_alloc,
    SharedIndex* translation_indices,
    double* calib,
    unsigned int calib_num_alloc,
    SharedIndex* calib_indices,
    double* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* weight_loss,
    unsigned int weight_loss_num_alloc,
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
    double* out_calib_jac,
    unsigned int out_calib_jac_num_alloc,
    double* const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc,
    double* const out_calib_precond_diag,
    unsigned int out_calib_precond_diag_num_alloc,
    double* const out_calib_precond_tril,
    unsigned int out_calib_precond_tril_num_alloc,
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
  PinholeFixedRotationResJacFirstKernel<<<n_blocks, 1024>>>(
      rotation,
      rotation_num_alloc,
      translation,
      translation_num_alloc,
      translation_indices,
      calib,
      calib_num_alloc,
      calib_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      weight_loss,
      weight_loss_num_alloc,
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
      out_calib_jac,
      out_calib_jac_num_alloc,
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      out_calib_precond_diag,
      out_calib_precond_diag_num_alloc,
      out_calib_precond_tril,
      out_calib_precond_tril_num_alloc,
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