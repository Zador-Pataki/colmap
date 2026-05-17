#include "kernel_pinhole_split_fixed_rotation_fixed_principal_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedRotationFixedPrincipalPointResJacFirstKernel(
        double* rotation,
        unsigned int rotation_num_alloc,
        double* translation,
        unsigned int translation_num_alloc,
        SharedIndex* translation_indices,
        double* focal,
        unsigned int focal_num_alloc,
        SharedIndex* focal_indices,
        double* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* weight_loss,
        unsigned int weight_loss_num_alloc,
        double* principal_point,
        unsigned int principal_point_num_alloc,
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
        double* out_focal_jac,
        unsigned int out_focal_jac_num_alloc,
        double* const out_focal_njtr,
        unsigned int out_focal_njtr_num_alloc,
        double* const out_focal_precond_diag,
        unsigned int out_focal_precond_diag_num_alloc,
        double* const out_focal_precond_tril,
        unsigned int out_focal_precond_tril_num_alloc,
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
  __shared__ SharedIndex focal_indices_loc[1024];
  focal_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_indices[global_thread_idx]
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
      r61, r62, r63, r64, r65;

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
      focal, 0 * focal_num_alloc, focal_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, focal_indices_loc[threadIdx.x].target, r26, r27);
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
    r5 = fma(r4, r4, r19 * r19);
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 4 * weight_loss_num_alloc, global_thread_idx, r35, r36);
    r37 = 5.00000000000000000e-01;
    r36 = fmax(r36, r28);
    r38 = r36 * r36;
    r39 = r6 * r36;
    r40 = r21 * r36;
    r41 = fmax(r28, r5);
    r42 = sqrt(r41);
    r40 = fma(r42, r40, r36 * r39);
    r40 = r5 <= r38 ? r5 : r40;
    r39 = 2.50000000000000000e+00;
    r42 = r36 * r36;
    r43 = 1.0 / r38;
    r43 = fma(r5, r43, r14);
    r44 = log(r43);
    r42 = r42 * r44;
    r40 = r35 < r39 ? r42 : r40;
    r42 = 1.50000000000000000e+00;
    r44 = r21 * r36;
    r45 = sqrt(r43);
    r45 = r6 + r45;
    r44 = r44 * r36;
    r44 = r44 * r45;
    r40 = r35 < r42 ? r44 : r40;
    r40 = r35 < r37 ? r5 : r40;
    r44 = fmax(r10, r40);
    r45 = 1.0 / r41;
    r45 = r25 * r45;
    r46 = r44 * r45;
    r47 = sqrt(r46);
    r47 = r5 <= r28 ? r30 : r47;
    r30 = r19 * r47;
    r48 = r4 * r47;
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r30, r48);
    r48 = r19 * r19;
    r48 = r48 * r47;
    r30 = r4 * r4;
    r30 = r30 * r47;
    r30 = fma(r47, r30, r47 * r48);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r30);
  if (global_thread_idx < problem_size) {
    r30 = r6 * r19;
    r48 = 2.50000000000000000e-01;
    r49 = r5 <= r38 ? r10 : r10;
    r49 = r35 < r39 ? r10 : r49;
    r49 = r35 < r42 ? r10 : r49;
    r49 = r35 < r37 ? r10 : r49;
    r49 = r48 * r49;
    r46 = rsqrt(r46);
    r40 = copysign(1.0, r40);
    r40 = r14 + r40;
    r45 = r40 * r45;
    r49 = r49 * r46;
    r49 = r49 * r45;
    r49 = r5 <= r28 ? r10 : r49;
    r30 = r30 * r49;
    r44 = r25 * r44;
    r25 = -5.00000000000000000e-01;
    r40 = -1.00000000000000008e-15;
    r40 = r40 + r5;
    r40 = copysign(1.0, r40);
    r40 = r14 + r40;
    r14 = r41 * r41;
    r14 = 1.0 / r14;
    r44 = r44 * r25;
    r44 = r44 * r40;
    r44 = r44 * r14;
    r14 = r21 * r4;
    r25 = r29 * r14;
    r48 = r2 * r25;
    r50 = r26 * r0;
    r50 = r50 * r21;
    r50 = r50 * r19;
    r50 = fma(r29, r50, r26 * r48);
    r51 = r37 * r36;
    r41 = rsqrt(r41);
    r51 = r51 * r40;
    r51 = r51 * r41;
    r41 = r50 * r51;
    r41 = r5 <= r38 ? r50 : r41;
    r40 = 1.0 / r43;
    r52 = r50 * r40;
    r41 = r35 < r39 ? r52 : r41;
    r43 = rsqrt(r43);
    r52 = r50 * r43;
    r41 = r35 < r42 ? r52 : r41;
    r41 = r35 < r37 ? r50 : r41;
    r52 = r37 * r41;
    r52 = fma(r45, r52, r50 * r44);
    r52 = r37 * r52;
    r52 = r52 * r46;
    r52 = r5 <= r28 ? r10 : r52;
    r50 = fma(r19, r52, r30);
    r53 = r26 * r0;
    r53 = r53 * r47;
    r50 = fma(r29, r53, r50);
    r53 = r6 * r4;
    r53 = r53 * r49;
    r52 = fma(r4, r52, r53);
    r49 = r26 * r2;
    r49 = r49 * r47;
    r52 = fma(r29, r49, r52);
    WriteIdx2<1024, double, double, double2>(out_translation_jac,
                                             0 * out_translation_jac_num_alloc,
                                             global_thread_idx,
                                             r50,
                                             r52);
    r49 = r34 * r33;
    r54 = r1 * r21;
    r54 = r54 * r19;
    r54 = fma(r33, r54, r14 * r49);
    r55 = r54 * r51;
    r55 = r5 <= r38 ? r54 : r55;
    r56 = r54 * r40;
    r55 = r35 < r39 ? r56 : r55;
    r56 = r54 * r43;
    r55 = r35 < r42 ? r56 : r55;
    r55 = r35 < r37 ? r54 : r55;
    r56 = r37 * r55;
    r56 = fma(r45, r56, r54 * r44);
    r56 = r37 * r56;
    r56 = r56 * r46;
    r56 = r5 <= r28 ? r10 : r56;
    r54 = fma(r19, r56, r30);
    r57 = r1 * r47;
    r54 = fma(r33, r57, r54);
    r56 = fma(r4, r56, r53);
    r56 = fma(r47, r49, r56);
    WriteIdx2<1024, double, double, double2>(out_translation_jac,
                                             2 * out_translation_jac_num_alloc,
                                             global_thread_idx,
                                             r54,
                                             r56);
    r49 = r21 * r19;
    r27 = r27 * r6;
    r20 = r20 * r20;
    r20 = 1.0 / r20;
    r27 = r27 * r7;
    r27 = r27 * r20;
    r57 = r0 * r6;
    r57 = r57 * r20;
    r57 = fma(r9, r57, r1 * r27);
    r58 = r2 * r6;
    r58 = r58 * r20;
    r58 = fma(r9, r58, r34 * r27);
    r49 = fma(r58, r14, r57 * r49);
    r59 = r49 * r51;
    r59 = r5 <= r38 ? r49 : r59;
    r60 = r49 * r40;
    r59 = r35 < r39 ? r60 : r59;
    r60 = r49 * r43;
    r59 = r35 < r42 ? r60 : r59;
    r59 = r35 < r37 ? r49 : r59;
    r60 = r37 * r59;
    r60 = fma(r45, r60, r49 * r44);
    r60 = r37 * r60;
    r60 = r60 * r46;
    r60 = r5 <= r28 ? r10 : r60;
    r49 = fma(r19, r60, r30);
    r49 = fma(r47, r57, r49);
    r58 = fma(r47, r58, r53);
    r58 = fma(r4, r60, r58);
    WriteIdx2<1024, double, double, double2>(out_translation_jac,
                                             4 * out_translation_jac_num_alloc,
                                             global_thread_idx,
                                             r49,
                                             r58);
    r60 = r4 * r52;
    r57 = r6 * r47;
    r61 = r19 * r57;
    r60 = fma(r50, r61, r57 * r60);
    r62 = r4 * r56;
    r62 = fma(r57, r62, r54 * r61);
    WriteSum2<double, double>((double*)inout_shared, r60, r62);
  };
  FlushSumShared<2, double>(out_translation_njtr,
                            0 * out_translation_njtr_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r62 = r4 * r58;
    r62 = fma(r57, r62, r49 * r61);
    WriteSum1<double, double>((double*)inout_shared, r62);
  };
  FlushSumShared<1, double>(out_translation_njtr,
                            2 * out_translation_njtr_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r62 = fma(r52, r52, r50 * r50);
    r60 = fma(r56, r56, r54 * r54);
    WriteSum2<double, double>((double*)inout_shared, r62, r60);
  };
  FlushSumShared<2, double>(out_translation_precond_diag,
                            0 * out_translation_precond_diag_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r60 = fma(r49, r49, r58 * r58);
    WriteSum1<double, double>((double*)inout_shared, r60);
  };
  FlushSumShared<1, double>(out_translation_precond_diag,
                            2 * out_translation_precond_diag_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r60 = fma(r52, r56, r50 * r54);
    r50 = fma(r52, r58, r50 * r49);
    WriteSum2<double, double>((double*)inout_shared, r60, r50);
  };
  FlushSumShared<2, double>(out_translation_precond_tril,
                            0 * out_translation_precond_tril_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = fma(r56, r58, r54 * r49);
    WriteSum1<double, double>((double*)inout_shared, r49);
  };
  FlushSumShared<1, double>(out_translation_precond_tril,
                            2 * out_translation_precond_tril_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = r0 * r21;
    r49 = r49 * r8;
    r49 = r49 * r19;
    r48 = fma(r8, r48, r29 * r49);
    r49 = r48 * r51;
    r49 = r5 <= r38 ? r48 : r49;
    r54 = r48 * r40;
    r49 = r35 < r39 ? r54 : r49;
    r54 = r48 * r43;
    r49 = r35 < r42 ? r54 : r49;
    r49 = r35 < r37 ? r48 : r49;
    r54 = r37 * r49;
    r48 = fma(r48, r44, r45 * r54);
    r48 = r37 * r48;
    r48 = r48 * r46;
    r48 = r5 <= r28 ? r10 : r48;
    r54 = fma(r19, r48, r30);
    r50 = r0 * r8;
    r50 = r50 * r47;
    r54 = fma(r29, r50, r54);
    r48 = fma(r4, r48, r53);
    r50 = r2 * r8;
    r50 = r50 * r47;
    r48 = fma(r29, r50, r48);
    WriteIdx2<1024, double, double, double2>(out_focal_jac,
                                             0 * out_focal_jac_num_alloc,
                                             global_thread_idx,
                                             r54,
                                             r48);
    r50 = r1 * r21;
    r50 = r50 * r7;
    r50 = r50 * r19;
    r60 = r34 * r7;
    r60 = fma(r25, r60, r29 * r50);
    r50 = r60 * r51;
    r50 = r5 <= r38 ? r60 : r50;
    r25 = r60 * r40;
    r50 = r35 < r39 ? r25 : r50;
    r25 = r60 * r43;
    r50 = r35 < r42 ? r25 : r50;
    r50 = r35 < r37 ? r60 : r50;
    r25 = r37 * r50;
    r25 = fma(r45, r25, r60 * r44);
    r25 = r37 * r25;
    r25 = r25 * r46;
    r25 = r5 <= r28 ? r10 : r25;
    r60 = fma(r19, r25, r30);
    r62 = r1 * r7;
    r62 = r62 * r47;
    r60 = fma(r29, r62, r60);
    r25 = fma(r4, r25, r53);
    r62 = r34 * r7;
    r62 = r62 * r47;
    r25 = fma(r29, r62, r25);
    WriteIdx2<1024, double, double, double2>(out_focal_jac,
                                             2 * out_focal_jac_num_alloc,
                                             global_thread_idx,
                                             r60,
                                             r25);
    r62 = r4 * r48;
    r62 = fma(r54, r61, r57 * r62);
    r63 = r4 * r25;
    r63 = fma(r60, r61, r57 * r63);
    WriteSum2<double, double>((double*)inout_shared, r62, r63);
  };
  FlushSumShared<2, double>(out_focal_njtr,
                            0 * out_focal_njtr_num_alloc,
                            focal_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r63 = fma(r48, r48, r54 * r54);
    r62 = fma(r25, r25, r60 * r60);
    WriteSum2<double, double>((double*)inout_shared, r63, r62);
  };
  FlushSumShared<2, double>(out_focal_precond_diag,
                            0 * out_focal_precond_diag_num_alloc,
                            focal_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r60 = fma(r48, r25, r54 * r60);
    WriteSum1<double, double>((double*)inout_shared, r60);
  };
  FlushSumShared<1, double>(out_focal_precond_tril,
                            0 * out_focal_precond_tril_num_alloc,
                            focal_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r60 = r21 * r19;
    r24 = fma(r31, r27, r24 * r33);
    r54 = r26 * r13;
    r62 = r31 * r6;
    r62 = r62 * r20;
    r62 = fma(r9, r62, r29 * r54);
    r54 = fma(r0, r62, r1 * r24);
    r62 = fma(r2, r62, r34 * r24);
    r60 = fma(r62, r14, r54 * r60);
    r24 = r60 * r51;
    r24 = r5 <= r38 ? r60 : r24;
    r63 = r60 * r40;
    r24 = r35 < r39 ? r63 : r24;
    r63 = r60 * r43;
    r24 = r35 < r42 ? r63 : r24;
    r24 = r35 < r37 ? r60 : r24;
    r63 = r37 * r24;
    r60 = fma(r60, r44, r45 * r63);
    r60 = r37 * r60;
    r60 = r60 * r46;
    r60 = r5 <= r28 ? r10 : r60;
    r63 = fma(r19, r60, r30);
    r63 = fma(r47, r54, r63);
    r60 = fma(r4, r60, r53);
    r60 = fma(r47, r62, r60);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             0 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r63,
                                             r60);
    r62 = r21 * r19;
    r54 = r26 * r22;
    r64 = r32 * r6;
    r64 = r64 * r20;
    r64 = fma(r9, r64, r29 * r54);
    r18 = fma(r32, r27, r18 * r33);
    r54 = fma(r1, r18, r0 * r64);
    r18 = fma(r34, r18, r2 * r64);
    r62 = fma(r18, r14, r54 * r62);
    r64 = r62 * r51;
    r64 = r5 <= r38 ? r62 : r64;
    r65 = r62 * r40;
    r64 = r35 < r39 ? r65 : r64;
    r65 = r62 * r43;
    r64 = r35 < r42 ? r65 : r64;
    r64 = r35 < r37 ? r62 : r64;
    r65 = r37 * r64;
    r65 = fma(r45, r65, r62 * r44);
    r65 = r37 * r65;
    r65 = r65 * r46;
    r65 = r5 <= r28 ? r10 : r65;
    r62 = fma(r19, r65, r30);
    r62 = fma(r47, r54, r62);
    r65 = fma(r4, r65, r53);
    r65 = fma(r47, r18, r65);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             2 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r62,
                                             r65);
    r18 = r26 * r3;
    r54 = r17 * r6;
    r54 = r54 * r20;
    r54 = fma(r9, r54, r29 * r18);
    r33 = fma(r23, r33, r17 * r27);
    r23 = fma(r1, r33, r0 * r54);
    r30 = fma(r47, r23, r30);
    r27 = r21 * r19;
    r33 = fma(r34, r33, r2 * r54);
    r14 = fma(r33, r14, r23 * r27);
    r51 = r14 * r51;
    r51 = r5 <= r38 ? r14 : r51;
    r40 = r14 * r40;
    r51 = r35 < r39 ? r40 : r51;
    r43 = r14 * r43;
    r51 = r35 < r42 ? r43 : r51;
    r51 = r35 < r37 ? r14 : r51;
    r35 = r37 * r51;
    r35 = fma(r45, r35, r14 * r44);
    r35 = r37 * r35;
    r35 = r35 * r46;
    r35 = r5 <= r28 ? r10 : r35;
    r30 = fma(r19, r35, r30);
    r33 = fma(r47, r33, r53);
    r33 = fma(r4, r35, r33);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             4 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r30,
                                             r33);
    r35 = r4 * r60;
    r35 = fma(r63, r61, r57 * r35);
    r53 = r4 * r65;
    r53 = fma(r57, r53, r62 * r61);
    WriteSum2<double, double>((double*)inout_shared, r35, r53);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r53 = r4 * r33;
    r53 = fma(r57, r53, r30 * r61);
    WriteSum1<double, double>((double*)inout_shared, r53);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r53 = fma(r63, r63, r60 * r60);
    r61 = fma(r65, r65, r62 * r62);
    WriteSum2<double, double>((double*)inout_shared, r53, r61);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r61 = fma(r33, r33, r30 * r30);
    WriteSum1<double, double>((double*)inout_shared, r61);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r61 = fma(r60, r65, r63 * r62);
    r63 = fma(r63, r30, r60 * r33);
    WriteSum2<double, double>((double*)inout_shared, r61, r63);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = fma(r65, r33, r62 * r30);
    WriteSum1<double, double>((double*)inout_shared, r30);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void PinholeSplitFixedRotationFixedPrincipalPointResJacFirst(
    double* rotation,
    unsigned int rotation_num_alloc,
    double* translation,
    unsigned int translation_num_alloc,
    SharedIndex* translation_indices,
    double* focal,
    unsigned int focal_num_alloc,
    SharedIndex* focal_indices,
    double* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* weight_loss,
    unsigned int weight_loss_num_alloc,
    double* principal_point,
    unsigned int principal_point_num_alloc,
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
    double* out_focal_jac,
    unsigned int out_focal_jac_num_alloc,
    double* const out_focal_njtr,
    unsigned int out_focal_njtr_num_alloc,
    double* const out_focal_precond_diag,
    unsigned int out_focal_precond_diag_num_alloc,
    double* const out_focal_precond_tril,
    unsigned int out_focal_precond_tril_num_alloc,
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
  PinholeSplitFixedRotationFixedPrincipalPointResJacFirstKernel<<<n_blocks,
                                                                  1024>>>(
      rotation,
      rotation_num_alloc,
      translation,
      translation_num_alloc,
      translation_indices,
      focal,
      focal_num_alloc,
      focal_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      weight_loss,
      weight_loss_num_alloc,
      principal_point,
      principal_point_num_alloc,
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
      out_focal_jac,
      out_focal_jac_num_alloc,
      out_focal_njtr,
      out_focal_njtr_num_alloc,
      out_focal_precond_diag,
      out_focal_precond_diag_num_alloc,
      out_focal_precond_tril,
      out_focal_precond_tril_num_alloc,
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