#include "kernel_pinhole_fixed_rotation_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeFixedRotationFixedPointResJacKernel(
        double* rotation,
        unsigned int rotation_num_alloc,
        double* translation,
        unsigned int translation_num_alloc,
        SharedIndex* translation_indices,
        double* calib,
        unsigned int calib_num_alloc,
        SharedIndex* calib_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* weight_loss,
        unsigned int weight_loss_num_alloc,
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
        double* out_calib_jac,
        unsigned int out_calib_jac_num_alloc,
        double* const out_calib_njtr,
        unsigned int out_calib_njtr_num_alloc,
        double* const out_calib_precond_diag,
        unsigned int out_calib_precond_diag_num_alloc,
        double* const out_calib_precond_tril,
        unsigned int out_calib_precond_tril_num_alloc,
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

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53;

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
    r3 = 1.00000000000000008e-15;
  };
  LoadShared<1, double, double>(translation,
                                2 * translation_num_alloc,
                                translation_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, translation_indices_loc[threadIdx.x].target, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx1<1024, double, double, double>(
        point, 2 * point_num_alloc, global_thread_idx, r8);
    r9 = -2.00000000000000000e+00;
    ReadIdx2<1024, double, double, double2>(
        rotation, 0 * rotation_num_alloc, global_thread_idx, r10, r11);
    r12 = r11 * r11;
    r12 = r9 * r12;
    r13 = 1.00000000000000000e+00;
    r14 = r10 * r10;
    r14 = fma(r9, r14, r13);
    r15 = r12 + r14;
    r15 = fma(r8, r15, r7);
    ReadIdx2<1024, double, double, double2>(
        point, 0 * point_num_alloc, global_thread_idx, r7, r16);
    ReadIdx2<1024, double, double, double2>(
        rotation, 2 * rotation_num_alloc, global_thread_idx, r17, r18);
    r19 = r10 * r17;
    r20 = 2.00000000000000000e+00;
    r19 = r19 * r20;
    r21 = r18 * r9;
    r22 = fma(r11, r21, r19);
    r23 = r10 * r18;
    r24 = r11 * r17;
    r24 = r24 * r20;
    r23 = fma(r20, r23, r24);
    r15 = fma(r7, r22, r15);
    r15 = fma(r16, r23, r15);
    r23 = copysign(1.0, r15);
    r23 = fma(r3, r23, r15);
    r15 = 1.0 / r23;
  };
  LoadShared<2, double, double>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r22, r25);
  };
  __syncthreads();
  LoadShared<2, double, double>(translation,
                                0 * translation_num_alloc,
                                translation_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        translation_indices_loc[threadIdx.x].target,
                        r26,
                        r27);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r28 = r17 * r17;
    r28 = r9 * r28;
    r14 = r28 + r14;
    r14 = fma(r16, r14, r27);
    r24 = fma(r10, r21, r24);
    r27 = r17 * r18;
    r9 = r10 * r11;
    r9 = r9 * r20;
    r27 = fma(r20, r27, r9);
    r14 = fma(r8, r24, r14);
    r14 = fma(r7, r27, r14);
    r27 = r25 * r14;
    r5 = fma(r15, r27, r5);
    r4 = fma(r4, r6, r2);
    r28 = r13 + r28;
    r28 = r28 + r12;
    r28 = fma(r7, r28, r26);
    r7 = r11 * r18;
    r7 = fma(r20, r7, r19);
    r21 = fma(r17, r21, r9);
    r28 = fma(r8, r7, r28);
    r28 = fma(r16, r21, r28);
    r21 = r22 * r28;
    r4 = fma(r15, r21, r4);
    r16 = fma(r0, r4, r1 * r5);
    ReadIdx1<1024, double, double, double>(
        weight_loss, 6 * weight_loss_num_alloc, global_thread_idx, r7);
    r8 = 0.00000000000000000e+00;
    r7 = fmax(r7, r8);
    r9 = sqrt(r7);
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 2 * weight_loss_num_alloc, global_thread_idx, r19, r26);
    r4 = fma(r19, r4, r26 * r5);
    r5 = fma(r16, r16, r4 * r4);
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 4 * weight_loss_num_alloc, global_thread_idx, r12, r2);
    r24 = 5.00000000000000000e-01;
    r2 = fmax(r2, r3);
    r29 = r2 * r2;
    r30 = r20 * r2;
    r31 = fmax(r3, r5);
    r32 = sqrt(r31);
    r33 = r6 * r2;
    r33 = fma(r2, r33, r32 * r30);
    r33 = r5 <= r29 ? r5 : r33;
    r30 = 2.50000000000000000e+00;
    r32 = r2 * r2;
    r34 = 1.0 / r29;
    r34 = fma(r5, r34, r13);
    r35 = log(r34);
    r32 = r32 * r35;
    r33 = r12 < r30 ? r32 : r33;
    r32 = 1.50000000000000000e+00;
    r35 = r20 * r2;
    r36 = sqrt(r34);
    r36 = r6 + r36;
    r35 = r35 * r2;
    r35 = r35 * r36;
    r33 = r12 < r32 ? r35 : r33;
    r33 = r12 < r24 ? r5 : r33;
    r35 = fmax(r8, r33);
    r36 = 1.0 / r31;
    r36 = r7 * r36;
    r37 = r35 * r36;
    r38 = sqrt(r37);
    r38 = r5 <= r3 ? r9 : r38;
    r9 = r16 * r38;
    r39 = r4 * r38;
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r9, r39);
    r35 = r7 * r35;
    r7 = -5.00000000000000000e-01;
    r39 = -1.00000000000000008e-15;
    r39 = r39 + r5;
    r39 = copysign(1.0, r39);
    r39 = r13 + r39;
    r9 = r31 * r31;
    r9 = 1.0 / r9;
    r35 = r35 * r7;
    r35 = r35 * r39;
    r35 = r35 * r9;
    r9 = r22 * r0;
    r7 = r20 * r15;
    r40 = r16 * r7;
    r41 = r22 * r19;
    r41 = r41 * r4;
    r41 = fma(r7, r41, r40 * r9);
    r9 = r24 * r2;
    r31 = rsqrt(r31);
    r9 = r9 * r39;
    r9 = r9 * r31;
    r31 = r41 * r9;
    r31 = r5 <= r29 ? r41 : r31;
    r39 = 1.0 / r34;
    r42 = r41 * r39;
    r31 = r12 < r30 ? r42 : r31;
    r34 = rsqrt(r34);
    r42 = r41 * r34;
    r31 = r12 < r32 ? r42 : r31;
    r31 = r12 < r24 ? r41 : r31;
    r42 = r24 * r31;
    r33 = copysign(1.0, r33);
    r33 = r13 + r33;
    r36 = r33 * r36;
    r42 = fma(r36, r42, r41 * r35);
    r42 = r24 * r42;
    r37 = rsqrt(r37);
    r42 = r42 * r37;
    r42 = r5 <= r3 ? r8 : r42;
    r41 = r6 * r16;
    r33 = 2.50000000000000000e-01;
    r13 = r5 <= r29 ? r8 : r8;
    r13 = r12 < r30 ? r8 : r13;
    r13 = r12 < r32 ? r8 : r13;
    r13 = r12 < r24 ? r8 : r13;
    r13 = r33 * r13;
    r13 = r13 * r37;
    r13 = r13 * r36;
    r13 = r5 <= r3 ? r8 : r13;
    r41 = r41 * r13;
    r33 = fma(r16, r42, r41);
    r43 = r22 * r0;
    r43 = r43 * r38;
    r33 = fma(r15, r43, r33);
    r43 = r6 * r4;
    r43 = r43 * r13;
    r42 = fma(r4, r42, r43);
    r13 = r22 * r19;
    r13 = r13 * r38;
    r42 = fma(r15, r13, r42);
    WriteIdx2<1024, double, double, double2>(out_translation_jac,
                                             0 * out_translation_jac_num_alloc,
                                             global_thread_idx,
                                             r33,
                                             r42);
    r13 = r1 * r40;
    r44 = r25 * r26;
    r44 = r44 * r4;
    r44 = fma(r7, r44, r25 * r13);
    r45 = r44 * r9;
    r45 = r5 <= r29 ? r44 : r45;
    r46 = r44 * r39;
    r45 = r12 < r30 ? r46 : r45;
    r46 = r44 * r34;
    r45 = r12 < r32 ? r46 : r45;
    r45 = r12 < r24 ? r44 : r45;
    r46 = r24 * r45;
    r44 = fma(r44, r35, r36 * r46);
    r44 = r24 * r44;
    r44 = r44 * r37;
    r44 = r5 <= r3 ? r8 : r44;
    r46 = fma(r16, r44, r41);
    r47 = r25 * r1;
    r47 = r47 * r38;
    r46 = fma(r15, r47, r46);
    r44 = fma(r4, r44, r43);
    r47 = r25 * r26;
    r47 = r47 * r38;
    r44 = fma(r15, r47, r44);
    WriteIdx2<1024, double, double, double2>(out_translation_jac,
                                             2 * out_translation_jac_num_alloc,
                                             global_thread_idx,
                                             r46,
                                             r44);
    r47 = r20 * r4;
    r23 = r23 * r23;
    r23 = 1.0 / r23;
    r23 = r6 * r23;
    r27 = r27 * r23;
    r48 = r19 * r21;
    r48 = fma(r23, r48, r26 * r27);
    r49 = r20 * r16;
    r50 = r0 * r21;
    r50 = fma(r23, r50, r1 * r27);
    r49 = fma(r50, r49, r48 * r47);
    r47 = r49 * r9;
    r47 = r5 <= r29 ? r49 : r47;
    r27 = r49 * r39;
    r47 = r12 < r30 ? r27 : r47;
    r27 = r49 * r34;
    r47 = r12 < r32 ? r27 : r47;
    r47 = r12 < r24 ? r49 : r47;
    r27 = r24 * r47;
    r27 = fma(r36, r27, r49 * r35);
    r27 = r24 * r27;
    r27 = r27 * r37;
    r27 = r5 <= r3 ? r8 : r27;
    r49 = fma(r16, r27, r41);
    r49 = fma(r38, r50, r49);
    r27 = fma(r4, r27, r43);
    r27 = fma(r38, r48, r27);
    WriteIdx2<1024, double, double, double2>(out_translation_jac,
                                             4 * out_translation_jac_num_alloc,
                                             global_thread_idx,
                                             r49,
                                             r27);
    r48 = r16 * r33;
    r50 = r6 * r38;
    r23 = r4 * r50;
    r48 = fma(r42, r23, r50 * r48);
    r51 = r16 * r46;
    r51 = fma(r50, r51, r44 * r23);
    WriteSum2<double, double>((double*)inout_shared, r48, r51);
  };
  FlushSumShared<2, double>(out_translation_njtr,
                            0 * out_translation_njtr_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r51 = r16 * r49;
    r51 = fma(r50, r51, r27 * r23);
    WriteSum1<double, double>((double*)inout_shared, r51);
  };
  FlushSumShared<1, double>(out_translation_njtr,
                            2 * out_translation_njtr_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r51 = fma(r42, r42, r33 * r33);
    r48 = fma(r46, r46, r44 * r44);
    WriteSum2<double, double>((double*)inout_shared, r51, r48);
  };
  FlushSumShared<2, double>(out_translation_precond_diag,
                            0 * out_translation_precond_diag_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = fma(r49, r49, r27 * r27);
    WriteSum1<double, double>((double*)inout_shared, r48);
  };
  FlushSumShared<1, double>(out_translation_precond_diag,
                            2 * out_translation_precond_diag_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r48 = fma(r33, r46, r42 * r44);
    r42 = fma(r42, r27, r33 * r49);
    WriteSum2<double, double>((double*)inout_shared, r48, r42);
  };
  FlushSumShared<2, double>(out_translation_precond_tril,
                            0 * out_translation_precond_tril_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fma(r44, r27, r46 * r49);
    WriteSum1<double, double>((double*)inout_shared, r27);
  };
  FlushSumShared<1, double>(out_translation_precond_tril,
                            2 * out_translation_precond_tril_num_alloc,
                            translation_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = r19 * r28;
    r27 = r27 * r4;
    r44 = r0 * r28;
    r44 = fma(r40, r44, r7 * r27);
    r27 = r44 * r9;
    r27 = r5 <= r29 ? r44 : r27;
    r40 = r44 * r39;
    r27 = r12 < r30 ? r40 : r27;
    r40 = r44 * r34;
    r27 = r12 < r32 ? r40 : r27;
    r27 = r12 < r24 ? r44 : r27;
    r40 = r24 * r27;
    r40 = fma(r36, r40, r44 * r35);
    r40 = r24 * r40;
    r40 = r40 * r37;
    r40 = r5 <= r3 ? r8 : r40;
    r44 = fma(r16, r40, r41);
    r42 = r0 * r28;
    r42 = r42 * r38;
    r44 = fma(r15, r42, r44);
    r40 = fma(r4, r40, r43);
    r42 = r19 * r28;
    r42 = r42 * r38;
    r40 = fma(r15, r42, r40);
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             0 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r44,
                                             r40);
    r42 = r26 * r14;
    r42 = r42 * r4;
    r13 = fma(r14, r13, r7 * r42);
    r42 = r13 * r9;
    r42 = r5 <= r29 ? r13 : r42;
    r7 = r13 * r39;
    r42 = r12 < r30 ? r7 : r42;
    r7 = r13 * r34;
    r42 = r12 < r32 ? r7 : r42;
    r42 = r12 < r24 ? r13 : r42;
    r7 = r24 * r42;
    r13 = fma(r13, r35, r36 * r7);
    r13 = r24 * r13;
    r13 = r13 * r37;
    r13 = r5 <= r3 ? r8 : r13;
    r7 = fma(r16, r13, r41);
    r48 = r1 * r14;
    r48 = r48 * r38;
    r7 = fma(r15, r48, r7);
    r13 = fma(r4, r13, r43);
    r48 = r26 * r14;
    r48 = r48 * r38;
    r13 = fma(r15, r48, r13);
    WriteIdx2<1024, double, double, double2>(
        out_calib_jac, 2 * out_calib_jac_num_alloc, global_thread_idx, r7, r13);
    r48 = fma(r0, r38, r41);
    r15 = r19 * r20;
    r51 = r0 * r20;
    r51 = fma(r16, r51, r4 * r15);
    r15 = r51 * r9;
    r15 = r5 <= r29 ? r51 : r15;
    r52 = r51 * r39;
    r15 = r12 < r30 ? r52 : r15;
    r52 = r51 * r34;
    r15 = r12 < r32 ? r52 : r15;
    r15 = r12 < r24 ? r51 : r15;
    r52 = r24 * r15;
    r52 = fma(r36, r52, r51 * r35);
    r52 = r24 * r52;
    r52 = r52 * r37;
    r52 = r5 <= r3 ? r8 : r52;
    r48 = fma(r16, r52, r48);
    r51 = fma(r19, r38, r43);
    r51 = fma(r4, r52, r51);
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             4 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r48,
                                             r51);
    r52 = r26 * r20;
    r53 = r1 * r20;
    r53 = fma(r16, r53, r4 * r52);
    r9 = r53 * r9;
    r9 = r5 <= r29 ? r53 : r9;
    r39 = r53 * r39;
    r9 = r12 < r30 ? r39 : r9;
    r34 = r53 * r34;
    r9 = r12 < r32 ? r34 : r9;
    r9 = r12 < r24 ? r53 : r9;
    r12 = r24 * r9;
    r12 = fma(r36, r12, r53 * r35);
    r12 = r24 * r12;
    r12 = r12 * r37;
    r12 = r5 <= r3 ? r8 : r12;
    r41 = fma(r16, r12, r41);
    r41 = fma(r1, r38, r41);
    r12 = fma(r4, r12, r43);
    r12 = fma(r26, r38, r12);
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             6 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r41,
                                             r12);
    r38 = r16 * r44;
    r38 = fma(r40, r23, r50 * r38);
    r43 = r16 * r7;
    r43 = fma(r50, r43, r13 * r23);
    WriteSum2<double, double>((double*)inout_shared, r38, r43);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            0 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = r16 * r48;
    r43 = fma(r51, r23, r50 * r43);
    r38 = r16 * r41;
    r38 = fma(r50, r38, r12 * r23);
    WriteSum2<double, double>((double*)inout_shared, r43, r38);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            2 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r38 = fma(r40, r40, r44 * r44);
    r43 = fma(r7, r7, r13 * r13);
    WriteSum2<double, double>((double*)inout_shared, r38, r43);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            0 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = fma(r51, r51, r48 * r48);
    r38 = fma(r41, r41, r12 * r12);
    WriteSum2<double, double>((double*)inout_shared, r43, r38);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            2 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r38 = fma(r40, r13, r44 * r7);
    r43 = fma(r40, r51, r44 * r48);
    WriteSum2<double, double>((double*)inout_shared, r38, r43);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            0 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = fma(r40, r12, r44 * r41);
    r43 = fma(r7, r48, r13 * r51);
    WriteSum2<double, double>((double*)inout_shared, r40, r43);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            2 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = fma(r13, r12, r7 * r41);
    r12 = fma(r48, r41, r51 * r12);
    WriteSum2<double, double>((double*)inout_shared, r13, r12);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            4 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
}

void PinholeFixedRotationFixedPointResJac(
    double* rotation,
    unsigned int rotation_num_alloc,
    double* translation,
    unsigned int translation_num_alloc,
    SharedIndex* translation_indices,
    double* calib,
    unsigned int calib_num_alloc,
    SharedIndex* calib_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* weight_loss,
    unsigned int weight_loss_num_alloc,
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
    double* out_calib_jac,
    unsigned int out_calib_jac_num_alloc,
    double* const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc,
    double* const out_calib_precond_diag,
    unsigned int out_calib_precond_diag_num_alloc,
    double* const out_calib_precond_tril,
    unsigned int out_calib_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeFixedRotationFixedPointResJacKernel<<<n_blocks, 1024>>>(
      rotation,
      rotation_num_alloc,
      translation,
      translation_num_alloc,
      translation_indices,
      calib,
      calib_num_alloc,
      calib_indices,
      pixel,
      pixel_num_alloc,
      weight_loss,
      weight_loss_num_alloc,
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
      out_calib_jac,
      out_calib_jac_num_alloc,
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      out_calib_precond_diag,
      out_calib_precond_diag_num_alloc,
      out_calib_precond_tril,
      out_calib_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar