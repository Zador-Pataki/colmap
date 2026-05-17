#include "kernel_pinhole_fixed_pose_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeFixedPoseResJacKernel(double* calib,
                                 unsigned int calib_num_alloc,
                                 SharedIndex* calib_indices,
                                 double* point,
                                 unsigned int point_num_alloc,
                                 SharedIndex* point_indices,
                                 double* pixel,
                                 unsigned int pixel_num_alloc,
                                 double* weight_loss,
                                 unsigned int weight_loss_num_alloc,
                                 double* pose,
                                 unsigned int pose_num_alloc,
                                 double* out_res,
                                 unsigned int out_res_num_alloc,
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

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65;

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
    r4 = fma(r4, r6, r2);
  };
  LoadShared<2, double, double>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r2, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        pose, 4 * pose_num_alloc, global_thread_idx, r8, r9);
  };
  LoadShared<2, double, double>(
      point, 0 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r10, r11);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        pose, 0 * pose_num_alloc, global_thread_idx, r12, r13);
    r14 = r12 * r13;
    r15 = 2.00000000000000000e+00;
    r14 = r14 * r15;
    ReadIdx2<1024, double, double, double2>(
        pose, 2 * pose_num_alloc, global_thread_idx, r16, r17);
    r18 = -2.00000000000000000e+00;
    r19 = r17 * r18;
    r20 = fma(r16, r19, r14);
    r8 = fma(r11, r20, r8);
  };
  LoadShared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r21);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r22 = r12 * r16;
    r22 = r22 * r15;
    r23 = r13 * r17;
    r23 = fma(r15, r23, r22);
    r24 = r16 * r16;
    r24 = r18 * r24;
    r25 = 1.00000000000000000e+00;
    r26 = r13 * r13;
    r26 = fma(r18, r26, r25);
    r27 = r24 + r26;
    r8 = fma(r21, r23, r8);
    r8 = fma(r10, r27, r8);
    r28 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        pose, 6 * pose_num_alloc, global_thread_idx, r29);
    r30 = r13 * r16;
    r30 = r30 * r15;
    r31 = r12 * r17;
    r31 = fma(r15, r31, r30);
    r29 = fma(r11, r31, r29);
    r22 = fma(r13, r19, r22);
    r32 = r12 * r12;
    r32 = r18 * r32;
    r26 = r32 + r26;
    r29 = fma(r10, r22, r29);
    r29 = fma(r21, r26, r29);
    r18 = copysign(1.0, r29);
    r18 = fma(r28, r18, r29);
    r29 = 1.0 / r18;
    r33 = r8 * r29;
    r4 = fma(r2, r33, r4);
    r5 = fma(r5, r6, r3);
    r3 = r16 * r17;
    r3 = fma(r15, r3, r14);
    r10 = fma(r10, r3, r9);
    r19 = fma(r12, r19, r30);
    r24 = r25 + r24;
    r24 = r24 + r32;
    r10 = fma(r21, r19, r10);
    r10 = fma(r11, r24, r10);
    r11 = r7 * r10;
    r5 = fma(r29, r11, r5);
    r21 = fma(r1, r5, r0 * r4);
    ReadIdx1<1024, double, double, double>(
        weight_loss, 6 * weight_loss_num_alloc, global_thread_idx, r32);
    r30 = 0.00000000000000000e+00;
    r32 = fmax(r32, r30);
    r9 = sqrt(r32);
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 2 * weight_loss_num_alloc, global_thread_idx, r14, r34);
    r5 = fma(r34, r5, r14 * r4);
    r4 = fma(r21, r21, r5 * r5);
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 4 * weight_loss_num_alloc, global_thread_idx, r35, r36);
    r37 = 5.00000000000000000e-01;
    r36 = fmax(r36, r28);
    r38 = r36 * r36;
    r39 = r15 * r36;
    r40 = fmax(r28, r4);
    r41 = sqrt(r40);
    r42 = r6 * r36;
    r42 = fma(r36, r42, r41 * r39);
    r42 = r4 <= r38 ? r4 : r42;
    r39 = 2.50000000000000000e+00;
    r41 = r36 * r36;
    r43 = 1.0 / r38;
    r43 = fma(r4, r43, r25);
    r44 = log(r43);
    r41 = r41 * r44;
    r42 = r35 < r39 ? r41 : r42;
    r41 = 1.50000000000000000e+00;
    r44 = r15 * r36;
    r45 = sqrt(r43);
    r45 = r6 + r45;
    r44 = r44 * r36;
    r44 = r44 * r45;
    r42 = r35 < r41 ? r44 : r42;
    r42 = r35 < r37 ? r4 : r42;
    r44 = fmax(r30, r42);
    r45 = 1.0 / r40;
    r45 = r32 * r45;
    r46 = r44 * r45;
    r47 = sqrt(r46);
    r47 = r4 <= r28 ? r9 : r47;
    r9 = r21 * r47;
    r48 = r5 * r47;
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r9, r48);
    r48 = r6 * r21;
    r9 = 2.50000000000000000e-01;
    r49 = r4 <= r38 ? r30 : r30;
    r49 = r35 < r39 ? r30 : r49;
    r49 = r35 < r41 ? r30 : r49;
    r49 = r35 < r37 ? r30 : r49;
    r49 = r9 * r49;
    r46 = rsqrt(r46);
    r42 = copysign(1.0, r42);
    r42 = r25 + r42;
    r45 = r42 * r45;
    r49 = r49 * r46;
    r49 = r49 * r45;
    r49 = r4 <= r28 ? r30 : r49;
    r48 = r48 * r49;
    r42 = r0 * r15;
    r42 = r42 * r21;
    r9 = r14 * r15;
    r9 = r9 * r5;
    r9 = fma(r33, r9, r33 * r42);
    r50 = r37 * r36;
    r51 = -1.00000000000000008e-15;
    r51 = r51 + r4;
    r51 = copysign(1.0, r51);
    r51 = r25 + r51;
    r25 = rsqrt(r40);
    r50 = r50 * r51;
    r50 = r50 * r25;
    r25 = r9 * r50;
    r25 = r4 <= r38 ? r9 : r25;
    r52 = 1.0 / r43;
    r53 = r9 * r52;
    r25 = r35 < r39 ? r53 : r25;
    r43 = rsqrt(r43);
    r53 = r9 * r43;
    r25 = r35 < r41 ? r53 : r25;
    r25 = r35 < r37 ? r9 : r25;
    r53 = r37 * r25;
    r44 = r32 * r44;
    r32 = -5.00000000000000000e-01;
    r40 = r40 * r40;
    r40 = 1.0 / r40;
    r44 = r44 * r32;
    r44 = r44 * r51;
    r44 = r44 * r40;
    r9 = fma(r9, r44, r45 * r53);
    r9 = r37 * r9;
    r9 = r9 * r46;
    r9 = r4 <= r28 ? r30 : r9;
    r53 = fma(r21, r9, r48);
    r33 = r47 * r33;
    r53 = fma(r0, r33, r53);
    r40 = r6 * r5;
    r40 = r40 * r49;
    r9 = fma(r5, r9, r40);
    r9 = fma(r14, r33, r9);
    WriteIdx2<1024, double, double, double2>(
        out_calib_jac, 0 * out_calib_jac_num_alloc, global_thread_idx, r53, r9);
    r33 = r1 * r10;
    r49 = r15 * r21;
    r33 = r33 * r29;
    r51 = r34 * r15;
    r51 = r51 * r10;
    r51 = r51 * r5;
    r51 = fma(r29, r51, r49 * r33);
    r33 = r51 * r50;
    r33 = r4 <= r38 ? r51 : r33;
    r32 = r51 * r52;
    r33 = r35 < r39 ? r32 : r33;
    r32 = r51 * r43;
    r33 = r35 < r41 ? r32 : r33;
    r33 = r35 < r37 ? r51 : r33;
    r32 = r37 * r33;
    r32 = fma(r45, r32, r51 * r44);
    r32 = r37 * r32;
    r32 = r32 * r46;
    r32 = r4 <= r28 ? r30 : r32;
    r51 = fma(r21, r32, r48);
    r54 = r1 * r10;
    r54 = r54 * r47;
    r51 = fma(r29, r54, r51);
    r32 = fma(r5, r32, r40);
    r54 = r34 * r10;
    r54 = r54 * r47;
    r32 = fma(r29, r54, r32);
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             2 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r51,
                                             r32);
    r54 = fma(r0, r47, r48);
    r55 = r14 * r15;
    r55 = fma(r5, r55, r42);
    r42 = r55 * r50;
    r42 = r4 <= r38 ? r55 : r42;
    r56 = r55 * r52;
    r42 = r35 < r39 ? r56 : r42;
    r56 = r55 * r43;
    r42 = r35 < r41 ? r56 : r42;
    r42 = r35 < r37 ? r55 : r42;
    r56 = r37 * r42;
    r55 = fma(r55, r44, r45 * r56);
    r55 = r37 * r55;
    r55 = r55 * r46;
    r55 = r4 <= r28 ? r30 : r55;
    r54 = fma(r21, r55, r54);
    r56 = fma(r14, r47, r40);
    r56 = fma(r5, r55, r56);
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             4 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r54,
                                             r56);
    r55 = r34 * r15;
    r55 = fma(r1, r49, r5 * r55);
    r57 = r55 * r50;
    r57 = r4 <= r38 ? r55 : r57;
    r58 = r55 * r52;
    r57 = r35 < r39 ? r58 : r57;
    r58 = r55 * r43;
    r57 = r35 < r41 ? r58 : r57;
    r57 = r35 < r37 ? r55 : r57;
    r58 = r37 * r57;
    r58 = fma(r45, r58, r55 * r44);
    r58 = r37 * r58;
    r58 = r58 * r46;
    r58 = r4 <= r28 ? r30 : r58;
    r55 = fma(r21, r58, r48);
    r55 = fma(r1, r47, r55);
    r58 = fma(r5, r58, r40);
    r58 = fma(r34, r47, r58);
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             6 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r55,
                                             r58);
    r59 = r21 * r53;
    r60 = r6 * r47;
    r61 = r5 * r60;
    r59 = fma(r9, r61, r60 * r59);
    r62 = r21 * r51;
    r62 = fma(r32, r61, r60 * r62);
    WriteSum2<double, double>((double*)inout_shared, r59, r62);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            0 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r62 = r21 * r54;
    r62 = fma(r60, r62, r56 * r61);
    r59 = r21 * r55;
    r59 = fma(r58, r61, r60 * r59);
    WriteSum2<double, double>((double*)inout_shared, r62, r59);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            2 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r59 = fma(r53, r53, r9 * r9);
    r62 = fma(r51, r51, r32 * r32);
    WriteSum2<double, double>((double*)inout_shared, r59, r62);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            0 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r62 = fma(r54, r54, r56 * r56);
    r59 = fma(r58, r58, r55 * r55);
    WriteSum2<double, double>((double*)inout_shared, r62, r59);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            2 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r59 = fma(r9, r32, r53 * r51);
    r62 = fma(r9, r56, r53 * r54);
    WriteSum2<double, double>((double*)inout_shared, r59, r62);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            0 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r9 = fma(r53, r55, r9 * r58);
    r62 = fma(r51, r54, r32 * r56);
    WriteSum2<double, double>((double*)inout_shared, r9, r62);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            2 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fma(r51, r55, r32 * r58);
    r58 = fma(r54, r55, r56 * r58);
    WriteSum2<double, double>((double*)inout_shared, r32, r58);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            4 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r58 = r2 * r27;
    r8 = r2 * r8;
    r18 = r18 * r18;
    r18 = 1.0 / r18;
    r8 = r8 * r6;
    r8 = r8 * r18;
    r58 = fma(r22, r8, r29 * r58);
    r32 = r7 * r3;
    r56 = r22 * r6;
    r56 = r56 * r18;
    r56 = fma(r11, r56, r29 * r32);
    r32 = fma(r1, r56, r0 * r58);
    r62 = fma(r47, r32, r48);
    r9 = r15 * r5;
    r56 = fma(r34, r56, r14 * r58);
    r32 = fma(r32, r49, r56 * r9);
    r9 = r32 * r50;
    r9 = r4 <= r38 ? r32 : r9;
    r58 = r32 * r52;
    r9 = r35 < r39 ? r58 : r9;
    r58 = r32 * r43;
    r9 = r35 < r41 ? r58 : r9;
    r9 = r35 < r37 ? r32 : r9;
    r58 = r37 * r9;
    r32 = fma(r32, r44, r45 * r58);
    r32 = r37 * r32;
    r32 = r32 * r46;
    r32 = r4 <= r28 ? r30 : r32;
    r62 = fma(r21, r32, r62);
    r56 = fma(r47, r56, r40);
    r56 = fma(r5, r32, r56);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             0 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r62,
                                             r56);
    r32 = r7 * r24;
    r58 = r31 * r6;
    r58 = r58 * r18;
    r58 = fma(r11, r58, r29 * r32);
    r32 = r2 * r20;
    r32 = fma(r31, r8, r29 * r32);
    r59 = fma(r0, r32, r1 * r58);
    r63 = fma(r47, r59, r48);
    r64 = r15 * r5;
    r32 = fma(r14, r32, r34 * r58);
    r59 = fma(r59, r49, r32 * r64);
    r64 = r59 * r50;
    r64 = r4 <= r38 ? r59 : r64;
    r58 = r59 * r52;
    r64 = r35 < r39 ? r58 : r64;
    r58 = r59 * r43;
    r64 = r35 < r41 ? r58 : r64;
    r64 = r35 < r37 ? r59 : r64;
    r58 = r37 * r64;
    r59 = fma(r59, r44, r45 * r58);
    r59 = r37 * r59;
    r59 = r59 * r46;
    r59 = r4 <= r28 ? r30 : r59;
    r63 = fma(r21, r59, r63);
    r59 = fma(r5, r59, r40);
    r59 = fma(r47, r32, r59);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             2 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r63,
                                             r59);
    r32 = r15 * r5;
    r58 = r7 * r19;
    r65 = r26 * r6;
    r65 = r65 * r18;
    r65 = fma(r11, r65, r29 * r58);
    r58 = r2 * r23;
    r58 = fma(r29, r58, r26 * r8);
    r8 = fma(r14, r58, r34 * r65);
    r58 = fma(r0, r58, r1 * r65);
    r49 = fma(r58, r49, r8 * r32);
    r50 = r49 * r50;
    r50 = r4 <= r38 ? r49 : r50;
    r52 = r49 * r52;
    r50 = r35 < r39 ? r52 : r50;
    r43 = r49 * r43;
    r50 = r35 < r41 ? r43 : r50;
    r50 = r35 < r37 ? r49 : r50;
    r35 = r37 * r50;
    r44 = fma(r49, r44, r45 * r35);
    r44 = r37 * r44;
    r44 = r44 * r46;
    r44 = r4 <= r28 ? r30 : r44;
    r48 = fma(r21, r44, r48);
    r48 = fma(r47, r58, r48);
    r44 = fma(r5, r44, r40);
    r44 = fma(r47, r8, r44);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             4 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r48,
                                             r44);
    r8 = r21 * r62;
    r8 = fma(r56, r61, r60 * r8);
    r47 = r21 * r63;
    r47 = fma(r59, r61, r60 * r47);
    WriteSum2<double, double>((double*)inout_shared, r8, r47);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = r21 * r48;
    r61 = fma(r44, r61, r60 * r47);
    WriteSum1<double, double>((double*)inout_shared, r61);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r61 = fma(r62, r62, r56 * r56);
    r47 = fma(r59, r59, r63 * r63);
    WriteSum2<double, double>((double*)inout_shared, r61, r47);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = fma(r48, r48, r44 * r44);
    WriteSum1<double, double>((double*)inout_shared, r47);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = fma(r62, r63, r56 * r59);
    r56 = fma(r56, r44, r62 * r48);
    WriteSum2<double, double>((double*)inout_shared, r47, r56);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r44 = fma(r63, r48, r59 * r44);
    WriteSum1<double, double>((double*)inout_shared, r44);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
}

void PinholeFixedPoseResJac(double* calib,
                            unsigned int calib_num_alloc,
                            SharedIndex* calib_indices,
                            double* point,
                            unsigned int point_num_alloc,
                            SharedIndex* point_indices,
                            double* pixel,
                            unsigned int pixel_num_alloc,
                            double* weight_loss,
                            unsigned int weight_loss_num_alloc,
                            double* pose,
                            unsigned int pose_num_alloc,
                            double* out_res,
                            unsigned int out_res_num_alloc,
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
  PinholeFixedPoseResJacKernel<<<n_blocks, 1024>>>(
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
      pose,
      pose_num_alloc,
      out_res,
      out_res_num_alloc,
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