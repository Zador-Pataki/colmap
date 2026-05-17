#include "kernel_pinhole_split_fixed_focal_fixed_principal_point_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedFocalFixedPrincipalPointFixedPointResJacKernel(
        double* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
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
        double* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        double* const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        double* const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67;

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
  LoadShared<2, double, double>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r3, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        point, 0 * point_num_alloc, global_thread_idx, r8, r9);
  };
  LoadShared<2, double, double>(
      pose, 2 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r10, r11);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r12 = r10 * r11;
    r13 = 2.00000000000000000e+00;
    r12 = r12 * r13;
  };
  LoadShared<2, double, double>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r14, r15);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r16 = r14 * r15;
    r16 = r16 * r13;
    r17 = r12 + r16;
    r7 = fma(r8, r17, r7);
    ReadIdx1<1024, double, double, double>(
        point, 2 * point_num_alloc, global_thread_idx, r18);
    r19 = r15 * r10;
    r19 = r19 * r13;
    r20 = -2.00000000000000000e+00;
    r21 = r14 * r20;
    r22 = r11 * r21;
    r23 = r19 + r22;
    r24 = r10 * r10;
    r25 = r20 * r24;
    r26 = 1.00000000000000000e+00;
    r27 = fma(r14, r21, r26);
    r28 = r25 + r27;
    r7 = fma(r18, r23, r7);
    r7 = fma(r9, r28, r7);
    ReadIdx2<1024, double, double, double2>(
        focal, 0 * focal_num_alloc, global_thread_idx, r28, r29);
    r30 = 1.00000000000000008e-15;
  };
  LoadShared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r31);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r32 = r14 * r11;
    r32 = r32 * r13;
    r19 = r19 + r32;
    r31 = fma(r9, r19, r31);
    r33 = r14 * r10;
    r33 = r33 * r13;
    r34 = r15 * r11;
    r34 = r34 * r20;
    r35 = r33 + r34;
    r36 = r15 * r15;
    r37 = r20 * r36;
    r27 = r37 + r27;
    r31 = fma(r8, r35, r31);
    r31 = fma(r18, r27, r31);
    r27 = copysign(1.0, r31);
    r27 = fma(r30, r27, r31);
    r31 = 1.0 / r27;
    r38 = r29 * r31;
    r5 = fma(r7, r38, r5);
    r4 = fma(r4, r6, r2);
    r2 = r10 * r11;
    r2 = r2 * r20;
    r16 = r16 + r2;
    r3 = fma(r9, r16, r3);
    r39 = r15 * r11;
    r39 = r39 * r13;
    r33 = r33 + r39;
    r25 = r26 + r25;
    r25 = r25 + r37;
    r3 = fma(r18, r33, r3);
    r3 = fma(r8, r25, r3);
    r3 = r28 * r3;
    r4 = fma(r31, r3, r4);
    r25 = fma(r0, r4, r1 * r5);
    ReadIdx1<1024, double, double, double>(
        weight_loss, 6 * weight_loss_num_alloc, global_thread_idx, r37);
    r40 = 0.00000000000000000e+00;
    r37 = fmax(r37, r40);
    r41 = sqrt(r37);
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 2 * weight_loss_num_alloc, global_thread_idx, r42, r43);
    r4 = fma(r42, r4, r43 * r5);
    r5 = fma(r4, r4, r25 * r25);
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 4 * weight_loss_num_alloc, global_thread_idx, r44, r45);
    r46 = 5.00000000000000000e-01;
    r45 = fmax(r45, r30);
    r47 = r45 * r45;
    r48 = r6 * r45;
    r49 = r13 * r45;
    r50 = fmax(r30, r5);
    r51 = sqrt(r50);
    r49 = fma(r51, r49, r45 * r48);
    r49 = r5 <= r47 ? r5 : r49;
    r48 = 2.50000000000000000e+00;
    r51 = r45 * r45;
    r52 = 1.0 / r47;
    r52 = fma(r5, r52, r26);
    r53 = log(r52);
    r51 = r51 * r53;
    r49 = r44 < r48 ? r51 : r49;
    r51 = 1.50000000000000000e+00;
    r53 = r13 * r45;
    r54 = sqrt(r52);
    r54 = r6 + r54;
    r53 = r53 * r45;
    r53 = r53 * r54;
    r49 = r44 < r51 ? r53 : r49;
    r49 = r44 < r46 ? r5 : r49;
    r53 = fmax(r40, r49);
    r54 = 1.0 / r50;
    r54 = r37 * r54;
    r55 = r53 * r54;
    r56 = sqrt(r55);
    r56 = r5 <= r30 ? r41 : r56;
    r41 = r25 * r56;
    r57 = r4 * r56;
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r41, r57);
    r53 = r37 * r53;
    r37 = -5.00000000000000000e-01;
    r57 = -1.00000000000000008e-15;
    r57 = r57 + r5;
    r57 = copysign(1.0, r57);
    r57 = r26 + r57;
    r41 = r50 * r50;
    r41 = 1.0 / r41;
    r53 = r53 * r37;
    r53 = r53 * r57;
    r53 = r53 * r41;
    r41 = r13 * r4;
    r37 = r15 * r10;
    r37 = r37 * r20;
    r22 = r37 + r22;
    r20 = r6 * r36;
    r58 = r24 + r20;
    r11 = r11 * r11;
    r59 = r14 * r14;
    r59 = r59 * r6;
    r60 = r11 + r59;
    r61 = r58 + r60;
    r61 = fma(r9, r61, r18 * r22);
    r22 = r6 * r61;
    r27 = r27 * r27;
    r27 = 1.0 / r27;
    r22 = r22 * r27;
    r15 = r15 * r21;
    r12 = r12 + r15;
    r33 = fma(r9, r33, r18 * r12);
    r12 = r28 * r33;
    r12 = fma(r31, r12, r3 * r22);
    r14 = r14 * r14;
    r22 = r6 * r11;
    r62 = r14 + r22;
    r58 = r58 + r62;
    r58 = fma(r18, r58, r9 * r23);
    r7 = r29 * r7;
    r7 = r7 * r6;
    r7 = r7 * r27;
    r58 = fma(r61, r7, r58 * r38);
    r29 = fma(r43, r58, r42 * r12);
    r58 = fma(r1, r58, r0 * r12);
    r12 = r13 * r25;
    r41 = fma(r58, r12, r29 * r41);
    r23 = r46 * r45;
    r50 = rsqrt(r50);
    r23 = r23 * r57;
    r23 = r23 * r50;
    r50 = r41 * r23;
    r50 = r5 <= r47 ? r41 : r50;
    r57 = 1.0 / r52;
    r63 = r41 * r57;
    r50 = r44 < r48 ? r63 : r50;
    r52 = rsqrt(r52);
    r63 = r41 * r52;
    r50 = r44 < r51 ? r63 : r50;
    r50 = r44 < r46 ? r41 : r50;
    r63 = r46 * r50;
    r49 = copysign(1.0, r49);
    r49 = r26 + r49;
    r54 = r49 * r54;
    r63 = fma(r54, r63, r41 * r53);
    r63 = r46 * r63;
    r55 = rsqrt(r55);
    r63 = r63 * r55;
    r63 = r5 <= r30 ? r40 : r63;
    r41 = r6 * r25;
    r49 = 2.50000000000000000e-01;
    r26 = r5 <= r47 ? r40 : r40;
    r26 = r44 < r48 ? r40 : r26;
    r26 = r44 < r51 ? r40 : r26;
    r26 = r44 < r46 ? r40 : r26;
    r26 = r49 * r26;
    r26 = r26 * r55;
    r26 = r26 * r54;
    r26 = r5 <= r30 ? r40 : r26;
    r41 = r41 * r26;
    r49 = fma(r25, r63, r41);
    r49 = fma(r56, r58, r49);
    r58 = r25 * r49;
    r64 = r6 * r56;
    r65 = r6 * r4;
    r65 = r65 * r26;
    r63 = fma(r4, r63, r65);
    r63 = fma(r56, r29, r63);
    r29 = r4 * r64;
    r58 = fma(r63, r29, r64 * r58);
    r26 = r13 * r4;
    r66 = r6 * r24;
    r67 = r36 + r66;
    r62 = r62 + r67;
    r62 = fma(r8, r62, r18 * r35);
    r37 = r32 + r37;
    r37 = fma(r8, r37, r18 * r17);
    r37 = fma(r37, r38, r62 * r7);
    r21 = r10 * r21;
    r34 = r34 + r21;
    r11 = r14 + r11;
    r11 = r11 + r20;
    r11 = r11 + r66;
    r11 = fma(r18, r11, r8 * r34);
    r18 = r28 * r11;
    r34 = r6 * r62;
    r34 = r34 * r27;
    r34 = fma(r3, r34, r31 * r18);
    r18 = fma(r42, r34, r43 * r37);
    r34 = fma(r0, r34, r1 * r37);
    r26 = fma(r34, r12, r18 * r26);
    r37 = r26 * r23;
    r37 = r5 <= r47 ? r26 : r37;
    r66 = r26 * r57;
    r37 = r44 < r48 ? r66 : r37;
    r66 = r26 * r52;
    r37 = r44 < r51 ? r66 : r37;
    r37 = r44 < r46 ? r26 : r37;
    r66 = r46 * r37;
    r66 = fma(r54, r66, r26 * r53);
    r66 = r46 * r66;
    r66 = r66 * r55;
    r66 = r5 <= r30 ? r40 : r66;
    r26 = fma(r4, r66, r65);
    r26 = fma(r56, r18, r26);
    r66 = fma(r25, r66, r41);
    r66 = fma(r56, r34, r66);
    r34 = r25 * r66;
    r34 = fma(r64, r34, r26 * r29);
    WriteSum2<double, double>((double*)inout_shared, r58, r34);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = r13 * r4;
    r21 = r39 + r21;
    r21 = fma(r9, r21, r8 * r19);
    r15 = r2 + r15;
    r67 = r60 + r67;
    r67 = fma(r8, r67, r9 * r15);
    r67 = fma(r67, r38, r21 * r7);
    r15 = r6 * r21;
    r15 = r15 * r27;
    r36 = r24 + r36;
    r36 = r36 + r59;
    r36 = r36 + r22;
    r36 = fma(r9, r36, r8 * r16);
    r9 = r28 * r36;
    r9 = fma(r31, r9, r3 * r15);
    r15 = fma(r42, r9, r43 * r67);
    r9 = fma(r0, r9, r1 * r67);
    r34 = fma(r9, r12, r15 * r34);
    r67 = r34 * r23;
    r67 = r5 <= r47 ? r34 : r67;
    r16 = r34 * r57;
    r67 = r44 < r48 ? r16 : r67;
    r16 = r34 * r52;
    r67 = r44 < r51 ? r16 : r67;
    r67 = r44 < r46 ? r34 : r67;
    r16 = r46 * r67;
    r34 = fma(r34, r53, r54 * r16);
    r34 = r46 * r34;
    r34 = r34 * r55;
    r34 = r5 <= r30 ? r40 : r34;
    r16 = fma(r25, r34, r41);
    r16 = fma(r56, r9, r16);
    r9 = r25 * r16;
    r34 = fma(r4, r34, r65);
    r34 = fma(r56, r15, r34);
    r9 = fma(r34, r29, r64 * r9);
    r15 = r0 * r28;
    r15 = r15 * r31;
    r8 = r42 * r28;
    r8 = r8 * r13;
    r8 = r8 * r4;
    r8 = fma(r31, r8, r12 * r15);
    r15 = r8 * r23;
    r15 = r5 <= r47 ? r8 : r15;
    r22 = r8 * r57;
    r15 = r44 < r48 ? r22 : r15;
    r22 = r8 * r52;
    r15 = r44 < r51 ? r22 : r15;
    r15 = r44 < r46 ? r8 : r15;
    r22 = r46 * r15;
    r8 = fma(r8, r53, r54 * r22);
    r8 = r46 * r8;
    r8 = r8 * r55;
    r8 = r5 <= r30 ? r40 : r8;
    r22 = fma(r4, r8, r65);
    r59 = r42 * r28;
    r59 = r59 * r56;
    r22 = fma(r31, r59, r22);
    r8 = fma(r25, r8, r41);
    r59 = r0 * r28;
    r59 = r59 * r56;
    r8 = fma(r31, r59, r8);
    r59 = r25 * r8;
    r59 = fma(r64, r59, r22 * r29);
    WriteSum2<double, double>((double*)inout_shared, r9, r59);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r59 = r1 * r38;
    r9 = r43 * r13;
    r9 = r9 * r4;
    r9 = fma(r38, r9, r12 * r59);
    r31 = r9 * r23;
    r31 = r5 <= r47 ? r9 : r31;
    r24 = r9 * r57;
    r31 = r44 < r48 ? r24 : r31;
    r24 = r9 * r52;
    r31 = r44 < r51 ? r24 : r31;
    r31 = r44 < r46 ? r9 : r31;
    r24 = r46 * r31;
    r24 = fma(r54, r24, r9 * r53);
    r24 = r46 * r24;
    r24 = r24 * r55;
    r24 = r5 <= r30 ? r40 : r24;
    r9 = fma(r25, r24, r41);
    r9 = fma(r56, r59, r9);
    r59 = r25 * r9;
    r24 = fma(r4, r24, r65);
    r60 = r43 * r56;
    r24 = fma(r38, r60, r24);
    r59 = fma(r24, r29, r64 * r59);
    r60 = r13 * r4;
    r38 = r42 * r6;
    r38 = r38 * r27;
    r38 = fma(r43, r7, r3 * r38);
    r2 = r0 * r6;
    r2 = r2 * r27;
    r7 = fma(r1, r7, r3 * r2);
    r12 = fma(r7, r12, r38 * r60);
    r23 = r12 * r23;
    r23 = r5 <= r47 ? r12 : r23;
    r57 = r12 * r57;
    r23 = r44 < r48 ? r57 : r23;
    r52 = r12 * r52;
    r23 = r44 < r51 ? r52 : r23;
    r23 = r44 < r46 ? r12 : r23;
    r44 = r46 * r23;
    r44 = fma(r54, r44, r12 * r53);
    r44 = r46 * r44;
    r44 = r44 * r55;
    r44 = r5 <= r30 ? r40 : r44;
    r41 = fma(r25, r44, r41);
    r41 = fma(r56, r7, r41);
    r7 = r25 * r41;
    r44 = fma(r4, r44, r65);
    r44 = fma(r56, r38, r44);
    r29 = fma(r44, r29, r64 * r7);
    WriteSum2<double, double>((double*)inout_shared, r59, r29);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r29 = fma(r63, r63, r49 * r49);
    r59 = fma(r26, r26, r66 * r66);
    WriteSum2<double, double>((double*)inout_shared, r29, r59);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r59 = fma(r16, r16, r34 * r34);
    r29 = fma(r22, r22, r8 * r8);
    WriteSum2<double, double>((double*)inout_shared, r59, r29);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r29 = fma(r24, r24, r9 * r9);
    r59 = fma(r44, r44, r41 * r41);
    WriteSum2<double, double>((double*)inout_shared, r29, r59);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r59 = fma(r63, r26, r49 * r66);
    r29 = fma(r63, r34, r49 * r16);
    WriteSum2<double, double>((double*)inout_shared, r59, r29);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r29 = fma(r49, r8, r63 * r22);
    r59 = fma(r49, r9, r63 * r24);
    WriteSum2<double, double>((double*)inout_shared, r29, r59);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r63 = fma(r63, r44, r49 * r41);
    r59 = fma(r26, r34, r66 * r16);
    WriteSum2<double, double>((double*)inout_shared, r63, r59);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r59 = fma(r66, r8, r26 * r22);
    r63 = fma(r26, r24, r66 * r9);
    WriteSum2<double, double>((double*)inout_shared, r59, r63);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = fma(r66, r41, r26 * r44);
    r63 = fma(r34, r22, r16 * r8);
    WriteSum2<double, double>((double*)inout_shared, r26, r63);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r63 = fma(r34, r24, r16 * r9);
    r34 = fma(r34, r44, r16 * r41);
    WriteSum2<double, double>((double*)inout_shared, r63, r34);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = fma(r8, r9, r22 * r24);
    r22 = fma(r8, r41, r22 * r44);
    WriteSum2<double, double>((double*)inout_shared, r34, r22);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r44 = fma(r24, r44, r9 * r41);
    WriteSum1<double, double>((double*)inout_shared, r44);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
}

void PinholeSplitFixedFocalFixedPrincipalPointFixedPointResJac(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
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
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    double* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitFixedFocalFixedPrincipalPointFixedPointResJacKernel<<<n_blocks,
                                                                    1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
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
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_pose_precond_diag,
      out_pose_precond_diag_num_alloc,
      out_pose_precond_tril,
      out_pose_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar