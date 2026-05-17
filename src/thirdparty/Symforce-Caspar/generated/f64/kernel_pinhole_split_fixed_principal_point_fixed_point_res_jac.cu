#include "kernel_pinhole_split_fixed_principal_point_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedPrincipalPointFixedPointResJacKernel(
        double* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        double* focal,
        unsigned int focal_num_alloc,
        SharedIndex* focal_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* weight_loss,
        unsigned int weight_loss_num_alloc,
        double* principal_point,
        unsigned int principal_point_num_alloc,
        double* point,
        unsigned int point_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* out_pose_jac,
        unsigned int out_pose_jac_num_alloc,
        double* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        double* const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        double* const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
        double* out_focal_jac,
        unsigned int out_focal_jac_num_alloc,
        double* const out_focal_njtr,
        unsigned int out_focal_njtr_num_alloc,
        double* const out_focal_precond_diag,
        unsigned int out_focal_precond_diag_num_alloc,
        double* const out_focal_precond_tril,
        unsigned int out_focal_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex focal_indices_loc[1024];
  focal_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66;

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
  };
  LoadShared<2, double, double>(
      focal, 0 * focal_num_alloc, focal_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, focal_indices_loc[threadIdx.x].target, r28, r29);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
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
    r25 = r28 * r3;
    r4 = fma(r31, r25, r4);
    r37 = fma(r0, r4, r1 * r5);
    ReadIdx1<1024, double, double, double>(
        weight_loss, 6 * weight_loss_num_alloc, global_thread_idx, r40);
    r41 = 0.00000000000000000e+00;
    r40 = fmax(r40, r41);
    r42 = sqrt(r40);
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 2 * weight_loss_num_alloc, global_thread_idx, r43, r44);
    r4 = fma(r43, r4, r44 * r5);
    r5 = fma(r4, r4, r37 * r37);
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 4 * weight_loss_num_alloc, global_thread_idx, r45, r46);
    r47 = 5.00000000000000000e-01;
    r46 = fmax(r46, r30);
    r48 = r46 * r46;
    r49 = r6 * r46;
    r50 = r13 * r46;
    r51 = fmax(r30, r5);
    r52 = sqrt(r51);
    r50 = fma(r52, r50, r46 * r49);
    r50 = r5 <= r48 ? r5 : r50;
    r49 = 2.50000000000000000e+00;
    r52 = r46 * r46;
    r53 = 1.0 / r48;
    r53 = fma(r5, r53, r26);
    r54 = log(r53);
    r52 = r52 * r54;
    r50 = r45 < r49 ? r52 : r50;
    r52 = 1.50000000000000000e+00;
    r54 = r13 * r46;
    r55 = sqrt(r53);
    r55 = r6 + r55;
    r54 = r54 * r46;
    r54 = r54 * r55;
    r50 = r45 < r52 ? r54 : r50;
    r50 = r45 < r47 ? r5 : r50;
    r54 = fmax(r41, r50);
    r55 = 1.0 / r51;
    r55 = r40 * r55;
    r56 = r54 * r55;
    r57 = sqrt(r56);
    r57 = r5 <= r30 ? r42 : r57;
    r42 = r37 * r57;
    r58 = r4 * r57;
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r42, r58);
    r54 = r40 * r54;
    r40 = -5.00000000000000000e-01;
    r58 = -1.00000000000000008e-15;
    r58 = r58 + r5;
    r58 = copysign(1.0, r58);
    r58 = r26 + r58;
    r42 = r51 * r51;
    r42 = 1.0 / r42;
    r54 = r54 * r40;
    r54 = r54 * r58;
    r54 = r54 * r42;
    r42 = r13 * r4;
    r40 = r15 * r10;
    r40 = r40 * r20;
    r22 = r40 + r22;
    r20 = r6 * r36;
    r59 = r24 + r20;
    r11 = r11 * r11;
    r60 = r14 * r14;
    r60 = r60 * r6;
    r61 = r11 + r60;
    r62 = r59 + r61;
    r62 = fma(r9, r62, r18 * r22);
    r22 = r6 * r62;
    r27 = r27 * r27;
    r27 = 1.0 / r27;
    r22 = r22 * r27;
    r15 = r15 * r21;
    r12 = r12 + r15;
    r33 = fma(r9, r33, r18 * r12);
    r12 = r28 * r33;
    r12 = fma(r31, r12, r25 * r22);
    r14 = r14 * r14;
    r22 = r6 * r11;
    r63 = r14 + r22;
    r59 = r59 + r63;
    r59 = fma(r18, r59, r9 * r23);
    r29 = r29 * r7;
    r29 = r29 * r6;
    r29 = r29 * r27;
    r59 = fma(r62, r29, r59 * r38);
    r23 = fma(r44, r59, r43 * r12);
    r59 = fma(r1, r59, r0 * r12);
    r12 = r13 * r37;
    r42 = fma(r59, r12, r23 * r42);
    r64 = r47 * r46;
    r51 = rsqrt(r51);
    r64 = r64 * r58;
    r64 = r64 * r51;
    r51 = r42 * r64;
    r51 = r5 <= r48 ? r42 : r51;
    r58 = 1.0 / r53;
    r65 = r42 * r58;
    r51 = r45 < r49 ? r65 : r51;
    r53 = rsqrt(r53);
    r65 = r42 * r53;
    r51 = r45 < r52 ? r65 : r51;
    r51 = r45 < r47 ? r42 : r51;
    r65 = r47 * r51;
    r50 = copysign(1.0, r50);
    r50 = r26 + r50;
    r55 = r50 * r55;
    r65 = fma(r55, r65, r42 * r54);
    r65 = r47 * r65;
    r56 = rsqrt(r56);
    r65 = r65 * r56;
    r65 = r5 <= r30 ? r41 : r65;
    r42 = r6 * r37;
    r50 = 2.50000000000000000e-01;
    r26 = r5 <= r48 ? r41 : r41;
    r26 = r45 < r49 ? r41 : r26;
    r26 = r45 < r52 ? r41 : r26;
    r26 = r45 < r47 ? r41 : r26;
    r26 = r50 * r26;
    r26 = r26 * r56;
    r26 = r26 * r55;
    r26 = r5 <= r30 ? r41 : r26;
    r42 = r42 * r26;
    r50 = fma(r37, r65, r42);
    r50 = fma(r57, r59, r50);
    r59 = r6 * r4;
    r59 = r59 * r26;
    r65 = fma(r4, r65, r59);
    r65 = fma(r57, r23, r65);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r50, r65);
    r23 = r13 * r4;
    r26 = r6 * r24;
    r66 = r36 + r26;
    r63 = r63 + r66;
    r63 = fma(r8, r63, r18 * r35);
    r40 = r32 + r40;
    r40 = fma(r8, r40, r18 * r17);
    r40 = fma(r40, r38, r63 * r29);
    r21 = r10 * r21;
    r34 = r34 + r21;
    r11 = r14 + r11;
    r11 = r11 + r20;
    r11 = r11 + r26;
    r11 = fma(r18, r11, r8 * r34);
    r18 = r28 * r11;
    r34 = r6 * r63;
    r34 = r34 * r27;
    r34 = fma(r25, r34, r31 * r18);
    r18 = fma(r43, r34, r44 * r40);
    r34 = fma(r0, r34, r1 * r40);
    r23 = fma(r34, r12, r18 * r23);
    r40 = r23 * r64;
    r40 = r5 <= r48 ? r23 : r40;
    r26 = r23 * r58;
    r40 = r45 < r49 ? r26 : r40;
    r26 = r23 * r53;
    r40 = r45 < r52 ? r26 : r40;
    r40 = r45 < r47 ? r23 : r40;
    r26 = r47 * r40;
    r26 = fma(r55, r26, r23 * r54);
    r26 = r47 * r26;
    r26 = r26 * r56;
    r26 = r5 <= r30 ? r41 : r26;
    r23 = fma(r37, r26, r42);
    r23 = fma(r57, r34, r23);
    r26 = fma(r4, r26, r59);
    r26 = fma(r57, r18, r26);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r23, r26);
    r18 = r13 * r4;
    r21 = r39 + r21;
    r21 = fma(r9, r21, r8 * r19);
    r15 = r2 + r15;
    r66 = r61 + r66;
    r66 = fma(r8, r66, r9 * r15);
    r66 = fma(r66, r38, r21 * r29);
    r15 = r6 * r21;
    r15 = r15 * r27;
    r36 = r24 + r36;
    r36 = r36 + r60;
    r36 = r36 + r22;
    r36 = fma(r9, r36, r8 * r16);
    r9 = r28 * r36;
    r9 = fma(r31, r9, r25 * r15);
    r15 = fma(r43, r9, r44 * r66);
    r9 = fma(r0, r9, r1 * r66);
    r18 = fma(r9, r12, r15 * r18);
    r66 = r18 * r64;
    r66 = r5 <= r48 ? r18 : r66;
    r16 = r18 * r58;
    r66 = r45 < r49 ? r16 : r66;
    r16 = r18 * r53;
    r66 = r45 < r52 ? r16 : r66;
    r66 = r45 < r47 ? r18 : r66;
    r16 = r47 * r66;
    r18 = fma(r18, r54, r55 * r16);
    r18 = r47 * r18;
    r18 = r18 * r56;
    r18 = r5 <= r30 ? r41 : r18;
    r16 = fma(r37, r18, r42);
    r16 = fma(r57, r9, r16);
    r18 = fma(r4, r18, r59);
    r18 = fma(r57, r15, r18);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r16, r18);
    r15 = r31 * r12;
    r9 = r0 * r15;
    r8 = r28 * r43;
    r8 = r8 * r13;
    r8 = r8 * r4;
    r8 = fma(r31, r8, r28 * r9);
    r22 = r8 * r64;
    r22 = r5 <= r48 ? r8 : r22;
    r60 = r8 * r58;
    r22 = r45 < r49 ? r60 : r22;
    r60 = r8 * r53;
    r22 = r45 < r52 ? r60 : r22;
    r22 = r45 < r47 ? r8 : r22;
    r60 = r47 * r22;
    r8 = fma(r8, r54, r55 * r60);
    r8 = r47 * r8;
    r8 = r8 * r56;
    r8 = r5 <= r30 ? r41 : r8;
    r60 = fma(r37, r8, r42);
    r24 = r28 * r0;
    r24 = r24 * r57;
    r60 = fma(r31, r24, r60);
    r8 = fma(r4, r8, r59);
    r24 = r28 * r43;
    r24 = r24 * r57;
    r8 = fma(r31, r24, r8);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r60, r8);
    r24 = r1 * r38;
    r61 = r44 * r13;
    r61 = r61 * r4;
    r61 = fma(r38, r61, r12 * r24);
    r2 = r61 * r64;
    r2 = r5 <= r48 ? r61 : r2;
    r19 = r61 * r58;
    r2 = r45 < r49 ? r19 : r2;
    r19 = r61 * r53;
    r2 = r45 < r52 ? r19 : r2;
    r2 = r45 < r47 ? r61 : r2;
    r19 = r47 * r2;
    r19 = fma(r55, r19, r61 * r54);
    r19 = r47 * r19;
    r19 = r19 * r56;
    r19 = r5 <= r30 ? r41 : r19;
    r61 = fma(r37, r19, r42);
    r61 = fma(r57, r24, r61);
    r19 = fma(r4, r19, r59);
    r24 = r44 * r57;
    r19 = fma(r38, r24, r19);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r61, r19);
    r24 = r13 * r4;
    r38 = r43 * r6;
    r38 = r38 * r27;
    r38 = fma(r44, r29, r25 * r38);
    r39 = r0 * r6;
    r39 = r39 * r27;
    r29 = fma(r1, r29, r25 * r39);
    r12 = fma(r29, r12, r38 * r24);
    r24 = r12 * r64;
    r24 = r5 <= r48 ? r12 : r24;
    r39 = r12 * r58;
    r24 = r45 < r49 ? r39 : r24;
    r39 = r12 * r53;
    r24 = r45 < r52 ? r39 : r24;
    r24 = r45 < r47 ? r12 : r24;
    r39 = r47 * r24;
    r39 = fma(r55, r39, r12 * r54);
    r39 = r47 * r39;
    r39 = r39 * r56;
    r39 = r5 <= r30 ? r41 : r39;
    r12 = fma(r37, r39, r42);
    r12 = fma(r57, r29, r12);
    r39 = fma(r4, r39, r59);
    r39 = fma(r57, r38, r39);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r12, r39);
    r38 = r37 * r50;
    r29 = r6 * r57;
    r25 = r4 * r29;
    r38 = fma(r65, r25, r29 * r38);
    r27 = r37 * r23;
    r27 = fma(r29, r27, r26 * r25);
    WriteSum2<double, double>((double*)inout_shared, r38, r27);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = r37 * r16;
    r27 = fma(r18, r25, r29 * r27);
    r38 = r37 * r60;
    r38 = fma(r29, r38, r8 * r25);
    WriteSum2<double, double>((double*)inout_shared, r27, r38);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r38 = r37 * r61;
    r38 = fma(r19, r25, r29 * r38);
    r27 = r37 * r12;
    r27 = fma(r39, r25, r29 * r27);
    WriteSum2<double, double>((double*)inout_shared, r38, r27);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fma(r65, r65, r50 * r50);
    r38 = fma(r26, r26, r23 * r23);
    WriteSum2<double, double>((double*)inout_shared, r27, r38);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r38 = fma(r16, r16, r18 * r18);
    r27 = fma(r8, r8, r60 * r60);
    WriteSum2<double, double>((double*)inout_shared, r38, r27);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fma(r19, r19, r61 * r61);
    r38 = fma(r39, r39, r12 * r12);
    WriteSum2<double, double>((double*)inout_shared, r27, r38);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r38 = fma(r65, r26, r50 * r23);
    r27 = fma(r65, r18, r50 * r16);
    WriteSum2<double, double>((double*)inout_shared, r38, r27);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fma(r50, r60, r65 * r8);
    r38 = fma(r50, r61, r65 * r19);
    WriteSum2<double, double>((double*)inout_shared, r27, r38);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r65 = fma(r65, r39, r50 * r12);
    r38 = fma(r26, r18, r23 * r16);
    WriteSum2<double, double>((double*)inout_shared, r65, r38);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r38 = fma(r23, r60, r26 * r8);
    r65 = fma(r26, r19, r23 * r61);
    WriteSum2<double, double>((double*)inout_shared, r38, r65);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = fma(r23, r12, r26 * r39);
    r65 = fma(r18, r8, r16 * r60);
    WriteSum2<double, double>((double*)inout_shared, r26, r65);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r65 = fma(r18, r19, r16 * r61);
    r18 = fma(r18, r39, r16 * r12);
    WriteSum2<double, double>((double*)inout_shared, r65, r18);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r18 = fma(r60, r61, r8 * r19);
    r8 = fma(r60, r12, r8 * r39);
    WriteSum2<double, double>((double*)inout_shared, r18, r8);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = fma(r19, r39, r61 * r12);
    WriteSum1<double, double>((double*)inout_shared, r39);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = r43 * r13;
    r39 = r39 * r3;
    r39 = r39 * r4;
    r39 = fma(r31, r39, r3 * r9);
    r9 = r39 * r64;
    r9 = r5 <= r48 ? r39 : r9;
    r19 = r39 * r58;
    r9 = r45 < r49 ? r19 : r9;
    r19 = r39 * r53;
    r9 = r45 < r52 ? r19 : r9;
    r9 = r45 < r47 ? r39 : r9;
    r19 = r47 * r9;
    r19 = fma(r55, r19, r39 * r54);
    r19 = r47 * r19;
    r19 = r19 * r56;
    r19 = r5 <= r30 ? r41 : r19;
    r39 = fma(r37, r19, r42);
    r8 = r0 * r3;
    r8 = r8 * r57;
    r39 = fma(r31, r8, r39);
    r19 = fma(r4, r19, r59);
    r8 = r43 * r3;
    r8 = r8 * r57;
    r19 = fma(r31, r8, r19);
    WriteIdx2<1024, double, double, double2>(out_focal_jac,
                                             0 * out_focal_jac_num_alloc,
                                             global_thread_idx,
                                             r39,
                                             r19);
    r8 = r1 * r7;
    r18 = r44 * r13;
    r18 = r18 * r7;
    r18 = r18 * r4;
    r18 = fma(r31, r18, r15 * r8);
    r64 = r18 * r64;
    r64 = r5 <= r48 ? r18 : r64;
    r58 = r18 * r58;
    r64 = r45 < r49 ? r58 : r64;
    r53 = r18 * r53;
    r64 = r45 < r52 ? r53 : r64;
    r64 = r45 < r47 ? r18 : r64;
    r45 = r47 * r64;
    r45 = fma(r55, r45, r18 * r54);
    r45 = r47 * r45;
    r45 = r45 * r56;
    r45 = r5 <= r30 ? r41 : r45;
    r42 = fma(r37, r45, r42);
    r41 = r1 * r7;
    r41 = r41 * r57;
    r42 = fma(r31, r41, r42);
    r45 = fma(r4, r45, r59);
    r59 = r44 * r7;
    r59 = r59 * r57;
    r45 = fma(r31, r59, r45);
    WriteIdx2<1024, double, double, double2>(out_focal_jac,
                                             2 * out_focal_jac_num_alloc,
                                             global_thread_idx,
                                             r42,
                                             r45);
    r59 = r37 * r39;
    r59 = fma(r29, r59, r19 * r25);
    r31 = r37 * r42;
    r31 = fma(r29, r31, r45 * r25);
    WriteSum2<double, double>((double*)inout_shared, r59, r31);
  };
  FlushSumShared<2, double>(out_focal_njtr,
                            0 * out_focal_njtr_num_alloc,
                            focal_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = fma(r39, r39, r19 * r19);
    r59 = fma(r45, r45, r42 * r42);
    WriteSum2<double, double>((double*)inout_shared, r31, r59);
  };
  FlushSumShared<2, double>(out_focal_precond_diag,
                            0 * out_focal_precond_diag_num_alloc,
                            focal_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r45 = fma(r19, r45, r39 * r42);
    WriteSum1<double, double>((double*)inout_shared, r45);
  };
  FlushSumShared<1, double>(out_focal_precond_tril,
                            0 * out_focal_precond_tril_num_alloc,
                            focal_indices_loc,
                            (double*)inout_shared);
}

void PinholeSplitFixedPrincipalPointFixedPointResJac(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* focal,
    unsigned int focal_num_alloc,
    SharedIndex* focal_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* weight_loss,
    unsigned int weight_loss_num_alloc,
    double* principal_point,
    unsigned int principal_point_num_alloc,
    double* point,
    unsigned int point_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* out_pose_jac,
    unsigned int out_pose_jac_num_alloc,
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    double* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    double* out_focal_jac,
    unsigned int out_focal_jac_num_alloc,
    double* const out_focal_njtr,
    unsigned int out_focal_njtr_num_alloc,
    double* const out_focal_precond_diag,
    unsigned int out_focal_precond_diag_num_alloc,
    double* const out_focal_precond_tril,
    unsigned int out_focal_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitFixedPrincipalPointFixedPointResJacKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      focal,
      focal_num_alloc,
      focal_indices,
      pixel,
      pixel_num_alloc,
      weight_loss,
      weight_loss_num_alloc,
      principal_point,
      principal_point_num_alloc,
      point,
      point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_pose_jac,
      out_pose_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_pose_precond_diag,
      out_pose_precond_diag_num_alloc,
      out_pose_precond_tril,
      out_pose_precond_tril_num_alloc,
      out_focal_jac,
      out_focal_jac_num_alloc,
      out_focal_njtr,
      out_focal_njtr_num_alloc,
      out_focal_precond_diag,
      out_focal_precond_diag_num_alloc,
      out_focal_precond_tril,
      out_focal_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar