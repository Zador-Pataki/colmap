#include "kernel_pinhole_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) PinholeFixedPointResJacFirstKernel(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
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
    double* const out_rTr,
    double* out_pose_jac,
    unsigned int out_pose_jac_num_alloc,
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    double* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
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

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex calib_indices_loc[1024];
  calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_indices[global_thread_idx]
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
    r4 = fma(r4, r6, r2);
  };
  LoadShared<2, double, double>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r2, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        point, 0 * point_num_alloc, global_thread_idx, r8, r9);
  };
  LoadShared<2, double, double>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
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
      pose, 2 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r14, r15);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r16 = -2.00000000000000000e+00;
    r17 = r14 * r16;
    r18 = r15 * r17;
    r19 = r12 + r18;
    r2 = fma(r9, r19, r2);
    ReadIdx1<1024, double, double, double>(
        point, 2 * point_num_alloc, global_thread_idx, r20);
    r21 = r10 * r14;
    r21 = r21 * r13;
    r22 = r11 * r15;
    r22 = r22 * r13;
    r23 = r21 + r22;
    r24 = r14 * r17;
    r25 = 1.00000000000000000e+00;
    r26 = r11 * r11;
    r27 = fma(r16, r26, r25);
    r28 = r24 + r27;
    r2 = fma(r20, r23, r2);
    r2 = fma(r8, r28, r2);
  };
  LoadShared<2, double, double>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r28, r29);
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
    r32 = r11 * r14;
    r32 = r32 * r13;
    r33 = r10 * r15;
    r33 = r33 * r13;
    r34 = r32 + r33;
    r31 = fma(r9, r34, r31);
    r35 = r11 * r15;
    r35 = r35 * r16;
    r21 = r21 + r35;
    r36 = r10 * r10;
    r37 = r16 * r36;
    r27 = r37 + r27;
    r31 = fma(r8, r21, r31);
    r31 = fma(r20, r27, r31);
    r27 = copysign(1.0, r31);
    r27 = fma(r30, r27, r31);
    r31 = 1.0 / r27;
    r38 = r28 * r31;
    r4 = fma(r2, r38, r4);
    r5 = fma(r5, r6, r3);
    r3 = r14 * r15;
    r3 = r3 * r13;
    r12 = r12 + r3;
    r7 = fma(r8, r12, r7);
    r39 = r10 * r15;
    r39 = r39 * r16;
    r32 = r32 + r39;
    r24 = r25 + r24;
    r24 = r24 + r37;
    r7 = fma(r20, r32, r7);
    r7 = fma(r9, r24, r7);
    r24 = r29 * r7;
    r5 = fma(r31, r24, r5);
    r37 = fma(r1, r5, r0 * r4);
    ReadIdx1<1024, double, double, double>(
        weight_loss, 6 * weight_loss_num_alloc, global_thread_idx, r40);
    r41 = 0.00000000000000000e+00;
    r40 = fmax(r40, r41);
    r42 = sqrt(r40);
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 2 * weight_loss_num_alloc, global_thread_idx, r43, r44);
    r5 = fma(r44, r5, r43 * r4);
    r4 = fma(r37, r37, r5 * r5);
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 4 * weight_loss_num_alloc, global_thread_idx, r45, r46);
    r47 = 5.00000000000000000e-01;
    r46 = fmax(r46, r30);
    r48 = r46 * r46;
    r49 = r13 * r46;
    r50 = fmax(r30, r4);
    r51 = sqrt(r50);
    r52 = r6 * r46;
    r52 = fma(r46, r52, r51 * r49);
    r52 = r4 <= r48 ? r4 : r52;
    r49 = 2.50000000000000000e+00;
    r51 = r46 * r46;
    r53 = 1.0 / r48;
    r53 = fma(r4, r53, r25);
    r54 = log(r53);
    r51 = r51 * r54;
    r52 = r45 < r49 ? r51 : r52;
    r51 = 1.50000000000000000e+00;
    r54 = r13 * r46;
    r55 = sqrt(r53);
    r55 = r6 + r55;
    r54 = r54 * r46;
    r54 = r54 * r55;
    r52 = r45 < r51 ? r54 : r52;
    r52 = r45 < r47 ? r4 : r52;
    r54 = fmax(r41, r52);
    r55 = 1.0 / r50;
    r55 = r40 * r55;
    r56 = r54 * r55;
    r57 = sqrt(r56);
    r57 = r4 <= r30 ? r42 : r57;
    r42 = r37 * r57;
    r58 = r5 * r57;
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r42, r58);
    r58 = r37 * r37;
    r58 = r58 * r57;
    r42 = r5 * r5;
    r42 = r42 * r57;
    r42 = fma(r57, r42, r57 * r58);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r42);
  if (global_thread_idx < problem_size) {
    r42 = r6 * r37;
    r58 = 2.50000000000000000e-01;
    r59 = r4 <= r48 ? r41 : r41;
    r59 = r45 < r49 ? r41 : r59;
    r59 = r45 < r51 ? r41 : r59;
    r59 = r45 < r47 ? r41 : r59;
    r59 = r58 * r59;
    r56 = rsqrt(r56);
    r52 = copysign(1.0, r52);
    r52 = r25 + r52;
    r55 = r52 * r55;
    r59 = r59 * r56;
    r59 = r59 * r55;
    r59 = r4 <= r30 ? r41 : r59;
    r42 = r42 * r59;
    r52 = r13 * r37;
    r58 = r10 * r11;
    r58 = r58 * r16;
    r3 = r3 + r58;
    r23 = fma(r9, r23, r20 * r3);
    r11 = r11 * r17;
    r39 = r39 + r11;
    r3 = r14 * r14;
    r16 = r6 * r26;
    r60 = r3 + r16;
    r15 = r15 * r15;
    r61 = r6 * r36;
    r62 = r15 + r61;
    r63 = r60 + r62;
    r63 = fma(r9, r63, r20 * r39);
    r28 = r28 * r2;
    r27 = r27 * r27;
    r27 = 1.0 / r27;
    r28 = r28 * r6;
    r28 = r28 * r27;
    r23 = fma(r63, r28, r23 * r38);
    r39 = r6 * r63;
    r39 = r39 * r27;
    r64 = r6 * r15;
    r65 = r36 + r64;
    r60 = r60 + r65;
    r60 = fma(r20, r60, r9 * r32);
    r32 = r29 * r60;
    r32 = fma(r31, r32, r24 * r39);
    r39 = fma(r1, r32, r0 * r23);
    r32 = fma(r44, r32, r43 * r23);
    r23 = r13 * r5;
    r52 = fma(r32, r23, r39 * r52);
    r66 = r47 * r46;
    r67 = -1.00000000000000008e-15;
    r67 = r67 + r4;
    r67 = copysign(1.0, r67);
    r67 = r25 + r67;
    r25 = rsqrt(r50);
    r66 = r66 * r67;
    r66 = r66 * r25;
    r25 = r52 * r66;
    r25 = r4 <= r48 ? r52 : r25;
    r68 = 1.0 / r53;
    r69 = r52 * r68;
    r25 = r45 < r49 ? r69 : r25;
    r53 = rsqrt(r53);
    r69 = r52 * r53;
    r25 = r45 < r51 ? r69 : r25;
    r25 = r45 < r47 ? r52 : r25;
    r69 = r47 * r25;
    r54 = r40 * r54;
    r40 = -5.00000000000000000e-01;
    r50 = r50 * r50;
    r50 = 1.0 / r50;
    r54 = r54 * r40;
    r54 = r54 * r67;
    r54 = r54 * r50;
    r52 = fma(r52, r54, r55 * r69);
    r52 = r47 * r52;
    r52 = r52 * r56;
    r52 = r4 <= r30 ? r41 : r52;
    r69 = fma(r37, r52, r42);
    r69 = fma(r57, r39, r69);
    r39 = r6 * r5;
    r39 = r39 * r59;
    r52 = fma(r5, r52, r39);
    r52 = fma(r57, r32, r52);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r69, r52);
    r32 = r13 * r37;
    r14 = r14 * r14;
    r14 = r14 * r6;
    r59 = r26 + r14;
    r65 = r65 + r59;
    r65 = fma(r8, r65, r20 * r21);
    r17 = r10 * r17;
    r35 = r35 + r17;
    r15 = r36 + r15;
    r15 = r15 + r16;
    r15 = r15 + r14;
    r15 = fma(r20, r15, r8 * r35);
    r15 = fma(r15, r38, r65 * r28);
    r11 = r33 + r11;
    r11 = fma(r8, r11, r20 * r12);
    r12 = r29 * r11;
    r20 = r6 * r65;
    r20 = r20 * r27;
    r20 = fma(r24, r20, r31 * r12);
    r12 = fma(r1, r20, r0 * r15);
    r20 = fma(r44, r20, r43 * r15);
    r32 = fma(r20, r23, r12 * r32);
    r15 = r32 * r66;
    r15 = r4 <= r48 ? r32 : r15;
    r33 = r32 * r68;
    r15 = r45 < r49 ? r33 : r15;
    r33 = r32 * r53;
    r15 = r45 < r51 ? r33 : r15;
    r15 = r45 < r47 ? r32 : r15;
    r33 = r47 * r15;
    r32 = fma(r32, r54, r55 * r33);
    r32 = r47 * r32;
    r32 = r32 * r56;
    r32 = r4 <= r30 ? r41 : r32;
    r33 = fma(r37, r32, r42);
    r33 = fma(r57, r12, r33);
    r20 = fma(r57, r20, r39);
    r20 = fma(r5, r32, r20);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r33, r20);
    r3 = r26 + r3;
    r3 = r3 + r61;
    r3 = r3 + r64;
    r3 = fma(r9, r3, r8 * r19);
    r17 = r22 + r17;
    r17 = fma(r9, r17, r8 * r34);
    r3 = fma(r17, r28, r3 * r38);
    r18 = r58 + r18;
    r59 = r62 + r59;
    r59 = fma(r8, r59, r9 * r18);
    r8 = r29 * r59;
    r18 = r6 * r17;
    r18 = r18 * r27;
    r18 = fma(r24, r18, r31 * r8);
    r8 = fma(r1, r18, r0 * r3);
    r9 = fma(r57, r8, r42);
    r62 = r13 * r37;
    r18 = fma(r44, r18, r43 * r3);
    r62 = fma(r18, r23, r8 * r62);
    r8 = r62 * r66;
    r8 = r4 <= r48 ? r62 : r8;
    r3 = r62 * r68;
    r8 = r45 < r49 ? r3 : r8;
    r3 = r62 * r53;
    r8 = r45 < r51 ? r3 : r8;
    r8 = r45 < r47 ? r62 : r8;
    r3 = r47 * r8;
    r3 = fma(r55, r3, r62 * r54);
    r3 = r47 * r3;
    r3 = r3 * r56;
    r3 = r4 <= r30 ? r41 : r3;
    r9 = fma(r37, r3, r9);
    r18 = fma(r57, r18, r39);
    r18 = fma(r5, r3, r18);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r9, r18);
    r3 = r43 * r13;
    r3 = r3 * r5;
    r62 = r13 * r37;
    r58 = r0 * r38;
    r62 = fma(r58, r62, r38 * r3);
    r34 = r62 * r66;
    r34 = r4 <= r48 ? r62 : r34;
    r22 = r62 * r68;
    r34 = r45 < r49 ? r22 : r34;
    r22 = r62 * r53;
    r34 = r45 < r51 ? r22 : r34;
    r34 = r45 < r47 ? r62 : r34;
    r22 = r47 * r34;
    r22 = fma(r55, r22, r62 * r54);
    r22 = r47 * r22;
    r22 = r22 * r56;
    r22 = r4 <= r30 ? r41 : r22;
    r62 = fma(r37, r22, r42);
    r62 = fma(r57, r58, r62);
    r22 = fma(r5, r22, r39);
    r58 = r43 * r57;
    r22 = fma(r38, r58, r22);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r62, r22);
    r58 = r29 * r44;
    r58 = r58 * r31;
    r38 = r29 * r1;
    r38 = r38 * r13;
    r38 = r38 * r37;
    r38 = fma(r31, r38, r23 * r58);
    r58 = r38 * r66;
    r58 = r4 <= r48 ? r38 : r58;
    r19 = r38 * r68;
    r58 = r45 < r49 ? r19 : r58;
    r19 = r38 * r53;
    r58 = r45 < r51 ? r19 : r58;
    r58 = r45 < r47 ? r38 : r58;
    r19 = r47 * r58;
    r38 = fma(r38, r54, r55 * r19);
    r38 = r47 * r38;
    r38 = r38 * r56;
    r38 = r4 <= r30 ? r41 : r38;
    r19 = fma(r37, r38, r42);
    r64 = r29 * r1;
    r64 = r64 * r57;
    r19 = fma(r31, r64, r19);
    r38 = fma(r5, r38, r39);
    r64 = r29 * r44;
    r64 = r64 * r57;
    r38 = fma(r31, r64, r38);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r19, r38);
    r64 = r1 * r6;
    r64 = r64 * r27;
    r64 = fma(r0, r28, r24 * r64);
    r61 = fma(r57, r64, r42);
    r26 = r13 * r37;
    r32 = r44 * r6;
    r32 = r32 * r27;
    r28 = fma(r43, r28, r24 * r32);
    r26 = fma(r28, r23, r64 * r26);
    r64 = r26 * r66;
    r64 = r4 <= r48 ? r26 : r64;
    r32 = r26 * r68;
    r64 = r45 < r49 ? r32 : r64;
    r32 = r26 * r53;
    r64 = r45 < r51 ? r32 : r64;
    r64 = r45 < r47 ? r26 : r64;
    r32 = r47 * r64;
    r26 = fma(r26, r54, r55 * r32);
    r26 = r47 * r26;
    r26 = r26 * r56;
    r26 = r4 <= r30 ? r41 : r26;
    r61 = fma(r37, r26, r61);
    r26 = fma(r5, r26, r39);
    r26 = fma(r57, r28, r26);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r61, r26);
    r28 = r6 * r57;
    r32 = r37 * r28;
    r24 = r5 * r52;
    r24 = fma(r28, r24, r69 * r32);
    r27 = r5 * r20;
    r27 = fma(r33, r32, r28 * r27);
    WriteSum2<double, double>((double*)inout_shared, r24, r27);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = r5 * r18;
    r27 = fma(r9, r32, r28 * r27);
    r24 = r5 * r22;
    r24 = fma(r28, r24, r62 * r32);
    WriteSum2<double, double>((double*)inout_shared, r27, r24);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = r5 * r38;
    r24 = fma(r19, r32, r28 * r24);
    r27 = r5 * r26;
    r27 = fma(r61, r32, r28 * r27);
    WriteSum2<double, double>((double*)inout_shared, r24, r27);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fma(r69, r69, r52 * r52);
    r24 = fma(r20, r20, r33 * r33);
    WriteSum2<double, double>((double*)inout_shared, r27, r24);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = fma(r9, r9, r18 * r18);
    r27 = fma(r22, r22, r62 * r62);
    WriteSum2<double, double>((double*)inout_shared, r24, r27);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fma(r38, r38, r19 * r19);
    r24 = fma(r26, r26, r61 * r61);
    WriteSum2<double, double>((double*)inout_shared, r27, r24);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = fma(r69, r33, r52 * r20);
    r27 = fma(r69, r9, r52 * r18);
    WriteSum2<double, double>((double*)inout_shared, r24, r27);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fma(r52, r22, r69 * r62);
    r24 = fma(r52, r38, r69 * r19);
    WriteSum2<double, double>((double*)inout_shared, r27, r24);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r69 = fma(r69, r61, r52 * r26);
    r24 = fma(r20, r18, r33 * r9);
    WriteSum2<double, double>((double*)inout_shared, r69, r24);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = fma(r33, r62, r20 * r22);
    r69 = fma(r33, r19, r20 * r38);
    WriteSum2<double, double>((double*)inout_shared, r24, r69);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = fma(r20, r26, r33 * r61);
    r69 = fma(r18, r22, r9 * r62);
    WriteSum2<double, double>((double*)inout_shared, r33, r69);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r69 = fma(r9, r19, r18 * r38);
    r9 = fma(r9, r61, r18 * r26);
    WriteSum2<double, double>((double*)inout_shared, r69, r9);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r9 = fma(r62, r19, r22 * r38);
    r62 = fma(r22, r26, r62 * r61);
    WriteSum2<double, double>((double*)inout_shared, r9, r62);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r61 = fma(r38, r26, r19 * r61);
    WriteSum1<double, double>((double*)inout_shared, r61);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r61 = r0 * r13;
    r61 = r61 * r2;
    r61 = r61 * r37;
    r19 = r2 * r31;
    r19 = fma(r3, r19, r31 * r61);
    r61 = r19 * r66;
    r61 = r4 <= r48 ? r19 : r61;
    r62 = r19 * r68;
    r61 = r45 < r49 ? r62 : r61;
    r62 = r19 * r53;
    r61 = r45 < r51 ? r62 : r61;
    r61 = r45 < r47 ? r19 : r61;
    r62 = r47 * r61;
    r19 = fma(r19, r54, r55 * r62);
    r19 = r47 * r19;
    r19 = r19 * r56;
    r19 = r4 <= r30 ? r41 : r19;
    r62 = fma(r37, r19, r42);
    r9 = r0 * r2;
    r9 = r9 * r57;
    r62 = fma(r31, r9, r62);
    r19 = fma(r5, r19, r39);
    r9 = r43 * r2;
    r9 = r9 * r57;
    r19 = fma(r31, r9, r19);
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             0 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r62,
                                             r19);
    r9 = r1 * r13;
    r9 = r9 * r7;
    r9 = r9 * r37;
    r69 = r44 * r7;
    r69 = r69 * r31;
    r69 = fma(r23, r69, r31 * r9);
    r9 = r69 * r66;
    r9 = r4 <= r48 ? r69 : r9;
    r33 = r69 * r68;
    r9 = r45 < r49 ? r33 : r9;
    r33 = r69 * r53;
    r9 = r45 < r51 ? r33 : r9;
    r9 = r45 < r47 ? r69 : r9;
    r33 = r47 * r9;
    r33 = fma(r55, r33, r69 * r54);
    r33 = r47 * r33;
    r33 = r33 * r56;
    r33 = r4 <= r30 ? r41 : r33;
    r69 = fma(r37, r33, r42);
    r24 = r1 * r7;
    r24 = r24 * r57;
    r69 = fma(r31, r24, r69);
    r33 = fma(r5, r33, r39);
    r24 = r44 * r7;
    r24 = r24 * r57;
    r33 = fma(r31, r24, r33);
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             2 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r69,
                                             r33);
    r24 = fma(r0, r57, r42);
    r27 = r0 * r13;
    r27 = fma(r37, r27, r3);
    r3 = r27 * r66;
    r3 = r4 <= r48 ? r27 : r3;
    r12 = r27 * r68;
    r3 = r45 < r49 ? r12 : r3;
    r12 = r27 * r53;
    r3 = r45 < r51 ? r12 : r3;
    r3 = r45 < r47 ? r27 : r3;
    r12 = r47 * r3;
    r27 = fma(r27, r54, r55 * r12);
    r27 = r47 * r27;
    r27 = r27 * r56;
    r27 = r4 <= r30 ? r41 : r27;
    r24 = fma(r37, r27, r24);
    r12 = fma(r43, r57, r39);
    r12 = fma(r5, r27, r12);
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             4 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r24,
                                             r12);
    r27 = r1 * r13;
    r23 = fma(r44, r23, r37 * r27);
    r66 = r23 * r66;
    r66 = r4 <= r48 ? r23 : r66;
    r68 = r23 * r68;
    r66 = r45 < r49 ? r68 : r66;
    r53 = r23 * r53;
    r66 = r45 < r51 ? r53 : r66;
    r66 = r45 < r47 ? r23 : r66;
    r45 = r47 * r66;
    r45 = fma(r55, r45, r23 * r54);
    r45 = r47 * r45;
    r45 = r45 * r56;
    r45 = r4 <= r30 ? r41 : r45;
    r42 = fma(r37, r45, r42);
    r42 = fma(r1, r57, r42);
    r45 = fma(r5, r45, r39);
    r45 = fma(r44, r57, r45);
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             6 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r42,
                                             r45);
    r39 = r5 * r19;
    r39 = fma(r28, r39, r62 * r32);
    r41 = r5 * r33;
    r41 = fma(r28, r41, r69 * r32);
    WriteSum2<double, double>((double*)inout_shared, r39, r41);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            0 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = r5 * r12;
    r41 = fma(r24, r32, r28 * r41);
    r39 = r5 * r45;
    r39 = fma(r28, r39, r42 * r32);
    WriteSum2<double, double>((double*)inout_shared, r41, r39);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            2 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = fma(r62, r62, r19 * r19);
    r41 = fma(r69, r69, r33 * r33);
    WriteSum2<double, double>((double*)inout_shared, r39, r41);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            0 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = fma(r24, r24, r12 * r12);
    r39 = fma(r45, r45, r42 * r42);
    WriteSum2<double, double>((double*)inout_shared, r41, r39);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            2 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = fma(r19, r33, r62 * r69);
    r41 = fma(r19, r12, r62 * r24);
    WriteSum2<double, double>((double*)inout_shared, r39, r41);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            0 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r62 = fma(r62, r42, r19 * r45);
    r41 = fma(r69, r24, r33 * r12);
    WriteSum2<double, double>((double*)inout_shared, r62, r41);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            2 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r69 = fma(r69, r42, r33 * r45);
    r42 = fma(r24, r42, r12 * r45);
    WriteSum2<double, double>((double*)inout_shared, r69, r42);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            4 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void PinholeFixedPointResJacFirst(double* pose,
                                  unsigned int pose_num_alloc,
                                  SharedIndex* pose_indices,
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
                                  double* const out_rTr,
                                  double* out_pose_jac,
                                  unsigned int out_pose_jac_num_alloc,
                                  double* const out_pose_njtr,
                                  unsigned int out_pose_njtr_num_alloc,
                                  double* const out_pose_precond_diag,
                                  unsigned int out_pose_precond_diag_num_alloc,
                                  double* const out_pose_precond_tril,
                                  unsigned int out_pose_precond_tril_num_alloc,
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
  PinholeFixedPointResJacFirstKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
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
      out_rTr,
      out_pose_jac,
      out_pose_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_pose_precond_diag,
      out_pose_precond_diag_num_alloc,
      out_pose_precond_tril,
      out_pose_precond_tril_num_alloc,
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