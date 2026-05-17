#include "kernel_pinhole_split_fixed_pose_fixed_focal_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedPoseFixedFocalResJacFirstKernel(
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
        double* pose,
        unsigned int pose_num_alloc,
        double* focal,
        unsigned int focal_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* const out_rTr,
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

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60;

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
    r3 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        pose, 6 * pose_num_alloc, global_thread_idx, r7);
  };
  LoadShared<2, double, double>(
      point, 0 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r8, r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        pose, 0 * pose_num_alloc, global_thread_idx, r10, r11);
    ReadIdx2<1024, double, double, double2>(
        pose, 2 * pose_num_alloc, global_thread_idx, r12, r13);
    r14 = r11 * r12;
    r15 = 2.00000000000000000e+00;
    r14 = r14 * r15;
    r16 = r10 * r13;
    r16 = fma(r15, r16, r14);
    r7 = fma(r9, r16, r7);
    r17 = r10 * r12;
    r17 = r17 * r15;
    r18 = -2.00000000000000000e+00;
    r19 = r13 * r18;
    r20 = fma(r11, r19, r17);
  };
  LoadShared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r21);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r22 = r11 * r11;
    r22 = r18 * r22;
    r23 = 1.00000000000000000e+00;
    r24 = r10 * r10;
    r24 = fma(r18, r24, r23);
    r25 = r22 + r24;
    r7 = fma(r8, r20, r7);
    r7 = fma(r21, r25, r7);
    r26 = copysign(1.0, r7);
    r26 = fma(r3, r26, r7);
    r7 = 1.0 / r26;
    ReadIdx2<1024, double, double, double2>(
        focal, 0 * focal_num_alloc, global_thread_idx, r27, r28);
    ReadIdx2<1024, double, double, double2>(
        pose, 4 * pose_num_alloc, global_thread_idx, r29, r30);
    r31 = r12 * r13;
    r32 = r10 * r11;
    r32 = r32 * r15;
    r31 = fma(r15, r31, r32);
    r30 = fma(r8, r31, r30);
    r14 = fma(r10, r19, r14);
    r33 = r12 * r12;
    r33 = r18 * r33;
    r24 = r33 + r24;
    r30 = fma(r21, r14, r30);
    r30 = fma(r9, r24, r30);
    r30 = r28 * r30;
    r5 = fma(r7, r30, r5);
    r4 = fma(r4, r6, r2);
    r19 = fma(r12, r19, r32);
    r9 = fma(r9, r19, r29);
    r29 = r11 * r13;
    r29 = fma(r15, r29, r17);
    r33 = r23 + r33;
    r33 = r33 + r22;
    r9 = fma(r21, r29, r9);
    r9 = fma(r8, r33, r9);
    r9 = r27 * r9;
    r4 = fma(r7, r9, r4);
    r8 = fma(r0, r4, r1 * r5);
    ReadIdx1<1024, double, double, double>(
        weight_loss, 6 * weight_loss_num_alloc, global_thread_idx, r21);
    r22 = 0.00000000000000000e+00;
    r21 = fmax(r21, r22);
    r17 = sqrt(r21);
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 2 * weight_loss_num_alloc, global_thread_idx, r32, r2);
    r4 = fma(r32, r4, r2 * r5);
    r5 = fma(r4, r4, r8 * r8);
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 4 * weight_loss_num_alloc, global_thread_idx, r18, r34);
    r35 = 5.00000000000000000e-01;
    r34 = fmax(r34, r3);
    r36 = r34 * r34;
    r37 = r6 * r34;
    r38 = r15 * r34;
    r39 = fmax(r3, r5);
    r40 = sqrt(r39);
    r38 = fma(r40, r38, r34 * r37);
    r38 = r5 <= r36 ? r5 : r38;
    r37 = 2.50000000000000000e+00;
    r40 = r34 * r34;
    r41 = 1.0 / r36;
    r41 = fma(r5, r41, r23);
    r42 = log(r41);
    r40 = r40 * r42;
    r38 = r18 < r37 ? r40 : r38;
    r40 = 1.50000000000000000e+00;
    r42 = r15 * r34;
    r43 = sqrt(r41);
    r43 = r6 + r43;
    r42 = r42 * r34;
    r42 = r42 * r43;
    r38 = r18 < r40 ? r42 : r38;
    r38 = r18 < r35 ? r5 : r38;
    r42 = fmax(r22, r38);
    r43 = 1.0 / r39;
    r43 = r21 * r43;
    r44 = r42 * r43;
    r45 = sqrt(r44);
    r45 = r5 <= r3 ? r17 : r45;
    r17 = r8 * r45;
    r46 = r4 * r45;
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r17, r46);
    r46 = r8 * r8;
    r46 = r46 * r45;
    r17 = r4 * r4;
    r17 = r17 * r45;
    r17 = fma(r45, r17, r45 * r46);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r17);
  if (global_thread_idx < problem_size) {
    r17 = r6 * r8;
    r46 = 2.50000000000000000e-01;
    r47 = r5 <= r36 ? r22 : r22;
    r47 = r18 < r37 ? r22 : r47;
    r47 = r18 < r40 ? r22 : r47;
    r47 = r18 < r35 ? r22 : r47;
    r47 = r46 * r47;
    r44 = rsqrt(r44);
    r38 = copysign(1.0, r38);
    r38 = r23 + r38;
    r43 = r38 * r43;
    r47 = r47 * r44;
    r47 = r47 * r43;
    r47 = r5 <= r3 ? r22 : r47;
    r17 = r17 * r47;
    r42 = r21 * r42;
    r21 = -5.00000000000000000e-01;
    r38 = -1.00000000000000008e-15;
    r38 = r38 + r5;
    r38 = copysign(1.0, r38);
    r38 = r23 + r38;
    r23 = r39 * r39;
    r23 = 1.0 / r23;
    r42 = r42 * r21;
    r42 = r42 * r38;
    r42 = r42 * r23;
    r23 = r32 * r15;
    r21 = r15 * r8;
    r23 = fma(r0, r21, r4 * r23);
    r46 = r35 * r34;
    r39 = rsqrt(r39);
    r46 = r46 * r38;
    r46 = r46 * r39;
    r39 = r23 * r46;
    r39 = r5 <= r36 ? r23 : r39;
    r38 = 1.0 / r41;
    r48 = r23 * r38;
    r39 = r18 < r37 ? r48 : r39;
    r41 = rsqrt(r41);
    r48 = r23 * r41;
    r39 = r18 < r40 ? r48 : r39;
    r39 = r18 < r35 ? r23 : r39;
    r48 = r35 * r39;
    r48 = fma(r43, r48, r23 * r42);
    r48 = r35 * r48;
    r48 = r48 * r44;
    r48 = r5 <= r3 ? r22 : r48;
    r23 = fma(r8, r48, r17);
    r23 = fma(r0, r45, r23);
    r49 = r6 * r4;
    r49 = r49 * r47;
    r48 = fma(r4, r48, r49);
    r48 = fma(r32, r45, r48);
    WriteIdx2<1024, double, double, double2>(
        out_principal_point_jac,
        0 * out_principal_point_jac_num_alloc,
        global_thread_idx,
        r23,
        r48);
    r47 = r2 * r15;
    r47 = fma(r1, r21, r4 * r47);
    r50 = r47 * r46;
    r50 = r5 <= r36 ? r47 : r50;
    r51 = r47 * r38;
    r50 = r18 < r37 ? r51 : r50;
    r51 = r47 * r41;
    r50 = r18 < r40 ? r51 : r50;
    r50 = r18 < r35 ? r47 : r50;
    r51 = r35 * r50;
    r51 = fma(r43, r51, r47 * r42);
    r51 = r35 * r51;
    r51 = r51 * r44;
    r51 = r5 <= r3 ? r22 : r51;
    r47 = fma(r8, r51, r17);
    r47 = fma(r1, r45, r47);
    r51 = fma(r4, r51, r49);
    r51 = fma(r2, r45, r51);
    WriteIdx2<1024, double, double, double2>(
        out_principal_point_jac,
        2 * out_principal_point_jac_num_alloc,
        global_thread_idx,
        r47,
        r51);
    r52 = r8 * r23;
    r53 = r6 * r45;
    r54 = r4 * r53;
    r52 = fma(r48, r54, r53 * r52);
    r55 = r8 * r47;
    r55 = fma(r53, r55, r51 * r54);
    WriteSum2<double, double>((double*)inout_shared, r52, r55);
  };
  FlushSumShared<2, double>(out_principal_point_njtr,
                            0 * out_principal_point_njtr_num_alloc,
                            principal_point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r55 = fma(r23, r23, r48 * r48);
    r52 = fma(r47, r47, r51 * r51);
    WriteSum2<double, double>((double*)inout_shared, r55, r52);
  };
  FlushSumShared<2, double>(out_principal_point_precond_diag,
                            0 * out_principal_point_precond_diag_num_alloc,
                            principal_point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r51 = fma(r23, r47, r48 * r51);
    WriteSum1<double, double>((double*)inout_shared, r51);
  };
  FlushSumShared<1, double>(out_principal_point_precond_tril,
                            0 * out_principal_point_precond_tril_num_alloc,
                            principal_point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r51 = r28 * r31;
    r26 = r26 * r26;
    r26 = 1.0 / r26;
    r26 = r6 * r26;
    r30 = r30 * r26;
    r51 = fma(r20, r30, r7 * r51);
    r48 = r20 * r9;
    r52 = r27 * r33;
    r52 = fma(r7, r52, r26 * r48);
    r48 = fma(r0, r52, r1 * r51);
    r55 = fma(r45, r48, r17);
    r56 = r15 * r4;
    r52 = fma(r32, r52, r2 * r51);
    r48 = fma(r48, r21, r52 * r56);
    r56 = r48 * r46;
    r56 = r5 <= r36 ? r48 : r56;
    r51 = r48 * r38;
    r56 = r18 < r37 ? r51 : r56;
    r51 = r48 * r41;
    r56 = r18 < r40 ? r51 : r56;
    r56 = r18 < r35 ? r48 : r56;
    r51 = r35 * r56;
    r48 = fma(r48, r42, r43 * r51);
    r48 = r35 * r48;
    r48 = r48 * r44;
    r48 = r5 <= r3 ? r22 : r48;
    r55 = fma(r8, r48, r55);
    r52 = fma(r45, r52, r49);
    r52 = fma(r4, r48, r52);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             0 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r55,
                                             r52);
    r48 = r28 * r24;
    r48 = fma(r7, r48, r16 * r30);
    r51 = r16 * r9;
    r57 = r27 * r19;
    r57 = fma(r7, r57, r26 * r51);
    r51 = fma(r0, r57, r1 * r48);
    r58 = fma(r45, r51, r17);
    r59 = r15 * r4;
    r57 = fma(r32, r57, r2 * r48);
    r51 = fma(r51, r21, r57 * r59);
    r59 = r51 * r46;
    r59 = r5 <= r36 ? r51 : r59;
    r48 = r51 * r38;
    r59 = r18 < r37 ? r48 : r59;
    r48 = r51 * r41;
    r59 = r18 < r40 ? r48 : r59;
    r59 = r18 < r35 ? r51 : r59;
    r48 = r35 * r59;
    r51 = fma(r51, r42, r43 * r48);
    r51 = r35 * r51;
    r51 = r51 * r44;
    r51 = r5 <= r3 ? r22 : r51;
    r58 = fma(r8, r51, r58);
    r57 = fma(r45, r57, r49);
    r57 = fma(r4, r51, r57);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             2 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r58,
                                             r57);
    r51 = r15 * r4;
    r48 = r25 * r9;
    r60 = r27 * r29;
    r60 = fma(r7, r60, r26 * r48);
    r48 = r28 * r14;
    r48 = fma(r7, r48, r25 * r30);
    r30 = fma(r2, r48, r32 * r60);
    r48 = fma(r1, r48, r0 * r60);
    r21 = fma(r48, r21, r30 * r51);
    r46 = r21 * r46;
    r46 = r5 <= r36 ? r21 : r46;
    r38 = r21 * r38;
    r46 = r18 < r37 ? r38 : r46;
    r41 = r21 * r41;
    r46 = r18 < r40 ? r41 : r46;
    r46 = r18 < r35 ? r21 : r46;
    r18 = r35 * r46;
    r42 = fma(r21, r42, r43 * r18);
    r42 = r35 * r42;
    r42 = r42 * r44;
    r42 = r5 <= r3 ? r22 : r42;
    r17 = fma(r8, r42, r17);
    r17 = fma(r45, r48, r17);
    r30 = fma(r45, r30, r49);
    r30 = fma(r4, r42, r30);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             4 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r17,
                                             r30);
    r42 = r8 * r55;
    r42 = fma(r52, r54, r53 * r42);
    r45 = r8 * r58;
    r45 = fma(r53, r45, r57 * r54);
    WriteSum2<double, double>((double*)inout_shared, r42, r45);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r45 = r8 * r17;
    r54 = fma(r30, r54, r53 * r45);
    WriteSum1<double, double>((double*)inout_shared, r54);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r54 = fma(r52, r52, r55 * r55);
    r45 = fma(r57, r57, r58 * r58);
    WriteSum2<double, double>((double*)inout_shared, r54, r45);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r45 = fma(r30, r30, r17 * r17);
    WriteSum1<double, double>((double*)inout_shared, r45);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r45 = fma(r55, r58, r52 * r57);
    r52 = fma(r52, r30, r55 * r17);
    WriteSum2<double, double>((double*)inout_shared, r45, r52);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = fma(r57, r30, r58 * r17);
    WriteSum1<double, double>((double*)inout_shared, r30);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void PinholeSplitFixedPoseFixedFocalResJacFirst(
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
    double* pose,
    unsigned int pose_num_alloc,
    double* focal,
    unsigned int focal_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_rTr,
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
  PinholeSplitFixedPoseFixedFocalResJacFirstKernel<<<n_blocks, 1024>>>(
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
      pose,
      pose_num_alloc,
      focal,
      focal_num_alloc,
      out_res,
      out_res_num_alloc,
      out_rTr,
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