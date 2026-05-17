#include "kernel_pinhole_split_fixed_pose_fixed_principal_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedPoseFixedPrincipalPointResJacFirstKernel(
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
        double* pose,
        unsigned int pose_num_alloc,
        double* principal_point,
        unsigned int principal_point_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* const out_rTr,
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
      r61;

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
      focal, 0 * focal_num_alloc, focal_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, focal_indices_loc[threadIdx.x].target, r3, r7);
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
        pose, 2 * pose_num_alloc, global_thread_idx, r12, r13);
    r14 = r12 * r13;
    r15 = 2.00000000000000000e+00;
    ReadIdx2<1024, double, double, double2>(
        pose, 0 * pose_num_alloc, global_thread_idx, r16, r17);
    r18 = r16 * r17;
    r18 = r18 * r15;
    r14 = fma(r15, r14, r18);
    r9 = fma(r10, r14, r9);
  };
  LoadShared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r19);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r20 = r17 * r12;
    r20 = r20 * r15;
    r21 = -2.00000000000000000e+00;
    r22 = r13 * r21;
    r23 = fma(r16, r22, r20);
    r24 = r12 * r12;
    r24 = r21 * r24;
    r25 = 1.00000000000000000e+00;
    r26 = r16 * r16;
    r26 = fma(r21, r26, r25);
    r27 = r24 + r26;
    r9 = fma(r19, r23, r9);
    r9 = fma(r11, r27, r9);
    r28 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        pose, 6 * pose_num_alloc, global_thread_idx, r29);
    r30 = r16 * r13;
    r30 = fma(r15, r30, r20);
    r29 = fma(r11, r30, r29);
    r20 = r16 * r12;
    r20 = r20 * r15;
    r31 = fma(r17, r22, r20);
    r32 = r17 * r17;
    r32 = r21 * r32;
    r26 = r32 + r26;
    r29 = fma(r10, r31, r29);
    r29 = fma(r19, r26, r29);
    r21 = copysign(1.0, r29);
    r21 = fma(r28, r21, r29);
    r29 = 1.0 / r21;
    r33 = r9 * r29;
    r5 = fma(r7, r33, r5);
    r4 = fma(r4, r6, r2);
    r22 = fma(r12, r22, r18);
    r11 = fma(r11, r22, r8);
    r8 = r17 * r13;
    r8 = fma(r15, r8, r20);
    r24 = r25 + r24;
    r24 = r24 + r32;
    r11 = fma(r19, r8, r11);
    r11 = fma(r10, r24, r11);
    r10 = r3 * r11;
    r4 = fma(r29, r10, r4);
    r19 = fma(r0, r4, r1 * r5);
    ReadIdx1<1024, double, double, double>(
        weight_loss, 6 * weight_loss_num_alloc, global_thread_idx, r32);
    r20 = 0.00000000000000000e+00;
    r32 = fmax(r32, r20);
    r18 = sqrt(r32);
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
    r40 = r15 * r36;
    r41 = fmax(r28, r5);
    r42 = sqrt(r41);
    r40 = fma(r42, r40, r36 * r39);
    r40 = r5 <= r38 ? r5 : r40;
    r39 = 2.50000000000000000e+00;
    r42 = r36 * r36;
    r43 = 1.0 / r38;
    r43 = fma(r5, r43, r25);
    r44 = log(r43);
    r42 = r42 * r44;
    r40 = r35 < r39 ? r42 : r40;
    r42 = 1.50000000000000000e+00;
    r44 = r15 * r36;
    r45 = sqrt(r43);
    r45 = r6 + r45;
    r44 = r44 * r36;
    r44 = r44 * r45;
    r40 = r35 < r42 ? r44 : r40;
    r40 = r35 < r37 ? r5 : r40;
    r44 = fmax(r20, r40);
    r45 = 1.0 / r41;
    r45 = r32 * r45;
    r46 = r44 * r45;
    r47 = sqrt(r46);
    r47 = r5 <= r28 ? r18 : r47;
    r18 = r19 * r47;
    r48 = r4 * r47;
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r18, r48);
    r48 = r19 * r19;
    r48 = r48 * r47;
    r18 = r4 * r4;
    r18 = r18 * r47;
    r18 = fma(r47, r18, r47 * r48);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r18);
  if (global_thread_idx < problem_size) {
    r44 = r32 * r44;
    r32 = -5.00000000000000000e-01;
    r18 = -1.00000000000000008e-15;
    r18 = r18 + r5;
    r18 = copysign(1.0, r18);
    r18 = r25 + r18;
    r48 = r41 * r41;
    r48 = 1.0 / r48;
    r44 = r44 * r32;
    r44 = r44 * r18;
    r44 = r44 * r48;
    r48 = r0 * r11;
    r32 = r15 * r19;
    r48 = r48 * r29;
    r49 = r2 * r15;
    r49 = r49 * r11;
    r49 = r49 * r4;
    r49 = fma(r29, r49, r32 * r48);
    r48 = r37 * r36;
    r41 = rsqrt(r41);
    r48 = r48 * r18;
    r48 = r48 * r41;
    r41 = r49 * r48;
    r41 = r5 <= r38 ? r49 : r41;
    r18 = 1.0 / r43;
    r50 = r49 * r18;
    r41 = r35 < r39 ? r50 : r41;
    r43 = rsqrt(r43);
    r50 = r49 * r43;
    r41 = r35 < r42 ? r50 : r41;
    r41 = r35 < r37 ? r49 : r41;
    r50 = r37 * r41;
    r40 = copysign(1.0, r40);
    r40 = r25 + r40;
    r45 = r40 * r45;
    r50 = fma(r45, r50, r49 * r44);
    r50 = r37 * r50;
    r46 = rsqrt(r46);
    r50 = r50 * r46;
    r50 = r5 <= r28 ? r20 : r50;
    r49 = r6 * r19;
    r40 = 2.50000000000000000e-01;
    r25 = r5 <= r38 ? r20 : r20;
    r25 = r35 < r39 ? r20 : r25;
    r25 = r35 < r42 ? r20 : r25;
    r25 = r35 < r37 ? r20 : r25;
    r25 = r40 * r25;
    r25 = r25 * r46;
    r25 = r25 * r45;
    r25 = r5 <= r28 ? r20 : r25;
    r49 = r49 * r25;
    r40 = fma(r19, r50, r49);
    r51 = r0 * r11;
    r51 = r51 * r47;
    r40 = fma(r29, r51, r40);
    r51 = r6 * r4;
    r51 = r51 * r25;
    r50 = fma(r4, r50, r51);
    r25 = r2 * r11;
    r25 = r25 * r47;
    r50 = fma(r29, r25, r50);
    WriteIdx2<1024, double, double, double2>(out_focal_jac,
                                             0 * out_focal_jac_num_alloc,
                                             global_thread_idx,
                                             r40,
                                             r50);
    r25 = r1 * r33;
    r52 = r15 * r4;
    r53 = r34 * r33;
    r52 = fma(r53, r52, r32 * r25);
    r25 = r52 * r48;
    r25 = r5 <= r38 ? r52 : r25;
    r54 = r52 * r18;
    r25 = r35 < r39 ? r54 : r25;
    r54 = r52 * r43;
    r25 = r35 < r42 ? r54 : r25;
    r25 = r35 < r37 ? r52 : r25;
    r54 = r37 * r25;
    r54 = fma(r45, r54, r52 * r44);
    r54 = r37 * r54;
    r54 = r54 * r46;
    r54 = r5 <= r28 ? r20 : r54;
    r52 = fma(r19, r54, r49);
    r55 = r1 * r47;
    r52 = fma(r33, r55, r52);
    r54 = fma(r4, r54, r51);
    r54 = fma(r47, r53, r54);
    WriteIdx2<1024, double, double, double2>(out_focal_jac,
                                             2 * out_focal_jac_num_alloc,
                                             global_thread_idx,
                                             r52,
                                             r54);
    r53 = r6 * r47;
    r55 = r4 * r53;
    r56 = r19 * r40;
    r56 = fma(r53, r56, r50 * r55);
    r57 = r19 * r52;
    r57 = fma(r53, r57, r54 * r55);
    WriteSum2<double, double>((double*)inout_shared, r56, r57);
  };
  FlushSumShared<2, double>(out_focal_njtr,
                            0 * out_focal_njtr_num_alloc,
                            focal_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r57 = fma(r40, r40, r50 * r50);
    r56 = fma(r54, r54, r52 * r52);
    WriteSum2<double, double>((double*)inout_shared, r57, r56);
  };
  FlushSumShared<2, double>(out_focal_precond_diag,
                            0 * out_focal_precond_diag_num_alloc,
                            focal_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r54 = fma(r50, r54, r40 * r52);
    WriteSum1<double, double>((double*)inout_shared, r54);
  };
  FlushSumShared<1, double>(out_focal_precond_tril,
                            0 * out_focal_precond_tril_num_alloc,
                            focal_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r54 = r7 * r14;
    r9 = r7 * r9;
    r21 = r21 * r21;
    r21 = 1.0 / r21;
    r9 = r9 * r6;
    r9 = r9 * r21;
    r54 = fma(r31, r9, r29 * r54);
    r50 = r31 * r6;
    r50 = r50 * r21;
    r56 = r3 * r24;
    r56 = fma(r29, r56, r10 * r50);
    r50 = fma(r0, r56, r1 * r54);
    r57 = fma(r47, r50, r49);
    r58 = r15 * r4;
    r56 = fma(r2, r56, r34 * r54);
    r50 = fma(r50, r32, r56 * r58);
    r58 = r50 * r48;
    r58 = r5 <= r38 ? r50 : r58;
    r54 = r50 * r18;
    r58 = r35 < r39 ? r54 : r58;
    r54 = r50 * r43;
    r58 = r35 < r42 ? r54 : r58;
    r58 = r35 < r37 ? r50 : r58;
    r54 = r37 * r58;
    r50 = fma(r50, r44, r45 * r54);
    r50 = r37 * r50;
    r50 = r50 * r46;
    r50 = r5 <= r28 ? r20 : r50;
    r57 = fma(r19, r50, r57);
    r56 = fma(r47, r56, r51);
    r56 = fma(r4, r50, r56);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             0 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r57,
                                             r56);
    r50 = r7 * r27;
    r50 = fma(r29, r50, r30 * r9);
    r54 = r30 * r6;
    r54 = r54 * r21;
    r59 = r3 * r22;
    r59 = fma(r29, r59, r10 * r54);
    r54 = fma(r0, r59, r1 * r50);
    r60 = fma(r47, r54, r49);
    r61 = r15 * r4;
    r59 = fma(r2, r59, r34 * r50);
    r54 = fma(r54, r32, r59 * r61);
    r61 = r54 * r48;
    r61 = r5 <= r38 ? r54 : r61;
    r50 = r54 * r18;
    r61 = r35 < r39 ? r50 : r61;
    r50 = r54 * r43;
    r61 = r35 < r42 ? r50 : r61;
    r61 = r35 < r37 ? r54 : r61;
    r50 = r37 * r61;
    r54 = fma(r54, r44, r45 * r50);
    r54 = r37 * r54;
    r54 = r54 * r46;
    r54 = r5 <= r28 ? r20 : r54;
    r60 = fma(r19, r54, r60);
    r59 = fma(r47, r59, r51);
    r59 = fma(r4, r54, r59);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             2 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r60,
                                             r59);
    r54 = r15 * r4;
    r50 = r26 * r6;
    r50 = r50 * r21;
    r21 = r3 * r8;
    r21 = fma(r29, r21, r10 * r50);
    r50 = r7 * r23;
    r50 = fma(r29, r50, r26 * r9);
    r34 = fma(r34, r50, r2 * r21);
    r50 = fma(r1, r50, r0 * r21);
    r32 = fma(r50, r32, r34 * r54);
    r48 = r32 * r48;
    r48 = r5 <= r38 ? r32 : r48;
    r18 = r32 * r18;
    r48 = r35 < r39 ? r18 : r48;
    r43 = r32 * r43;
    r48 = r35 < r42 ? r43 : r48;
    r48 = r35 < r37 ? r32 : r48;
    r35 = r37 * r48;
    r44 = fma(r32, r44, r45 * r35);
    r44 = r37 * r44;
    r44 = r44 * r46;
    r44 = r5 <= r28 ? r20 : r44;
    r49 = fma(r19, r44, r49);
    r49 = fma(r47, r50, r49);
    r34 = fma(r47, r34, r51);
    r34 = fma(r4, r44, r34);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             4 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r49,
                                             r34);
    r44 = r19 * r57;
    r44 = fma(r56, r55, r53 * r44);
    r51 = r19 * r60;
    r51 = fma(r53, r51, r59 * r55);
    WriteSum2<double, double>((double*)inout_shared, r44, r51);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r51 = r19 * r49;
    r55 = fma(r34, r55, r53 * r51);
    WriteSum1<double, double>((double*)inout_shared, r55);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r55 = fma(r56, r56, r57 * r57);
    r51 = fma(r59, r59, r60 * r60);
    WriteSum2<double, double>((double*)inout_shared, r55, r51);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r51 = fma(r34, r34, r49 * r49);
    WriteSum1<double, double>((double*)inout_shared, r51);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r51 = fma(r57, r60, r56 * r59);
    r56 = fma(r56, r34, r57 * r49);
    WriteSum2<double, double>((double*)inout_shared, r51, r56);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = fma(r59, r34, r60 * r49);
    WriteSum1<double, double>((double*)inout_shared, r34);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void PinholeSplitFixedPoseFixedPrincipalPointResJacFirst(
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
    double* pose,
    unsigned int pose_num_alloc,
    double* principal_point,
    unsigned int principal_point_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_rTr,
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
  PinholeSplitFixedPoseFixedPrincipalPointResJacFirstKernel<<<n_blocks, 1024>>>(
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
      pose,
      pose_num_alloc,
      principal_point,
      principal_point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_rTr,
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