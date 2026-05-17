#include "kernel_pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedPoseFixedFocalFixedPrincipalPointResJacKernel(
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
        double* principal_point,
        unsigned int principal_point_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* const out_point_njtr,
        unsigned int out_point_njtr_num_alloc,
        double* const out_point_precond_diag,
        unsigned int out_point_precond_diag_num_alloc,
        double* const out_point_precond_tril,
        unsigned int out_point_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54;

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
    r14 = r10 * r13;
    r15 = 2.00000000000000000e+00;
    r16 = r12 * r15;
    r17 = r11 * r16;
    r14 = fma(r15, r14, r17);
    r7 = fma(r9, r14, r7);
    r18 = r10 * r16;
    r19 = -2.00000000000000000e+00;
    r20 = r13 * r19;
    r21 = fma(r11, r20, r18);
  };
  LoadShared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r22);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r23 = r11 * r11;
    r23 = r19 * r23;
    r24 = 1.00000000000000000e+00;
    r25 = r10 * r10;
    r25 = fma(r19, r25, r24);
    r26 = r23 + r25;
    r7 = fma(r8, r21, r7);
    r7 = fma(r22, r26, r7);
    r27 = copysign(1.0, r7);
    r27 = fma(r3, r27, r7);
    r7 = 1.0 / r27;
    ReadIdx2<1024, double, double, double2>(
        focal, 0 * focal_num_alloc, global_thread_idx, r28, r29);
    ReadIdx2<1024, double, double, double2>(
        pose, 4 * pose_num_alloc, global_thread_idx, r30, r31);
    r32 = r10 * r11;
    r32 = r32 * r15;
    r16 = fma(r13, r16, r32);
    r31 = fma(r8, r16, r31);
    r17 = fma(r10, r20, r17);
    r33 = r12 * r12;
    r33 = r33 * r19;
    r25 = r33 + r25;
    r31 = fma(r22, r17, r31);
    r31 = fma(r9, r25, r31);
    r31 = r29 * r31;
    r5 = fma(r7, r31, r5);
    r4 = fma(r4, r6, r2);
    r20 = fma(r12, r20, r32);
    r9 = fma(r9, r20, r30);
    r30 = r11 * r13;
    r30 = fma(r15, r30, r18);
    r33 = r24 + r33;
    r33 = r33 + r23;
    r9 = fma(r22, r30, r9);
    r9 = fma(r8, r33, r9);
    r9 = r28 * r9;
    r4 = fma(r7, r9, r4);
    r8 = fma(r0, r4, r1 * r5);
    ReadIdx1<1024, double, double, double>(
        weight_loss, 6 * weight_loss_num_alloc, global_thread_idx, r22);
    r23 = 0.00000000000000000e+00;
    r22 = fmax(r22, r23);
    r18 = sqrt(r22);
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 2 * weight_loss_num_alloc, global_thread_idx, r12, r32);
    r4 = fma(r12, r4, r32 * r5);
    r5 = fma(r4, r4, r8 * r8);
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 4 * weight_loss_num_alloc, global_thread_idx, r2, r19);
    r34 = 5.00000000000000000e-01;
    r19 = fmax(r19, r3);
    r35 = r19 * r19;
    r36 = r15 * r19;
    r37 = fmax(r3, r5);
    r38 = sqrt(r37);
    r36 = fma(r38, r36, r6 * r35);
    r36 = r5 <= r35 ? r5 : r36;
    r38 = 2.50000000000000000e+00;
    r39 = 1.0 / r35;
    r39 = fma(r5, r39, r24);
    r40 = log(r39);
    r40 = r40 * r35;
    r36 = r2 < r38 ? r40 : r36;
    r40 = 1.50000000000000000e+00;
    r41 = sqrt(r39);
    r41 = r6 + r41;
    r41 = r15 * r41;
    r41 = r41 * r35;
    r36 = r2 < r40 ? r41 : r36;
    r36 = r2 < r34 ? r5 : r36;
    r41 = fmax(r23, r36);
    r42 = 1.0 / r37;
    r42 = r22 * r42;
    r43 = r41 * r42;
    r44 = sqrt(r43);
    r44 = r5 <= r3 ? r18 : r44;
    r18 = r8 * r44;
    r45 = r4 * r44;
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r18, r45);
    r45 = r6 * r4;
    r46 = r29 * r16;
    r27 = r27 * r27;
    r27 = 1.0 / r27;
    r27 = r6 * r27;
    r31 = r27 * r31;
    r46 = fma(r21, r31, r7 * r46);
    r47 = r21 * r27;
    r48 = r28 * r33;
    r48 = fma(r7, r48, r9 * r47);
    r47 = fma(r12, r48, r32 * r46);
    r49 = r6 * r4;
    r50 = 2.50000000000000000e-01;
    r51 = r5 <= r35 ? r23 : r23;
    r51 = r2 < r38 ? r23 : r51;
    r51 = r2 < r40 ? r23 : r51;
    r51 = r2 < r34 ? r23 : r51;
    r51 = r50 * r51;
    r43 = rsqrt(r43);
    r36 = copysign(1.0, r36);
    r36 = r24 + r36;
    r42 = r36 * r42;
    r51 = r51 * r43;
    r51 = r51 * r42;
    r51 = r5 <= r3 ? r23 : r51;
    r49 = r49 * r51;
    r36 = fma(r44, r47, r49);
    r50 = r15 * r8;
    r48 = fma(r0, r48, r1 * r46);
    r46 = r15 * r4;
    r46 = fma(r47, r46, r48 * r50);
    r50 = r34 * r19;
    r47 = -1.00000000000000008e-15;
    r47 = r47 + r5;
    r47 = copysign(1.0, r47);
    r47 = r24 + r47;
    r24 = rsqrt(r37);
    r50 = r50 * r47;
    r50 = r50 * r24;
    r24 = r46 * r50;
    r24 = r5 <= r35 ? r46 : r24;
    r52 = 1.0 / r39;
    r53 = r46 * r52;
    r24 = r2 < r38 ? r53 : r24;
    r39 = rsqrt(r39);
    r53 = r46 * r39;
    r24 = r2 < r40 ? r53 : r24;
    r24 = r2 < r34 ? r46 : r24;
    r53 = r34 * r24;
    r41 = r22 * r41;
    r22 = -5.00000000000000000e-01;
    r37 = r37 * r37;
    r37 = 1.0 / r37;
    r41 = r41 * r22;
    r41 = r41 * r47;
    r41 = r41 * r37;
    r46 = fma(r46, r41, r42 * r53);
    r46 = r34 * r46;
    r46 = r46 * r43;
    r46 = r5 <= r3 ? r23 : r46;
    r36 = fma(r4, r46, r36);
    r45 = r45 * r44;
    r53 = r6 * r8;
    r53 = r53 * r51;
    r48 = fma(r44, r48, r53);
    r48 = fma(r8, r46, r48);
    r18 = r6 * r18;
    r45 = fma(r48, r18, r36 * r45);
    r46 = r6 * r4;
    r51 = r29 * r25;
    r51 = fma(r7, r51, r14 * r31);
    r37 = r14 * r27;
    r47 = r28 * r20;
    r47 = fma(r7, r47, r9 * r37);
    r37 = fma(r12, r47, r32 * r51);
    r22 = fma(r44, r37, r49);
    r54 = r15 * r8;
    r47 = fma(r0, r47, r1 * r51);
    r51 = r15 * r4;
    r51 = fma(r37, r51, r47 * r54);
    r54 = r51 * r50;
    r54 = r5 <= r35 ? r51 : r54;
    r37 = r51 * r52;
    r54 = r2 < r38 ? r37 : r54;
    r37 = r51 * r39;
    r54 = r2 < r40 ? r37 : r54;
    r54 = r2 < r34 ? r51 : r54;
    r37 = r34 * r54;
    r51 = fma(r51, r41, r42 * r37);
    r51 = r34 * r51;
    r51 = r51 * r43;
    r51 = r5 <= r3 ? r23 : r51;
    r22 = fma(r4, r51, r22);
    r46 = r46 * r44;
    r47 = fma(r44, r47, r53);
    r47 = fma(r8, r51, r47);
    r46 = fma(r47, r18, r22 * r46);
    WriteSum2<double, double>((double*)inout_shared, r45, r46);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = r6 * r4;
    r45 = r26 * r27;
    r51 = r28 * r30;
    r51 = fma(r7, r51, r9 * r45);
    r45 = r29 * r17;
    r45 = fma(r7, r45, r26 * r31);
    r32 = fma(r32, r45, r12 * r51);
    r49 = fma(r44, r32, r49);
    r12 = r15 * r8;
    r45 = fma(r1, r45, r0 * r51);
    r1 = r15 * r4;
    r1 = fma(r32, r1, r45 * r12);
    r50 = r1 * r50;
    r50 = r5 <= r35 ? r1 : r50;
    r52 = r1 * r52;
    r50 = r2 < r38 ? r52 : r50;
    r39 = r1 * r39;
    r50 = r2 < r40 ? r39 : r50;
    r50 = r2 < r34 ? r1 : r50;
    r2 = r34 * r50;
    r41 = fma(r1, r41, r42 * r2);
    r41 = r34 * r41;
    r41 = r41 * r43;
    r41 = r5 <= r3 ? r23 : r41;
    r49 = fma(r4, r41, r49);
    r46 = r46 * r44;
    r41 = fma(r8, r41, r53);
    r41 = fma(r44, r45, r41);
    r18 = fma(r41, r18, r49 * r46);
    WriteSum1<double, double>((double*)inout_shared, r18);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r18 = fma(r36, r36, r48 * r48);
    r46 = fma(r22, r22, r47 * r47);
    WriteSum2<double, double>((double*)inout_shared, r18, r46);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = fma(r49, r49, r41 * r41);
    WriteSum1<double, double>((double*)inout_shared, r46);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = fma(r48, r47, r36 * r22);
    r36 = fma(r36, r49, r48 * r41);
    WriteSum2<double, double>((double*)inout_shared, r46, r36);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = fma(r22, r49, r47 * r41);
    WriteSum1<double, double>((double*)inout_shared, r49);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
}

void PinholeSplitFixedPoseFixedFocalFixedPrincipalPointResJac(
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
    double* principal_point,
    unsigned int principal_point_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
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
  PinholeSplitFixedPoseFixedFocalFixedPrincipalPointResJacKernel<<<n_blocks,
                                                                   1024>>>(
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
      principal_point,
      principal_point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      out_point_precond_diag,
      out_point_precond_diag_num_alloc,
      out_point_precond_tril,
      out_point_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar