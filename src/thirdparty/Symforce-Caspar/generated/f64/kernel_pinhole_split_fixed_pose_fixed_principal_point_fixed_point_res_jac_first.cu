#include "kernel_pinhole_split_fixed_pose_fixed_principal_point_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedPoseFixedPrincipalPointFixedPointResJacFirstKernel(
        double* focal,
        unsigned int focal_num_alloc,
        SharedIndex* focal_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* weight_loss,
        unsigned int weight_loss_num_alloc,
        double* pose,
        unsigned int pose_num_alloc,
        double* principal_point,
        unsigned int principal_point_num_alloc,
        double* point,
        unsigned int point_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* const out_rTr,
        double* const out_focal_njtr,
        unsigned int out_focal_njtr_num_alloc,
        double* const out_focal_precond_diag,
        unsigned int out_focal_precond_diag_num_alloc,
        double* const out_focal_precond_tril,
        unsigned int out_focal_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex focal_indices_loc[1024];
  focal_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39;

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
    ReadIdx2<1024, double, double, double2>(
        point, 0 * point_num_alloc, global_thread_idx, r10, r11);
    ReadIdx2<1024, double, double, double2>(
        pose, 0 * pose_num_alloc, global_thread_idx, r12, r13);
    r14 = r12 * r13;
    r15 = 2.00000000000000000e+00;
    r14 = r14 * r15;
    ReadIdx2<1024, double, double, double2>(
        pose, 2 * pose_num_alloc, global_thread_idx, r16, r17);
    r18 = r16 * r15;
    r19 = fma(r17, r18, r14);
    r19 = fma(r10, r19, r9);
    ReadIdx1<1024, double, double, double>(
        point, 2 * point_num_alloc, global_thread_idx, r9);
    r20 = r13 * r18;
    r21 = -2.00000000000000000e+00;
    r22 = r17 * r21;
    r23 = fma(r12, r22, r20);
    r24 = r16 * r16;
    r24 = r24 * r21;
    r25 = 1.00000000000000000e+00;
    r26 = r12 * r12;
    r26 = fma(r21, r26, r25);
    r27 = r24 + r26;
    r19 = fma(r9, r23, r19);
    r19 = fma(r11, r27, r19);
    r27 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        pose, 6 * pose_num_alloc, global_thread_idx, r23);
    r28 = r12 * r17;
    r28 = fma(r15, r28, r20);
    r28 = fma(r11, r28, r23);
    r18 = r12 * r18;
    r23 = fma(r13, r22, r18);
    r20 = r13 * r13;
    r20 = r21 * r20;
    r26 = r20 + r26;
    r28 = fma(r10, r23, r28);
    r28 = fma(r9, r26, r28);
    r26 = copysign(1.0, r28);
    r26 = fma(r27, r26, r28);
    r26 = 1.0 / r26;
    r19 = r19 * r26;
    r5 = fma(r7, r19, r5);
    r4 = fma(r4, r6, r2);
    r22 = fma(r16, r22, r14);
    r22 = fma(r11, r22, r8);
    r11 = r13 * r17;
    r11 = fma(r15, r11, r18);
    r24 = r25 + r24;
    r24 = r24 + r20;
    r22 = fma(r9, r11, r22);
    r22 = fma(r10, r24, r22);
    r24 = r3 * r22;
    r4 = fma(r26, r24, r4);
    r24 = fma(r0, r4, r1 * r5);
    ReadIdx1<1024, double, double, double>(
        weight_loss, 6 * weight_loss_num_alloc, global_thread_idx, r10);
    r11 = 0.00000000000000000e+00;
    r10 = fmax(r10, r11);
    r9 = sqrt(r10);
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 2 * weight_loss_num_alloc, global_thread_idx, r20, r18);
    r4 = fma(r20, r4, r18 * r5);
    r5 = fma(r4, r4, r24 * r24);
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 4 * weight_loss_num_alloc, global_thread_idx, r8, r16);
    r14 = 5.00000000000000000e-01;
    r16 = fmax(r16, r27);
    r2 = r16 * r16;
    r7 = r15 * r16;
    r28 = fmax(r27, r5);
    r23 = sqrt(r28);
    r7 = fma(r23, r7, r6 * r2);
    r7 = r5 <= r2 ? r5 : r7;
    r23 = 2.50000000000000000e+00;
    r21 = 1.0 / r2;
    r21 = fma(r5, r21, r25);
    r29 = log(r21);
    r29 = r29 * r2;
    r7 = r8 < r23 ? r29 : r7;
    r29 = 1.50000000000000000e+00;
    r30 = sqrt(r21);
    r30 = r6 + r30;
    r30 = r15 * r30;
    r30 = r30 * r2;
    r7 = r8 < r29 ? r30 : r7;
    r7 = r8 < r14 ? r5 : r7;
    r30 = fmax(r11, r7);
    r31 = 1.0 / r28;
    r31 = r10 * r31;
    r32 = r30 * r31;
    r33 = sqrt(r32);
    r33 = r5 <= r27 ? r9 : r33;
    r9 = r24 * r33;
    r34 = r4 * r33;
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r9, r34);
    r34 = r24 * r33;
    r35 = r4 * r4;
    r35 = r35 * r33;
    r35 = fma(r33, r35, r9 * r34);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r35);
  if (global_thread_idx < problem_size) {
    r35 = r6 * r4;
    r34 = r33 * r35;
    r30 = r10 * r30;
    r10 = -5.00000000000000000e-01;
    r36 = -1.00000000000000008e-15;
    r36 = r36 + r5;
    r36 = copysign(1.0, r36);
    r36 = r25 + r36;
    r37 = r28 * r28;
    r37 = 1.0 / r37;
    r30 = r30 * r10;
    r30 = r30 * r36;
    r30 = r30 * r37;
    r0 = r0 * r22;
    r0 = r0 * r26;
    r37 = r15 * r24;
    r10 = r20 * r15;
    r10 = r10 * r22;
    r10 = r10 * r4;
    r10 = fma(r26, r10, r0 * r37);
    r38 = r14 * r16;
    r28 = rsqrt(r28);
    r38 = r38 * r36;
    r38 = r38 * r28;
    r28 = r10 * r38;
    r28 = r5 <= r2 ? r10 : r28;
    r36 = 1.0 / r21;
    r39 = r10 * r36;
    r28 = r8 < r23 ? r39 : r28;
    r21 = rsqrt(r21);
    r39 = r10 * r21;
    r28 = r8 < r29 ? r39 : r28;
    r28 = r8 < r14 ? r10 : r28;
    r39 = r14 * r28;
    r7 = copysign(1.0, r7);
    r7 = r25 + r7;
    r31 = r7 * r31;
    r39 = fma(r31, r39, r10 * r30);
    r39 = r14 * r39;
    r32 = rsqrt(r32);
    r39 = r39 * r32;
    r39 = r5 <= r27 ? r11 : r39;
    r10 = 2.50000000000000000e-01;
    r7 = r5 <= r2 ? r11 : r11;
    r7 = r8 < r23 ? r11 : r7;
    r7 = r8 < r29 ? r11 : r7;
    r7 = r8 < r14 ? r11 : r7;
    r7 = r10 * r7;
    r7 = r7 * r32;
    r7 = r7 * r31;
    r7 = r5 <= r27 ? r11 : r7;
    r35 = r7 * r35;
    r10 = fma(r4, r39, r35);
    r25 = r20 * r22;
    r25 = r25 * r33;
    r10 = fma(r26, r25, r10);
    r25 = r6 * r24;
    r25 = r25 * r7;
    r39 = fma(r24, r39, r25);
    r39 = fma(r33, r0, r39);
    r0 = r6 * r39;
    r0 = fma(r9, r0, r10 * r34);
    r1 = r1 * r19;
    r7 = r18 * r15;
    r7 = r7 * r4;
    r7 = fma(r19, r7, r1 * r37);
    r38 = r7 * r38;
    r38 = r5 <= r2 ? r7 : r38;
    r36 = r7 * r36;
    r38 = r8 < r23 ? r36 : r38;
    r21 = r7 * r21;
    r38 = r8 < r29 ? r21 : r38;
    r38 = r8 < r14 ? r7 : r38;
    r8 = r14 * r38;
    r8 = fma(r31, r8, r7 * r30);
    r8 = r14 * r8;
    r8 = r8 * r32;
    r8 = r5 <= r27 ? r11 : r8;
    r35 = fma(r4, r8, r35);
    r11 = r18 * r33;
    r35 = fma(r19, r11, r35);
    r8 = fma(r24, r8, r25);
    r8 = fma(r33, r1, r8);
    r1 = r6 * r8;
    r1 = fma(r9, r1, r35 * r34);
    WriteSum2<double, double>((double*)inout_shared, r0, r1);
  };
  FlushSumShared<2, double>(out_focal_njtr,
                            0 * out_focal_njtr_num_alloc,
                            focal_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = fma(r39, r39, r10 * r10);
    r0 = fma(r35, r35, r8 * r8);
    WriteSum2<double, double>((double*)inout_shared, r1, r0);
  };
  FlushSumShared<2, double>(out_focal_precond_diag,
                            0 * out_focal_precond_diag_num_alloc,
                            focal_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = fma(r10, r35, r39 * r8);
    WriteSum1<double, double>((double*)inout_shared, r35);
  };
  FlushSumShared<1, double>(out_focal_precond_tril,
                            0 * out_focal_precond_tril_num_alloc,
                            focal_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void PinholeSplitFixedPoseFixedPrincipalPointFixedPointResJacFirst(
    double* focal,
    unsigned int focal_num_alloc,
    SharedIndex* focal_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* weight_loss,
    unsigned int weight_loss_num_alloc,
    double* pose,
    unsigned int pose_num_alloc,
    double* principal_point,
    unsigned int principal_point_num_alloc,
    double* point,
    unsigned int point_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_rTr,
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
  PinholeSplitFixedPoseFixedPrincipalPointFixedPointResJacFirstKernel<<<
      n_blocks,
      1024>>>(focal,
              focal_num_alloc,
              focal_indices,
              pixel,
              pixel_num_alloc,
              weight_loss,
              weight_loss_num_alloc,
              pose,
              pose_num_alloc,
              principal_point,
              principal_point_num_alloc,
              point,
              point_num_alloc,
              out_res,
              out_res_num_alloc,
              out_rTr,
              out_focal_njtr,
              out_focal_njtr_num_alloc,
              out_focal_precond_diag,
              out_focal_precond_diag_num_alloc,
              out_focal_precond_tril,
              out_focal_precond_tril_num_alloc,
              problem_size);
}

}  // namespace caspar