#include "kernel_pinhole_log_depth_fixed_scale_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeLogDepthFixedScaleFixedPointResJacFirstKernel(
        double* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        double* log_depth,
        unsigned int log_depth_num_alloc,
        double* loss,
        unsigned int loss_num_alloc,
        double* scale,
        unsigned int scale_num_alloc,
        double* point,
        unsigned int point_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* const out_rTr,
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

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42;

  if (global_thread_idx < problem_size) {
    r0 = 0.00000000000000000e+00;
  };
  LoadShared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        point, 0 * point_num_alloc, global_thread_idx, r2, r3);
  };
  LoadShared<2, double, double>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r4, r5);
  };
  __syncthreads();
  LoadShared<2, double, double>(
      pose, 2 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r6, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r8 = r5 * r6;
    r9 = 2.00000000000000000e+00;
    r10 = r4 * r7;
    r10 = fma(r9, r10, r9 * r8);
    r1 = fma(r3, r10, r1);
    r8 = r4 * r6;
    r11 = -2.00000000000000000e+00;
    r12 = r5 * r11;
    r8 = fma(r7, r12, r9 * r8);
    ReadIdx1<1024, double, double, double>(
        point, 2 * point_num_alloc, global_thread_idx, r13);
    r14 = 1.00000000000000000e+00;
    r15 = fma(r5, r12, r14);
    r16 = r4 * r4;
    r15 = fma(r11, r16, r15);
    r1 = fma(r2, r8, r1);
    r1 = fma(r13, r15, r1);
    ReadIdx1<1024, double, double, double>(
        scale, 0 * scale_num_alloc, global_thread_idx, r15);
    r17 = -1.00000000000000000e+00;
    ReadIdx1<1024, double, double, double>(
        log_depth, 0 * log_depth_num_alloc, global_thread_idx, r18);
    r18 = fma(r18, r17, r15 * r17);
    r15 = 1.00000000000000008e-15;
    r19 = fmax(r1, r15);
    r20 = log(r19);
    r18 = r18 + r20;
    r18 = r0 < r1 ? r18 : r0;
    ReadIdx1<1024, double, double, double>(
        loss, 2 * loss_num_alloc, global_thread_idx, r20);
    r20 = fmax(r20, r0);
    r21 = sqrt(r20);
    r22 = r18 * r18;
    r23 = fmax(r15, r22);
    r24 = 1.0 / r23;
    ReadIdx2<1024, double, double, double2>(
        loss, 0 * loss_num_alloc, global_thread_idx, r25, r26);
    r27 = 5.00000000000000000e-01;
    r26 = fmax(r26, r15);
    r28 = r26 * r26;
    r29 = r17 * r26;
    r30 = r9 * r26;
    r31 = sqrt(r23);
    r30 = fma(r31, r30, r26 * r29);
    r30 = r22 <= r28 ? r22 : r30;
    r29 = 2.50000000000000000e+00;
    r31 = r26 * r26;
    r32 = r18 * r18;
    r33 = 1.0 / r28;
    r32 = fma(r33, r32, r14);
    r33 = log(r32);
    r31 = r31 * r33;
    r30 = r25 < r29 ? r31 : r30;
    r31 = 1.50000000000000000e+00;
    r33 = r9 * r26;
    r34 = sqrt(r32);
    r34 = r17 + r34;
    r33 = r33 * r26;
    r33 = r33 * r34;
    r30 = r25 < r31 ? r33 : r30;
    r30 = r25 < r27 ? r22 : r30;
    r33 = fmax(r0, r30);
    r33 = r20 * r33;
    r34 = r24 * r33;
    r35 = sqrt(r34);
    r35 = r22 <= r15 ? r21 : r35;
    r21 = r18 * r35;
    WriteIdx1<1024, double, double, double>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r21);
    r21 = r18 * r18;
    r36 = r35 * r35;
    r21 = r21 * r36;
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r21);
  if (global_thread_idx < problem_size) {
    r21 = r17 * r18;
    r36 = r4 * r7;
    r12 = fma(r6, r12, r11 * r36);
    r36 = r7 * r7;
    r37 = r6 * r6;
    r38 = r36 + r37;
    r39 = r5 * r5;
    r38 = fma(r17, r39, r38);
    r38 = fma(r17, r16, r38);
    r38 = fma(r3, r38, r13 * r12);
    r12 = -1.00000000000000008e-15;
    r39 = r12 + r1;
    r39 = copysign(1.0, r39);
    r39 = r14 + r39;
    r39 = r27 * r39;
    r19 = 1.0 / r19;
    r39 = r39 * r19;
    r38 = r38 * r39;
    r38 = r0 < r1 ? r38 : r0;
    r19 = r17 * r18;
    r12 = r12 + r22;
    r12 = copysign(1.0, r12);
    r12 = r14 + r12;
    r40 = r23 * r23;
    r40 = 1.0 / r40;
    r19 = r19 * r12;
    r19 = r19 * r40;
    r19 = r19 * r33;
    r33 = r9 * r18;
    r40 = r38 * r33;
    r41 = r18 * r26;
    r23 = rsqrt(r23);
    r41 = r41 * r12;
    r41 = r41 * r23;
    r23 = r38 * r41;
    r23 = r22 <= r28 ? r40 : r23;
    r12 = 1.0 / r32;
    r42 = r38 * r12;
    r42 = r42 * r33;
    r23 = r25 < r29 ? r42 : r23;
    r32 = rsqrt(r32);
    r32 = r32 * r33;
    r42 = r38 * r32;
    r23 = r25 < r31 ? r42 : r23;
    r23 = r25 < r27 ? r40 : r23;
    r20 = r20 * r27;
    r30 = copysign(1.0, r30);
    r30 = r14 + r30;
    r20 = r20 * r30;
    r20 = r20 * r24;
    r23 = fma(r23, r20, r38 * r19);
    r23 = r27 * r23;
    r34 = rsqrt(r34);
    r23 = r23 * r34;
    r23 = r22 <= r15 ? r0 : r23;
    r38 = fma(r35, r38, r18 * r23);
    r23 = r17 * r35;
    r24 = r0 < r1 ? r0 : r0;
    r30 = r17 * r18;
    r14 = r24 * r33;
    r40 = r24 * r41;
    r40 = r22 <= r28 ? r14 : r40;
    r42 = r24 * r12;
    r42 = r42 * r33;
    r40 = r25 < r29 ? r42 : r40;
    r42 = r24 * r32;
    r40 = r25 < r31 ? r42 : r40;
    r40 = r25 < r27 ? r14 : r40;
    r40 = fma(r40, r20, r24 * r19);
    r40 = r27 * r40;
    r40 = r40 * r34;
    r40 = r22 <= r15 ? r0 : r40;
    r30 = fma(r40, r30, r24 * r23);
    r38 = r38 + r30;
    r21 = r21 * r35;
    r21 = r21 * r38;
    r23 = r17 * r18;
    r16 = fma(r5, r5, r16);
    r16 = fma(r17, r36, r16);
    r16 = fma(r17, r37, r16);
    r16 = fma(r2, r16, r13 * r8);
    r16 = r16 * r39;
    r16 = r0 < r1 ? r16 : r0;
    r8 = r16 * r33;
    r13 = r16 * r41;
    r13 = r22 <= r28 ? r8 : r13;
    r37 = r16 * r12;
    r37 = r37 * r33;
    r13 = r25 < r29 ? r37 : r13;
    r37 = r16 * r32;
    r13 = r25 < r31 ? r37 : r13;
    r13 = r25 < r27 ? r8 : r13;
    r13 = fma(r13, r20, r16 * r19);
    r13 = r27 * r13;
    r13 = r13 * r34;
    r13 = r22 <= r15 ? r0 : r13;
    r16 = fma(r35, r16, r18 * r13);
    r16 = r16 + r30;
    r23 = r23 * r35;
    r23 = r23 * r16;
    WriteSum2<double, double>((double*)inout_shared, r21, r23);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = r17 * r18;
    r21 = r4 * r6;
    r13 = r5 * r7;
    r13 = fma(r9, r13, r11 * r21);
    r13 = fma(r3, r13, r2 * r10);
    r13 = r13 * r39;
    r13 = r0 < r1 ? r13 : r0;
    r3 = r13 * r33;
    r10 = r13 * r41;
    r10 = r22 <= r28 ? r3 : r10;
    r2 = r13 * r12;
    r2 = r2 * r33;
    r10 = r25 < r29 ? r2 : r10;
    r2 = r13 * r32;
    r10 = r25 < r31 ? r2 : r10;
    r10 = r25 < r27 ? r3 : r10;
    r10 = fma(r13, r19, r10 * r20);
    r10 = r27 * r10;
    r10 = r10 * r34;
    r10 = r22 <= r15 ? r0 : r10;
    r13 = fma(r35, r13, r18 * r10);
    r13 = r13 + r30;
    r23 = r23 * r35;
    r23 = r23 * r13;
    WriteSum2<double, double>((double*)inout_shared, r23, r0);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = r17 * r18;
    r39 = r0 < r1 ? r39 : r0;
    r1 = r39 * r33;
    r41 = r39 * r41;
    r41 = r22 <= r28 ? r1 : r41;
    r12 = r39 * r12;
    r12 = r12 * r33;
    r41 = r25 < r29 ? r12 : r41;
    r32 = r39 * r32;
    r41 = r25 < r31 ? r32 : r41;
    r41 = r25 < r27 ? r1 : r41;
    r20 = fma(r41, r20, r39 * r19);
    r20 = r27 * r20;
    r20 = r20 * r34;
    r20 = r22 <= r15 ? r0 : r20;
    r20 = fma(r18, r20, r35 * r39);
    r20 = r20 + r30;
    r23 = r23 * r35;
    r23 = r23 * r20;
    WriteSum2<double, double>((double*)inout_shared, r0, r23);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = r38 * r38;
    r30 = r16 * r16;
    WriteSum2<double, double>((double*)inout_shared, r23, r30);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = r13 * r13;
    WriteSum2<double, double>((double*)inout_shared, r30, r0);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = r20 * r20;
    WriteSum2<double, double>((double*)inout_shared, r0, r30);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = r38 * r16;
    r23 = r38 * r13;
    WriteSum2<double, double>((double*)inout_shared, r30, r23);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r38 = r38 * r20;
    r23 = r16 * r13;
    WriteSum2<double, double>((double*)inout_shared, r38, r23);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r16 = r16 * r20;
    WriteSum2<double, double>((double*)inout_shared, r16, r0);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r20 = r13 * r20;
    WriteSum2<double, double>((double*)inout_shared, r0, r20);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void PinholeLogDepthFixedScaleFixedPointResJacFirst(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* log_depth,
    unsigned int log_depth_num_alloc,
    double* loss,
    unsigned int loss_num_alloc,
    double* scale,
    unsigned int scale_num_alloc,
    double* point,
    unsigned int point_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_rTr,
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
  PinholeLogDepthFixedScaleFixedPointResJacFirstKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      log_depth,
      log_depth_num_alloc,
      loss,
      loss_num_alloc,
      scale,
      scale_num_alloc,
      point,
      point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_rTr,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_pose_precond_diag,
      out_pose_precond_diag_num_alloc,
      out_pose_precond_tril,
      out_pose_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar