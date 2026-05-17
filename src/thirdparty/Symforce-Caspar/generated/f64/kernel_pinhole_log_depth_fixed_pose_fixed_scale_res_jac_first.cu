#include "kernel_pinhole_log_depth_fixed_pose_fixed_scale_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeLogDepthFixedPoseFixedScaleResJacFirstKernel(
        double* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        double* log_depth,
        unsigned int log_depth_num_alloc,
        double* loss,
        unsigned int loss_num_alloc,
        double* pose,
        unsigned int pose_num_alloc,
        double* scale,
        unsigned int scale_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* const out_rTr,
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

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35;

  if (global_thread_idx < problem_size) {
    r0 = 0.00000000000000000e+00;
    ReadIdx1<1024, double, double, double>(
        pose, 6 * pose_num_alloc, global_thread_idx, r1);
  };
  LoadShared<2, double, double>(
      point, 0 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r2, r3);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        pose, 0 * pose_num_alloc, global_thread_idx, r4, r5);
    ReadIdx2<1024, double, double, double2>(
        pose, 2 * pose_num_alloc, global_thread_idx, r6, r7);
    r8 = r5 * r6;
    r9 = 2.00000000000000000e+00;
    r10 = r4 * r7;
    r10 = fma(r9, r10, r9 * r8);
    r3 = fma(r3, r10, r1);
    r1 = r4 * r6;
    r8 = -2.00000000000000000e+00;
    r11 = r5 * r8;
    r1 = fma(r7, r11, r9 * r1);
  };
  LoadShared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r12);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r13 = 1.00000000000000000e+00;
    r11 = fma(r5, r11, r13);
    r14 = r4 * r4;
    r11 = fma(r8, r14, r11);
    r3 = fma(r2, r1, r3);
    r3 = fma(r12, r11, r3);
    ReadIdx1<1024, double, double, double>(
        scale, 0 * scale_num_alloc, global_thread_idx, r12);
    r2 = -1.00000000000000000e+00;
    ReadIdx1<1024, double, double, double>(
        log_depth, 0 * log_depth_num_alloc, global_thread_idx, r14);
    r14 = fma(r14, r2, r12 * r2);
    r12 = 1.00000000000000008e-15;
    r8 = fmax(r3, r12);
    r15 = log(r8);
    r14 = r14 + r15;
    r14 = r0 < r3 ? r14 : r0;
    ReadIdx1<1024, double, double, double>(
        loss, 2 * loss_num_alloc, global_thread_idx, r15);
    r15 = fmax(r15, r0);
    r16 = sqrt(r15);
    r17 = r14 * r14;
    r18 = fmax(r12, r17);
    r19 = 1.0 / r18;
    ReadIdx2<1024, double, double, double2>(
        loss, 0 * loss_num_alloc, global_thread_idx, r20, r21);
    r22 = 5.00000000000000000e-01;
    r21 = fmax(r21, r12);
    r23 = r21 * r21;
    r24 = r2 * r21;
    r25 = r9 * r21;
    r26 = sqrt(r18);
    r25 = fma(r26, r25, r21 * r24);
    r25 = r17 <= r23 ? r17 : r25;
    r24 = 2.50000000000000000e+00;
    r26 = r21 * r21;
    r27 = r14 * r14;
    r28 = 1.0 / r23;
    r27 = fma(r28, r27, r13);
    r28 = log(r27);
    r26 = r26 * r28;
    r25 = r20 < r24 ? r26 : r25;
    r26 = 1.50000000000000000e+00;
    r28 = r9 * r21;
    r29 = sqrt(r27);
    r29 = r2 + r29;
    r28 = r28 * r21;
    r28 = r28 * r29;
    r25 = r20 < r26 ? r28 : r25;
    r25 = r20 < r22 ? r17 : r25;
    r28 = fmax(r0, r25);
    r28 = r15 * r28;
    r29 = r19 * r28;
    r30 = sqrt(r29);
    r30 = r17 <= r12 ? r16 : r30;
    r16 = r14 * r30;
    WriteIdx1<1024, double, double, double>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r16);
    r16 = r14 * r14;
    r31 = r30 * r30;
    r16 = r16 * r31;
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r16);
  if (global_thread_idx < problem_size) {
    r16 = r2 * r14;
    r31 = -1.00000000000000008e-15;
    r32 = r31 + r3;
    r32 = copysign(1.0, r32);
    r32 = r13 + r32;
    r32 = r22 * r32;
    r8 = 1.0 / r8;
    r32 = r32 * r8;
    r1 = r1 * r32;
    r1 = r0 < r3 ? r1 : r0;
    r8 = r2 * r14;
    r31 = r31 + r17;
    r31 = copysign(1.0, r31);
    r31 = r13 + r31;
    r33 = r18 * r18;
    r33 = 1.0 / r33;
    r8 = r8 * r31;
    r8 = r8 * r33;
    r8 = r8 * r28;
    r28 = r9 * r14;
    r33 = r1 * r28;
    r34 = r14 * r21;
    r18 = rsqrt(r18);
    r34 = r34 * r31;
    r34 = r34 * r18;
    r18 = r1 * r34;
    r18 = r17 <= r23 ? r33 : r18;
    r31 = 1.0 / r27;
    r35 = r1 * r31;
    r35 = r35 * r28;
    r18 = r20 < r24 ? r35 : r18;
    r27 = rsqrt(r27);
    r27 = r27 * r28;
    r35 = r1 * r27;
    r18 = r20 < r26 ? r35 : r18;
    r18 = r20 < r22 ? r33 : r18;
    r15 = r15 * r22;
    r25 = copysign(1.0, r25);
    r25 = r13 + r25;
    r15 = r15 * r25;
    r15 = r15 * r19;
    r18 = fma(r18, r15, r1 * r8);
    r18 = r22 * r18;
    r29 = rsqrt(r29);
    r18 = r18 * r29;
    r18 = r17 <= r12 ? r0 : r18;
    r18 = fma(r14, r18, r30 * r1);
    r1 = r2 * r30;
    r19 = r0 < r3 ? r0 : r0;
    r25 = r2 * r14;
    r13 = r19 * r28;
    r33 = r19 * r34;
    r33 = r17 <= r23 ? r13 : r33;
    r35 = r19 * r31;
    r35 = r35 * r28;
    r33 = r20 < r24 ? r35 : r33;
    r35 = r19 * r27;
    r33 = r20 < r26 ? r35 : r33;
    r33 = r20 < r22 ? r13 : r33;
    r33 = fma(r33, r15, r19 * r8);
    r33 = r22 * r33;
    r33 = r33 * r29;
    r33 = r17 <= r12 ? r0 : r33;
    r25 = fma(r33, r25, r19 * r1);
    r18 = r18 + r25;
    r16 = r16 * r30;
    r16 = r16 * r18;
    r1 = r2 * r14;
    r10 = r10 * r32;
    r10 = r0 < r3 ? r10 : r0;
    r33 = r10 * r28;
    r19 = r10 * r34;
    r19 = r17 <= r23 ? r33 : r19;
    r13 = r10 * r31;
    r13 = r13 * r28;
    r19 = r20 < r24 ? r13 : r19;
    r13 = r10 * r27;
    r19 = r20 < r26 ? r13 : r19;
    r19 = r20 < r22 ? r33 : r19;
    r19 = fma(r19, r15, r10 * r8);
    r19 = r22 * r19;
    r19 = r19 * r29;
    r19 = r17 <= r12 ? r0 : r19;
    r10 = fma(r30, r10, r14 * r19);
    r10 = r10 + r25;
    r1 = r1 * r30;
    r1 = r1 * r10;
    WriteSum2<double, double>((double*)inout_shared, r16, r1);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r2 * r14;
    r32 = r11 * r32;
    r32 = r0 < r3 ? r32 : r0;
    r3 = r32 * r28;
    r34 = r32 * r34;
    r34 = r17 <= r23 ? r3 : r34;
    r31 = r32 * r31;
    r31 = r31 * r28;
    r34 = r20 < r24 ? r31 : r34;
    r27 = r32 * r27;
    r34 = r20 < r26 ? r27 : r34;
    r34 = r20 < r22 ? r3 : r34;
    r8 = fma(r32, r8, r34 * r15);
    r8 = r22 * r8;
    r8 = r8 * r29;
    r8 = r17 <= r12 ? r0 : r8;
    r32 = fma(r30, r32, r14 * r8);
    r32 = r32 + r25;
    r1 = r1 * r30;
    r1 = r1 * r32;
    WriteSum1<double, double>((double*)inout_shared, r1);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r18 * r18;
    r25 = r10 * r10;
    WriteSum2<double, double>((double*)inout_shared, r1, r25);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = r32 * r32;
    WriteSum1<double, double>((double*)inout_shared, r25);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = r18 * r10;
    r18 = r18 * r32;
    WriteSum2<double, double>((double*)inout_shared, r25, r18);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = r10 * r32;
    WriteSum1<double, double>((double*)inout_shared, r32);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void PinholeLogDepthFixedPoseFixedScaleResJacFirst(
    double* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    double* log_depth,
    unsigned int log_depth_num_alloc,
    double* loss,
    unsigned int loss_num_alloc,
    double* pose,
    unsigned int pose_num_alloc,
    double* scale,
    unsigned int scale_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_rTr,
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
  PinholeLogDepthFixedPoseFixedScaleResJacFirstKernel<<<n_blocks, 1024>>>(
      point,
      point_num_alloc,
      point_indices,
      log_depth,
      log_depth_num_alloc,
      loss,
      loss_num_alloc,
      pose,
      pose_num_alloc,
      scale,
      scale_num_alloc,
      out_res,
      out_res_num_alloc,
      out_rTr,
      out_point_njtr,
      out_point_njtr_num_alloc,
      out_point_precond_diag,
      out_point_precond_diag_num_alloc,
      out_point_precond_tril,
      out_point_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar