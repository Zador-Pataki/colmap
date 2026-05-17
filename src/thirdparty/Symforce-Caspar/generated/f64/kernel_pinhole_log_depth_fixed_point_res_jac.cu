#include "kernel_pinhole_log_depth_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeLogDepthFixedPointResJacKernel(
        double* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        double* scale,
        unsigned int scale_num_alloc,
        SharedIndex* scale_indices,
        double* log_depth,
        unsigned int log_depth_num_alloc,
        double* loss,
        unsigned int loss_num_alloc,
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
        double* out_scale_jac,
        unsigned int out_scale_jac_num_alloc,
        double* const out_scale_njtr,
        unsigned int out_scale_njtr_num_alloc,
        double* const out_scale_precond_diag,
        unsigned int out_scale_precond_diag_num_alloc,
        double* const out_scale_precond_tril,
        unsigned int out_scale_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex scale_indices_loc[1024];
  scale_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? scale_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41;

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
  };
  LoadShared<1, double, double>(
      scale, 0 * scale_num_alloc, scale_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, scale_indices_loc[threadIdx.x].target, r15);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
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
    r21 = r4 * r7;
    r12 = fma(r6, r12, r11 * r21);
    r21 = r7 * r7;
    r36 = r6 * r6;
    r37 = r21 + r36;
    r38 = r5 * r5;
    r37 = fma(r17, r38, r37);
    r37 = fma(r17, r16, r37);
    r37 = fma(r3, r37, r13 * r12);
    r12 = -1.00000000000000008e-15;
    r38 = r12 + r1;
    r38 = copysign(1.0, r38);
    r38 = r14 + r38;
    r38 = r27 * r38;
    r19 = 1.0 / r19;
    r38 = r38 * r19;
    r37 = r37 * r38;
    r37 = r0 < r1 ? r37 : r0;
    r19 = r17 * r18;
    r12 = r12 + r22;
    r12 = copysign(1.0, r12);
    r12 = r14 + r12;
    r39 = r23 * r23;
    r39 = 1.0 / r39;
    r19 = r19 * r12;
    r19 = r19 * r39;
    r19 = r19 * r33;
    r33 = r9 * r18;
    r39 = r37 * r33;
    r40 = r18 * r26;
    r23 = rsqrt(r23);
    r40 = r40 * r12;
    r40 = r40 * r23;
    r23 = r37 * r40;
    r23 = r22 <= r28 ? r39 : r23;
    r12 = 1.0 / r32;
    r41 = r37 * r12;
    r41 = r41 * r33;
    r23 = r25 < r29 ? r41 : r23;
    r32 = rsqrt(r32);
    r32 = r32 * r33;
    r41 = r37 * r32;
    r23 = r25 < r31 ? r41 : r23;
    r23 = r25 < r27 ? r39 : r23;
    r20 = r20 * r27;
    r30 = copysign(1.0, r30);
    r30 = r14 + r30;
    r20 = r20 * r30;
    r20 = r20 * r24;
    r23 = fma(r23, r20, r37 * r19);
    r23 = r27 * r23;
    r34 = rsqrt(r34);
    r23 = r23 * r34;
    r23 = r22 <= r15 ? r0 : r23;
    r37 = fma(r35, r37, r18 * r23);
    r23 = r17 * r35;
    r24 = r0 < r1 ? r0 : r0;
    r30 = r17 * r18;
    r14 = r24 * r33;
    r39 = r24 * r40;
    r39 = r22 <= r28 ? r14 : r39;
    r41 = r24 * r12;
    r41 = r41 * r33;
    r39 = r25 < r29 ? r41 : r39;
    r41 = r24 * r32;
    r39 = r25 < r31 ? r41 : r39;
    r39 = r25 < r27 ? r14 : r39;
    r39 = fma(r39, r20, r24 * r19);
    r39 = r27 * r39;
    r39 = r39 * r34;
    r39 = r22 <= r15 ? r0 : r39;
    r30 = fma(r39, r30, r24 * r23);
    r37 = r37 + r30;
    r16 = fma(r5, r5, r16);
    r16 = fma(r17, r21, r16);
    r16 = fma(r17, r36, r16);
    r16 = fma(r2, r16, r13 * r8);
    r16 = r16 * r38;
    r16 = r0 < r1 ? r16 : r0;
    r8 = r16 * r33;
    r13 = r16 * r40;
    r13 = r22 <= r28 ? r8 : r13;
    r36 = r16 * r12;
    r36 = r36 * r33;
    r13 = r25 < r29 ? r36 : r13;
    r36 = r16 * r32;
    r13 = r25 < r31 ? r36 : r13;
    r13 = r25 < r27 ? r8 : r13;
    r13 = fma(r13, r20, r16 * r19);
    r13 = r27 * r13;
    r13 = r13 * r34;
    r13 = r22 <= r15 ? r0 : r13;
    r16 = fma(r35, r16, r18 * r13);
    r16 = r16 + r30;
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r37, r16);
    r13 = r4 * r6;
    r8 = r5 * r7;
    r8 = fma(r9, r8, r11 * r13);
    r8 = fma(r3, r8, r2 * r10);
    r8 = r8 * r38;
    r8 = r0 < r1 ? r8 : r0;
    r3 = r8 * r33;
    r10 = r8 * r40;
    r10 = r22 <= r28 ? r3 : r10;
    r2 = r8 * r12;
    r2 = r2 * r33;
    r10 = r25 < r29 ? r2 : r10;
    r2 = r8 * r32;
    r10 = r25 < r31 ? r2 : r10;
    r10 = r25 < r27 ? r3 : r10;
    r10 = fma(r8, r19, r10 * r20);
    r10 = r27 * r10;
    r10 = r10 * r34;
    r10 = r22 <= r15 ? r0 : r10;
    r8 = fma(r35, r8, r18 * r10);
    r8 = r8 + r30;
    r38 = r0 < r1 ? r38 : r0;
    r10 = r38 * r33;
    r3 = r38 * r40;
    r3 = r22 <= r28 ? r10 : r3;
    r2 = r38 * r12;
    r2 = r2 * r33;
    r3 = r25 < r29 ? r2 : r3;
    r2 = r38 * r32;
    r3 = r25 < r31 ? r2 : r3;
    r3 = r25 < r27 ? r10 : r3;
    r3 = fma(r3, r20, r38 * r19);
    r3 = r27 * r3;
    r3 = r3 * r34;
    r3 = r22 <= r15 ? r0 : r3;
    r3 = fma(r18, r3, r35 * r38);
    r3 = r3 + r30;
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r8, r3);
    r38 = r17 * r18;
    r38 = r38 * r35;
    r38 = r38 * r37;
    r10 = r17 * r18;
    r10 = r10 * r35;
    r10 = r10 * r16;
    WriteSum2<double, double>((double*)inout_shared, r38, r10);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r10 = r17 * r18;
    r10 = r10 * r35;
    r10 = r10 * r8;
    WriteSum2<double, double>((double*)inout_shared, r10, r0);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r10 = r17 * r18;
    r10 = r10 * r35;
    r10 = r10 * r3;
    WriteSum2<double, double>((double*)inout_shared, r0, r10);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r10 = r37 * r37;
    r38 = r16 * r16;
    WriteSum2<double, double>((double*)inout_shared, r10, r38);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r38 = r8 * r8;
    WriteSum2<double, double>((double*)inout_shared, r38, r0);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r38 = r3 * r3;
    WriteSum2<double, double>((double*)inout_shared, r0, r38);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r38 = r37 * r16;
    r10 = r37 * r8;
    WriteSum2<double, double>((double*)inout_shared, r38, r10);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = r37 * r3;
    r10 = r16 * r8;
    WriteSum2<double, double>((double*)inout_shared, r37, r10);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r16 = r16 * r3;
    WriteSum2<double, double>((double*)inout_shared, r16, r0);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r3 = r8 * r3;
    WriteSum2<double, double>((double*)inout_shared, r0, r3);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r0 < r1 ? r17 : r0;
    r3 = r1 * r33;
    r40 = r1 * r40;
    r40 = r22 <= r28 ? r3 : r40;
    r12 = r1 * r12;
    r12 = r12 * r33;
    r40 = r25 < r29 ? r12 : r40;
    r32 = r1 * r32;
    r40 = r25 < r31 ? r32 : r40;
    r40 = r25 < r27 ? r3 : r40;
    r19 = fma(r1, r19, r40 * r20);
    r19 = r27 * r19;
    r19 = r19 * r34;
    r19 = r22 <= r15 ? r0 : r19;
    r1 = fma(r35, r1, r18 * r19);
    r1 = r1 + r30;
    WriteIdx1<1024, double, double, double>(
        out_scale_jac, 0 * out_scale_jac_num_alloc, global_thread_idx, r1);
    r30 = r17 * r18;
    r30 = r30 * r35;
    r30 = r30 * r1;
    WriteSum1<double, double>((double*)inout_shared, r30);
  };
  FlushSumShared<1, double>(out_scale_njtr,
                            0 * out_scale_njtr_num_alloc,
                            scale_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r1 * r1;
    WriteSum1<double, double>((double*)inout_shared, r1);
  };
  FlushSumShared<1, double>(out_scale_precond_diag,
                            0 * out_scale_precond_diag_num_alloc,
                            scale_indices_loc,
                            (double*)inout_shared);
}

void PinholeLogDepthFixedPointResJac(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* scale,
    unsigned int scale_num_alloc,
    SharedIndex* scale_indices,
    double* log_depth,
    unsigned int log_depth_num_alloc,
    double* loss,
    unsigned int loss_num_alloc,
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
    double* out_scale_jac,
    unsigned int out_scale_jac_num_alloc,
    double* const out_scale_njtr,
    unsigned int out_scale_njtr_num_alloc,
    double* const out_scale_precond_diag,
    unsigned int out_scale_precond_diag_num_alloc,
    double* const out_scale_precond_tril,
    unsigned int out_scale_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeLogDepthFixedPointResJacKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      scale,
      scale_num_alloc,
      scale_indices,
      log_depth,
      log_depth_num_alloc,
      loss,
      loss_num_alloc,
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
      out_scale_jac,
      out_scale_jac_num_alloc,
      out_scale_njtr,
      out_scale_njtr_num_alloc,
      out_scale_precond_diag,
      out_scale_precond_diag_num_alloc,
      out_scale_precond_tril,
      out_scale_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar