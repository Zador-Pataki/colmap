#include "kernel_pinhole_log_depth_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeLogDepthResJacKernel(double* pose,
                                unsigned int pose_num_alloc,
                                SharedIndex* pose_indices,
                                double* scale,
                                unsigned int scale_num_alloc,
                                SharedIndex* scale_indices,
                                double* point,
                                unsigned int point_num_alloc,
                                SharedIndex* point_indices,
                                double* log_depth,
                                unsigned int log_depth_num_alloc,
                                double* loss,
                                unsigned int loss_num_alloc,
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
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

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
  LoadShared<2, double, double>(
      point, 0 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r2, r3);
  };
  __syncthreads();
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
  };
  LoadShared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r13);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
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
        (double*)inout_shared, scale_indices_loc[threadIdx.x].target, r17);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r18 = -1.00000000000000000e+00;
    ReadIdx1<1024, double, double, double>(
        log_depth, 0 * log_depth_num_alloc, global_thread_idx, r19);
    r19 = fma(r19, r18, r17 * r18);
    r17 = 1.00000000000000008e-15;
    r20 = fmax(r1, r17);
    r21 = log(r20);
    r19 = r19 + r21;
    r19 = r0 < r1 ? r19 : r0;
    ReadIdx1<1024, double, double, double>(
        loss, 2 * loss_num_alloc, global_thread_idx, r21);
    r21 = fmax(r21, r0);
    r22 = sqrt(r21);
    r23 = r19 * r19;
    r24 = fmax(r17, r23);
    r25 = 1.0 / r24;
    ReadIdx2<1024, double, double, double2>(
        loss, 0 * loss_num_alloc, global_thread_idx, r26, r27);
    r28 = 5.00000000000000000e-01;
    r27 = fmax(r27, r17);
    r29 = r27 * r27;
    r30 = r18 * r27;
    r31 = r9 * r27;
    r32 = sqrt(r24);
    r31 = fma(r32, r31, r27 * r30);
    r31 = r23 <= r29 ? r23 : r31;
    r30 = 2.50000000000000000e+00;
    r32 = r27 * r27;
    r33 = r19 * r19;
    r34 = 1.0 / r29;
    r33 = fma(r34, r33, r14);
    r34 = log(r33);
    r32 = r32 * r34;
    r31 = r26 < r30 ? r32 : r31;
    r32 = 1.50000000000000000e+00;
    r34 = r9 * r27;
    r35 = sqrt(r33);
    r35 = r18 + r35;
    r34 = r34 * r27;
    r34 = r34 * r35;
    r31 = r26 < r32 ? r34 : r31;
    r31 = r26 < r28 ? r23 : r31;
    r34 = fmax(r0, r31);
    r34 = r21 * r34;
    r35 = r25 * r34;
    r36 = sqrt(r35);
    r36 = r23 <= r17 ? r22 : r36;
    r22 = r19 * r36;
    WriteIdx1<1024, double, double, double>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r22);
    r22 = r4 * r7;
    r12 = fma(r6, r12, r11 * r22);
    r22 = r7 * r7;
    r37 = r6 * r6;
    r38 = r22 + r37;
    r39 = r5 * r5;
    r38 = fma(r18, r39, r38);
    r38 = fma(r18, r16, r38);
    r38 = fma(r3, r38, r13 * r12);
    r12 = -1.00000000000000008e-15;
    r39 = r12 + r1;
    r39 = copysign(1.0, r39);
    r39 = r14 + r39;
    r39 = r28 * r39;
    r20 = 1.0 / r20;
    r39 = r39 * r20;
    r38 = r38 * r39;
    r38 = r0 < r1 ? r38 : r0;
    r20 = r18 * r19;
    r12 = r12 + r23;
    r12 = copysign(1.0, r12);
    r12 = r14 + r12;
    r40 = r24 * r24;
    r40 = 1.0 / r40;
    r20 = r20 * r12;
    r20 = r20 * r40;
    r20 = r20 * r34;
    r34 = r9 * r19;
    r40 = r38 * r34;
    r41 = r19 * r27;
    r24 = rsqrt(r24);
    r41 = r41 * r12;
    r41 = r41 * r24;
    r24 = r38 * r41;
    r24 = r23 <= r29 ? r40 : r24;
    r12 = 1.0 / r33;
    r42 = r38 * r12;
    r42 = r42 * r34;
    r24 = r26 < r30 ? r42 : r24;
    r33 = rsqrt(r33);
    r33 = r33 * r34;
    r42 = r38 * r33;
    r24 = r26 < r32 ? r42 : r24;
    r24 = r26 < r28 ? r40 : r24;
    r21 = r21 * r28;
    r31 = copysign(1.0, r31);
    r31 = r14 + r31;
    r21 = r21 * r31;
    r21 = r21 * r25;
    r24 = fma(r24, r21, r38 * r20);
    r24 = r28 * r24;
    r35 = rsqrt(r35);
    r24 = r24 * r35;
    r24 = r23 <= r17 ? r0 : r24;
    r38 = fma(r36, r38, r19 * r24);
    r24 = r18 * r36;
    r25 = r0 < r1 ? r0 : r0;
    r31 = r18 * r19;
    r14 = r25 * r34;
    r40 = r25 * r41;
    r40 = r23 <= r29 ? r14 : r40;
    r42 = r25 * r12;
    r42 = r42 * r34;
    r40 = r26 < r30 ? r42 : r40;
    r42 = r25 * r33;
    r40 = r26 < r32 ? r42 : r40;
    r40 = r26 < r28 ? r14 : r40;
    r40 = fma(r40, r21, r25 * r20);
    r40 = r28 * r40;
    r40 = r40 * r35;
    r40 = r23 <= r17 ? r0 : r40;
    r31 = fma(r40, r31, r25 * r24);
    r38 = r38 + r31;
    r16 = fma(r5, r5, r16);
    r16 = fma(r18, r22, r16);
    r16 = fma(r18, r37, r16);
    r16 = fma(r2, r16, r13 * r8);
    r16 = r16 * r39;
    r16 = r0 < r1 ? r16 : r0;
    r13 = r16 * r34;
    r37 = r16 * r41;
    r37 = r23 <= r29 ? r13 : r37;
    r22 = r16 * r12;
    r22 = r22 * r34;
    r37 = r26 < r30 ? r22 : r37;
    r22 = r16 * r33;
    r37 = r26 < r32 ? r22 : r37;
    r37 = r26 < r28 ? r13 : r37;
    r37 = fma(r37, r21, r16 * r20);
    r37 = r28 * r37;
    r37 = r37 * r35;
    r37 = r23 <= r17 ? r0 : r37;
    r16 = fma(r36, r16, r19 * r37);
    r16 = r16 + r31;
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r38, r16);
    r37 = r4 * r6;
    r13 = r5 * r7;
    r13 = fma(r9, r13, r11 * r37);
    r13 = fma(r3, r13, r2 * r10);
    r13 = r13 * r39;
    r13 = r0 < r1 ? r13 : r0;
    r3 = r13 * r34;
    r2 = r13 * r41;
    r2 = r23 <= r29 ? r3 : r2;
    r37 = r13 * r12;
    r37 = r37 * r34;
    r2 = r26 < r30 ? r37 : r2;
    r37 = r13 * r33;
    r2 = r26 < r32 ? r37 : r2;
    r2 = r26 < r28 ? r3 : r2;
    r2 = fma(r13, r20, r2 * r21);
    r2 = r28 * r2;
    r2 = r2 * r35;
    r2 = r23 <= r17 ? r0 : r2;
    r13 = fma(r36, r13, r19 * r2);
    r13 = r13 + r31;
    r2 = r0 < r1 ? r39 : r0;
    r3 = r2 * r34;
    r37 = r2 * r41;
    r37 = r23 <= r29 ? r3 : r37;
    r11 = r2 * r12;
    r11 = r11 * r34;
    r37 = r26 < r30 ? r11 : r37;
    r11 = r2 * r33;
    r37 = r26 < r32 ? r11 : r37;
    r37 = r26 < r28 ? r3 : r37;
    r37 = fma(r37, r21, r2 * r20);
    r37 = r28 * r37;
    r37 = r37 * r35;
    r37 = r23 <= r17 ? r0 : r37;
    r37 = fma(r19, r37, r36 * r2);
    r37 = r37 + r31;
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r13, r37);
    r2 = r18 * r19;
    r2 = r2 * r36;
    r2 = r2 * r38;
    r3 = r18 * r19;
    r3 = r3 * r36;
    r3 = r3 * r16;
    WriteSum2<double, double>((double*)inout_shared, r2, r3);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r3 = r18 * r19;
    r3 = r3 * r36;
    r3 = r3 * r13;
    WriteSum2<double, double>((double*)inout_shared, r3, r0);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r3 = r18 * r19;
    r3 = r3 * r36;
    r3 = r3 * r37;
    WriteSum2<double, double>((double*)inout_shared, r0, r3);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r3 = r38 * r38;
    r2 = r16 * r16;
    WriteSum2<double, double>((double*)inout_shared, r3, r2);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r2 = r13 * r13;
    WriteSum2<double, double>((double*)inout_shared, r2, r0);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r2 = r37 * r37;
    WriteSum2<double, double>((double*)inout_shared, r0, r2);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r2 = r38 * r16;
    r3 = r38 * r13;
    WriteSum2<double, double>((double*)inout_shared, r2, r3);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r38 = r38 * r37;
    r3 = r16 * r13;
    WriteSum2<double, double>((double*)inout_shared, r38, r3);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r16 = r16 * r37;
    WriteSum2<double, double>((double*)inout_shared, r16, r0);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = r13 * r37;
    WriteSum2<double, double>((double*)inout_shared, r0, r37);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = r0 < r1 ? r18 : r0;
    r13 = r37 * r34;
    r16 = r37 * r41;
    r16 = r23 <= r29 ? r13 : r16;
    r3 = r37 * r12;
    r3 = r3 * r34;
    r16 = r26 < r30 ? r3 : r16;
    r3 = r37 * r33;
    r16 = r26 < r32 ? r3 : r16;
    r16 = r26 < r28 ? r13 : r16;
    r16 = fma(r37, r20, r16 * r21);
    r16 = r28 * r16;
    r16 = r16 * r35;
    r16 = r23 <= r17 ? r0 : r16;
    r37 = fma(r36, r37, r19 * r16);
    r37 = r37 + r31;
    WriteIdx1<1024, double, double, double>(
        out_scale_jac, 0 * out_scale_jac_num_alloc, global_thread_idx, r37);
    r16 = r18 * r19;
    r16 = r16 * r36;
    r16 = r16 * r37;
    WriteSum1<double, double>((double*)inout_shared, r16);
  };
  FlushSumShared<1, double>(out_scale_njtr,
                            0 * out_scale_njtr_num_alloc,
                            scale_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = r37 * r37;
    WriteSum1<double, double>((double*)inout_shared, r37);
  };
  FlushSumShared<1, double>(out_scale_precond_diag,
                            0 * out_scale_precond_diag_num_alloc,
                            scale_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r8 = r8 * r39;
    r8 = r0 < r1 ? r8 : r0;
    r37 = r8 * r34;
    r16 = r8 * r41;
    r16 = r23 <= r29 ? r37 : r16;
    r13 = r8 * r12;
    r13 = r13 * r34;
    r16 = r26 < r30 ? r13 : r16;
    r13 = r8 * r33;
    r16 = r26 < r32 ? r13 : r16;
    r16 = r26 < r28 ? r37 : r16;
    r16 = fma(r16, r21, r8 * r20);
    r16 = r28 * r16;
    r16 = r16 * r35;
    r16 = r23 <= r17 ? r0 : r16;
    r16 = fma(r19, r16, r36 * r8);
    r16 = r16 + r31;
    r10 = r10 * r39;
    r10 = r0 < r1 ? r10 : r0;
    r8 = r10 * r34;
    r37 = r10 * r41;
    r37 = r23 <= r29 ? r8 : r37;
    r13 = r10 * r12;
    r13 = r13 * r34;
    r37 = r26 < r30 ? r13 : r37;
    r13 = r10 * r33;
    r37 = r26 < r32 ? r13 : r37;
    r37 = r26 < r28 ? r8 : r37;
    r37 = fma(r37, r21, r10 * r20);
    r37 = r28 * r37;
    r37 = r37 * r35;
    r37 = r23 <= r17 ? r0 : r37;
    r10 = fma(r36, r10, r19 * r37);
    r10 = r10 + r31;
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             0 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r16,
                                             r10);
    r39 = r15 * r39;
    r39 = r0 < r1 ? r39 : r0;
    r1 = r39 * r34;
    r41 = r39 * r41;
    r41 = r23 <= r29 ? r1 : r41;
    r12 = r39 * r12;
    r12 = r12 * r34;
    r41 = r26 < r30 ? r12 : r41;
    r33 = r39 * r33;
    r41 = r26 < r32 ? r33 : r41;
    r41 = r26 < r28 ? r1 : r41;
    r20 = fma(r39, r20, r41 * r21);
    r20 = r28 * r20;
    r20 = r20 * r35;
    r20 = r23 <= r17 ? r0 : r20;
    r39 = fma(r36, r39, r19 * r20);
    r39 = r39 + r31;
    WriteIdx1<1024, double, double, double>(
        out_point_jac, 2 * out_point_jac_num_alloc, global_thread_idx, r39);
    r31 = r18 * r19;
    r31 = r31 * r36;
    r31 = r31 * r16;
    r20 = r18 * r19;
    r20 = r20 * r36;
    r20 = r20 * r10;
    WriteSum2<double, double>((double*)inout_shared, r31, r20);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r20 = r18 * r19;
    r20 = r20 * r36;
    r20 = r20 * r39;
    WriteSum1<double, double>((double*)inout_shared, r20);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r20 = r16 * r16;
    r31 = r10 * r10;
    WriteSum2<double, double>((double*)inout_shared, r20, r31);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = r39 * r39;
    WriteSum1<double, double>((double*)inout_shared, r31);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = r16 * r10;
    r16 = r16 * r39;
    WriteSum2<double, double>((double*)inout_shared, r31, r16);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = r10 * r39;
    WriteSum1<double, double>((double*)inout_shared, r39);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
}

void PinholeLogDepthResJac(double* pose,
                           unsigned int pose_num_alloc,
                           SharedIndex* pose_indices,
                           double* scale,
                           unsigned int scale_num_alloc,
                           SharedIndex* scale_indices,
                           double* point,
                           unsigned int point_num_alloc,
                           SharedIndex* point_indices,
                           double* log_depth,
                           unsigned int log_depth_num_alloc,
                           double* loss,
                           unsigned int loss_num_alloc,
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
  PinholeLogDepthResJacKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      scale,
      scale_num_alloc,
      scale_indices,
      point,
      point_num_alloc,
      point_indices,
      log_depth,
      log_depth_num_alloc,
      loss,
      loss_num_alloc,
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