#include "kernel_pinhole_log_depth_fixed_pose_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeLogDepthFixedPoseFixedPointResJacKernel(
        double* scale,
        unsigned int scale_num_alloc,
        SharedIndex* scale_indices,
        double* log_depth,
        unsigned int log_depth_num_alloc,
        double* loss,
        unsigned int loss_num_alloc,
        double* pose,
        unsigned int pose_num_alloc,
        double* point,
        unsigned int point_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* const out_scale_njtr,
        unsigned int out_scale_njtr_num_alloc,
        double* const out_scale_precond_diag,
        unsigned int out_scale_precond_diag_num_alloc,
        double* const out_scale_precond_tril,
        unsigned int out_scale_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ SharedIndex scale_indices_loc[1024];
  scale_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? scale_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34;

  if (global_thread_idx < problem_size) {
    r0 = 0.00000000000000000e+00;
    ReadIdx1<1024, double, double, double>(
        pose, 6 * pose_num_alloc, global_thread_idx, r1);
    ReadIdx2<1024, double, double, double2>(
        point, 0 * point_num_alloc, global_thread_idx, r2, r3);
    ReadIdx2<1024, double, double, double2>(
        pose, 0 * pose_num_alloc, global_thread_idx, r4, r5);
    ReadIdx2<1024, double, double, double2>(
        pose, 2 * pose_num_alloc, global_thread_idx, r6, r7);
    r8 = r5 * r6;
    r9 = 2.00000000000000000e+00;
    r10 = r4 * r7;
    r10 = fma(r9, r10, r9 * r8);
    r10 = fma(r3, r10, r1);
    r3 = r4 * r6;
    r1 = -2.00000000000000000e+00;
    r8 = r5 * r1;
    r3 = fma(r7, r8, r9 * r3);
    ReadIdx1<1024, double, double, double>(
        point, 2 * point_num_alloc, global_thread_idx, r11);
    r12 = 1.00000000000000000e+00;
    r8 = fma(r5, r8, r12);
    r13 = r4 * r4;
    r8 = fma(r1, r13, r8);
    r10 = fma(r2, r3, r10);
    r10 = fma(r11, r8, r10);
  };
  LoadShared<1, double, double>(
      scale, 0 * scale_num_alloc, scale_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, scale_indices_loc[threadIdx.x].target, r8);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r11 = -1.00000000000000000e+00;
    ReadIdx1<1024, double, double, double>(
        log_depth, 0 * log_depth_num_alloc, global_thread_idx, r3);
    r3 = fma(r3, r11, r8 * r11);
    r8 = 1.00000000000000008e-15;
    r2 = fmax(r10, r8);
    r2 = log(r2);
    r3 = r3 + r2;
    r3 = r0 < r10 ? r3 : r0;
    ReadIdx1<1024, double, double, double>(
        loss, 2 * loss_num_alloc, global_thread_idx, r2);
    r2 = fmax(r2, r0);
    r13 = sqrt(r2);
    r1 = r3 * r3;
    r14 = fmax(r8, r1);
    r15 = 1.0 / r14;
    ReadIdx2<1024, double, double, double2>(
        loss, 0 * loss_num_alloc, global_thread_idx, r16, r17);
    r18 = 5.00000000000000000e-01;
    r17 = fmax(r17, r8);
    r19 = r17 * r17;
    r20 = r9 * r17;
    r21 = sqrt(r14);
    r20 = fma(r21, r20, r11 * r19);
    r20 = r1 <= r19 ? r1 : r20;
    r21 = 2.50000000000000000e+00;
    r22 = r3 * r3;
    r23 = 1.0 / r19;
    r22 = fma(r23, r22, r12);
    r23 = log(r22);
    r23 = r23 * r19;
    r20 = r16 < r21 ? r23 : r20;
    r23 = 1.50000000000000000e+00;
    r24 = sqrt(r22);
    r24 = r11 + r24;
    r24 = r9 * r24;
    r24 = r24 * r19;
    r20 = r16 < r23 ? r24 : r20;
    r20 = r16 < r18 ? r1 : r20;
    r24 = fmax(r0, r20);
    r24 = r2 * r24;
    r25 = r15 * r24;
    r26 = sqrt(r25);
    r26 = r1 <= r8 ? r13 : r26;
    r13 = r3 * r26;
    WriteIdx1<1024, double, double, double>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r13);
    r13 = r11 * r3;
    r27 = r0 < r10 ? r11 : r0;
    r28 = r9 * r3;
    r29 = r27 * r28;
    r30 = r3 * r17;
    r31 = -1.00000000000000008e-15;
    r31 = r31 + r1;
    r31 = copysign(1.0, r31);
    r31 = r12 + r31;
    r32 = rsqrt(r14);
    r30 = r30 * r27;
    r30 = r30 * r31;
    r30 = r30 * r32;
    r30 = r1 <= r19 ? r29 : r30;
    r33 = 1.0 / r22;
    r34 = r27 * r33;
    r34 = r34 * r28;
    r30 = r16 < r21 ? r34 : r30;
    r22 = rsqrt(r22);
    r34 = r27 * r22;
    r34 = r34 * r28;
    r30 = r16 < r23 ? r34 : r30;
    r30 = r16 < r18 ? r29 : r30;
    r2 = r2 * r18;
    r20 = copysign(1.0, r20);
    r20 = r12 + r20;
    r2 = r2 * r20;
    r2 = r2 * r15;
    r15 = r11 * r3;
    r14 = r14 * r14;
    r14 = 1.0 / r14;
    r15 = r15 * r31;
    r15 = r15 * r14;
    r15 = r15 * r24;
    r30 = fma(r27, r15, r30 * r2);
    r30 = r18 * r30;
    r25 = rsqrt(r25);
    r30 = r30 * r25;
    r30 = r1 <= r8 ? r0 : r30;
    r27 = fma(r26, r27, r3 * r30);
    r30 = r11 * r26;
    r10 = r0 < r10 ? r0 : r0;
    r27 = fma(r10, r30, r27);
    r24 = r11 * r3;
    r14 = r9 * r3;
    r14 = r14 * r10;
    r20 = r3 * r17;
    r20 = r20 * r31;
    r20 = r20 * r10;
    r20 = r20 * r32;
    r20 = r1 <= r19 ? r14 : r20;
    r33 = r33 * r14;
    r20 = r16 < r21 ? r33 : r20;
    r22 = r22 * r14;
    r20 = r16 < r23 ? r22 : r20;
    r20 = r16 < r18 ? r14 : r20;
    r2 = fma(r20, r2, r10 * r15);
    r2 = r18 * r2;
    r2 = r2 * r25;
    r2 = r1 <= r8 ? r0 : r2;
    r27 = fma(r2, r24, r27);
    r13 = r13 * r26;
    r13 = r13 * r27;
    WriteSum1<double, double>((double*)inout_shared, r13);
  };
  FlushSumShared<1, double>(out_scale_njtr,
                            0 * out_scale_njtr_num_alloc,
                            scale_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = r27 * r27;
    WriteSum1<double, double>((double*)inout_shared, r27);
  };
  FlushSumShared<1, double>(out_scale_precond_diag,
                            0 * out_scale_precond_diag_num_alloc,
                            scale_indices_loc,
                            (double*)inout_shared);
}

void PinholeLogDepthFixedPoseFixedPointResJac(
    double* scale,
    unsigned int scale_num_alloc,
    SharedIndex* scale_indices,
    double* log_depth,
    unsigned int log_depth_num_alloc,
    double* loss,
    unsigned int loss_num_alloc,
    double* pose,
    unsigned int pose_num_alloc,
    double* point,
    unsigned int point_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
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
  PinholeLogDepthFixedPoseFixedPointResJacKernel<<<n_blocks, 1024>>>(
      scale,
      scale_num_alloc,
      scale_indices,
      log_depth,
      log_depth_num_alloc,
      loss,
      loss_num_alloc,
      pose,
      pose_num_alloc,
      point,
      point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_scale_njtr,
      out_scale_njtr_num_alloc,
      out_scale_precond_diag,
      out_scale_precond_diag_num_alloc,
      out_scale_precond_tril,
      out_scale_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar