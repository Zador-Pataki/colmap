#include "kernel_pinhole_fixed_pose_fixed_point_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeFixedPoseFixedPointScoreKernel(double* calib,
                                          unsigned int calib_num_alloc,
                                          SharedIndex* calib_indices,
                                          double* pixel,
                                          unsigned int pixel_num_alloc,
                                          double* weight_loss,
                                          unsigned int weight_loss_num_alloc,
                                          double* pose,
                                          unsigned int pose_num_alloc,
                                          double* point,
                                          unsigned int point_num_alloc,
                                          double* const out_rTr,
                                          size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex calib_indices_loc[1024];
  calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29;

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
      calib, 0 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r2, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        pose, 4 * pose_num_alloc, global_thread_idx, r8, r9);
    ReadIdx2<1024, double, double, double2>(
        point, 0 * point_num_alloc, global_thread_idx, r10, r11);
    ReadIdx2<1024, double, double, double2>(
        pose, 2 * pose_num_alloc, global_thread_idx, r12, r13);
    r14 = -2.00000000000000000e+00;
    r15 = r13 * r14;
    ReadIdx2<1024, double, double, double2>(
        pose, 0 * pose_num_alloc, global_thread_idx, r16, r17);
    r18 = 2.00000000000000000e+00;
    r19 = r16 * r18;
    r20 = r17 * r19;
    r21 = fma(r12, r15, r20);
    r21 = fma(r11, r21, r8);
    ReadIdx1<1024, double, double, double>(
        point, 2 * point_num_alloc, global_thread_idx, r8);
    r22 = r17 * r13;
    r23 = r12 * r19;
    r22 = fma(r18, r22, r23);
    r24 = r12 * r12;
    r24 = r14 * r24;
    r25 = 1.00000000000000000e+00;
    r26 = r17 * r17;
    r26 = fma(r14, r26, r25);
    r27 = r24 + r26;
    r21 = fma(r8, r22, r21);
    r21 = fma(r10, r27, r21);
    r27 = r2 * r21;
    r22 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        pose, 6 * pose_num_alloc, global_thread_idx, r28);
    r29 = r17 * r12;
    r29 = r29 * r18;
    r19 = fma(r13, r19, r29);
    r19 = fma(r11, r19, r28);
    r23 = fma(r17, r15, r23);
    r28 = r16 * r16;
    r28 = r28 * r14;
    r26 = r28 + r26;
    r19 = fma(r10, r23, r19);
    r19 = fma(r8, r26, r19);
    r26 = copysign(1.0, r19);
    r26 = fma(r22, r26, r19);
    r26 = 1.0 / r26;
    r4 = fma(r26, r27, r4);
    r5 = fma(r5, r6, r3);
    r3 = r12 * r13;
    r3 = fma(r18, r3, r20);
    r3 = fma(r10, r3, r9);
    r15 = fma(r16, r15, r29);
    r24 = r25 + r24;
    r24 = r24 + r28;
    r3 = fma(r8, r15, r3);
    r3 = fma(r11, r24, r3);
    r24 = r7 * r3;
    r5 = fma(r26, r24, r5);
    r1 = fma(r1, r5, r0 * r4);
    r1 = r1 * r1;
    ReadIdx1<1024, double, double, double>(
        weight_loss, 6 * weight_loss_num_alloc, global_thread_idx, r0);
    r24 = 0.00000000000000000e+00;
    r0 = fmax(r0, r24);
    r26 = sqrt(r0);
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 2 * weight_loss_num_alloc, global_thread_idx, r11, r15);
    r5 = fma(r15, r5, r11 * r4);
    r5 = r5 * r5;
    r15 = r1 + r5;
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 4 * weight_loss_num_alloc, global_thread_idx, r4, r11);
    r8 = 5.00000000000000000e-01;
    r11 = fmax(r11, r22);
    r28 = r11 * r11;
    r16 = r18 * r11;
    r29 = fmax(r22, r15);
    r10 = sqrt(r29);
    r16 = fma(r6, r28, r10 * r16);
    r16 = r15 <= r28 ? r15 : r16;
    r10 = 2.50000000000000000e+00;
    r9 = 1.0 / r28;
    r9 = fma(r15, r9, r25);
    r25 = log(r9);
    r25 = r25 * r28;
    r16 = r4 < r10 ? r25 : r16;
    r25 = 1.50000000000000000e+00;
    r9 = sqrt(r9);
    r9 = r6 + r9;
    r9 = r18 * r9;
    r9 = r9 * r28;
    r16 = r4 < r25 ? r9 : r16;
    r16 = r4 < r8 ? r15 : r16;
    r16 = fmax(r24, r16);
    r16 = r0 * r16;
    r29 = 1.0 / r29;
    r16 = r16 * r29;
    r16 = sqrt(r16);
    r16 = r15 <= r22 ? r26 : r16;
    r16 = r16 * r16;
    r5 = fma(r16, r5, r16 * r1);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r5);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void PinholeFixedPoseFixedPointScore(double* calib,
                                     unsigned int calib_num_alloc,
                                     SharedIndex* calib_indices,
                                     double* pixel,
                                     unsigned int pixel_num_alloc,
                                     double* weight_loss,
                                     unsigned int weight_loss_num_alloc,
                                     double* pose,
                                     unsigned int pose_num_alloc,
                                     double* point,
                                     unsigned int point_num_alloc,
                                     double* const out_rTr,
                                     size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeFixedPoseFixedPointScoreKernel<<<n_blocks, 1024>>>(
      calib,
      calib_num_alloc,
      calib_indices,
      pixel,
      pixel_num_alloc,
      weight_loss,
      weight_loss_num_alloc,
      pose,
      pose_num_alloc,
      point,
      point_num_alloc,
      out_rTr,
      problem_size);
}

}  // namespace caspar