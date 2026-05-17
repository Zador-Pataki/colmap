#include "kernel_pinhole_split_fixed_focal_fixed_point_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedFocalFixedPointScoreKernel(
        double* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        double* principal_point,
        unsigned int principal_point_num_alloc,
        SharedIndex* principal_point_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* weight_loss,
        unsigned int weight_loss_num_alloc,
        double* focal,
        unsigned int focal_num_alloc,
        double* point,
        unsigned int point_num_alloc,
        double* const out_rTr,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex principal_point_indices_loc[1024];
  principal_point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? principal_point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29;

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
    ReadIdx2<1024, double, double, double2>(
        focal, 0 * focal_num_alloc, global_thread_idx, r3, r7);
  };
  LoadShared<2, double, double>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r8, r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        point, 0 * point_num_alloc, global_thread_idx, r10, r11);
  };
  LoadShared<2, double, double>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r12, r13);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r14 = r12 * r13;
    r15 = 2.00000000000000000e+00;
    r14 = r14 * r15;
  };
  LoadShared<2, double, double>(
      pose, 2 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r16, r17);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
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
    r27 = r7 * r19;
    r23 = 1.00000000000000008e-15;
  };
  LoadShared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r28);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r29 = r12 * r17;
    r29 = fma(r15, r29, r20);
    r29 = fma(r11, r29, r28);
    r18 = r12 * r18;
    r28 = fma(r13, r22, r18);
    r20 = r13 * r13;
    r20 = r21 * r20;
    r26 = r20 + r26;
    r29 = fma(r10, r28, r29);
    r29 = fma(r9, r26, r29);
    r26 = copysign(1.0, r29);
    r26 = fma(r23, r26, r29);
    r26 = 1.0 / r26;
    r5 = fma(r26, r27, r5);
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
    r0 = fma(r0, r4, r1 * r5);
    r0 = r0 * r0;
    ReadIdx1<1024, double, double, double>(
        weight_loss, 6 * weight_loss_num_alloc, global_thread_idx, r1);
    r24 = 0.00000000000000000e+00;
    r1 = fmax(r1, r24);
    r26 = sqrt(r1);
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 2 * weight_loss_num_alloc, global_thread_idx, r10, r11);
    r4 = fma(r10, r4, r11 * r5);
    r4 = r4 * r4;
    r10 = r0 + r4;
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 4 * weight_loss_num_alloc, global_thread_idx, r5, r11);
    r9 = 5.00000000000000000e-01;
    r11 = fmax(r11, r23);
    r20 = r11 * r11;
    r18 = r15 * r11;
    r8 = fmax(r23, r10);
    r16 = sqrt(r8);
    r18 = fma(r16, r18, r6 * r20);
    r18 = r10 <= r20 ? r10 : r18;
    r16 = 2.50000000000000000e+00;
    r14 = 1.0 / r20;
    r14 = fma(r10, r14, r25);
    r25 = log(r14);
    r25 = r25 * r20;
    r18 = r5 < r16 ? r25 : r18;
    r25 = 1.50000000000000000e+00;
    r14 = sqrt(r14);
    r14 = r6 + r14;
    r14 = r15 * r14;
    r14 = r14 * r20;
    r18 = r5 < r25 ? r14 : r18;
    r18 = r5 < r9 ? r10 : r18;
    r18 = fmax(r24, r18);
    r18 = r1 * r18;
    r8 = 1.0 / r8;
    r18 = r18 * r8;
    r18 = sqrt(r18);
    r18 = r10 <= r23 ? r26 : r18;
    r18 = r18 * r18;
    r4 = fma(r18, r4, r18 * r0);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r4);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void PinholeSplitFixedFocalFixedPointScore(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* principal_point,
    unsigned int principal_point_num_alloc,
    SharedIndex* principal_point_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* weight_loss,
    unsigned int weight_loss_num_alloc,
    double* focal,
    unsigned int focal_num_alloc,
    double* point,
    unsigned int point_num_alloc,
    double* const out_rTr,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitFixedFocalFixedPointScoreKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      principal_point,
      principal_point_num_alloc,
      principal_point_indices,
      pixel,
      pixel_num_alloc,
      weight_loss,
      weight_loss_num_alloc,
      focal,
      focal_num_alloc,
      point,
      point_num_alloc,
      out_rTr,
      problem_size);
}

}  // namespace caspar