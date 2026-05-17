#include "kernel_pinhole_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeScoreKernel(float* pose,
                       unsigned int pose_num_alloc,
                       SharedIndex* pose_indices,
                       float* calib,
                       unsigned int calib_num_alloc,
                       SharedIndex* calib_indices,
                       float* point,
                       unsigned int point_num_alloc,
                       SharedIndex* point_indices,
                       float* pixel,
                       unsigned int pixel_num_alloc,
                       float* weight_loss,
                       unsigned int weight_loss_num_alloc,
                       float* const out_rTr,
                       size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex calib_indices_loc[1024];
  calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(weight_loss,
                                         0 * weight_loss_num_alloc,
                                         global_thread_idx,
                                         r0,
                                         r1,
                                         r2,
                                         r3);
  };
  LoadShared<4, float, float>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       calib_indices_loc[threadIdx.x].target,
                       r4,
                       r5,
                       r6,
                       r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r8, r9);
    r10 = -1.00000000000000000e+00;
    r8 = fmaf(r8, r10, r6);
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r6,
                       r11,
                       r12);
  };
  __syncthreads();
  LoadShared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_indices_loc[threadIdx.x].target,
                       r13,
                       r14,
                       r15);
  };
  __syncthreads();
  LoadShared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r16,
                       r17,
                       r18,
                       r19);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r20 = -2.00000000000000000e+00;
    r21 = r19 * r20;
    r22 = 2.00000000000000000e+00;
    r23 = r16 * r22;
    r24 = r17 * r23;
    r25 = fmaf(r18, r21, r24);
    r25 = fmaf(r14, r25, r6);
    r6 = r17 * r19;
    r26 = r18 * r23;
    r6 = fmaf(r22, r6, r26);
    r27 = r18 * r18;
    r27 = r20 * r27;
    r28 = 1.00000000000000000e+00;
    r29 = r17 * r17;
    r29 = fmaf(r20, r29, r28);
    r30 = r27 + r29;
    r25 = fmaf(r15, r6, r25);
    r25 = fmaf(r13, r30, r25);
    r30 = r4 * r25;
    r6 = 9.99999999999999955e-07;
    r31 = r17 * r18;
    r31 = r31 * r22;
    r23 = fmaf(r19, r23, r31);
    r23 = fmaf(r14, r23, r12);
    r26 = fmaf(r17, r21, r26);
    r12 = r16 * r16;
    r12 = r12 * r20;
    r29 = r12 + r29;
    r23 = fmaf(r13, r26, r23);
    r23 = fmaf(r15, r29, r23);
    r29 = copysign(1.0, r23);
    r29 = fmaf(r6, r29, r23);
    r29 = 1.0 / r29;
    r8 = fmaf(r29, r30, r8);
    r9 = fmaf(r9, r10, r7);
    r7 = r18 * r19;
    r7 = fmaf(r22, r7, r24);
    r7 = fmaf(r13, r7, r11);
    r21 = fmaf(r16, r21, r31);
    r27 = r28 + r27;
    r27 = r27 + r12;
    r7 = fmaf(r15, r21, r7);
    r7 = fmaf(r14, r27, r7);
    r27 = r5 * r7;
    r9 = fmaf(r29, r27, r9);
    r3 = fmaf(r3, r9, r2 * r8);
    r3 = r3 * r3;
    ReadIdx3<1024, float, float, float4>(weight_loss,
                                         4 * weight_loss_num_alloc,
                                         global_thread_idx,
                                         r2,
                                         r27,
                                         r29);
    r14 = 0.00000000000000000e+00;
    r29 = fmaxf(r29, r14);
    r21 = sqrtf(r29);
    r9 = fmaf(r1, r9, r0 * r8);
    r9 = r9 * r9;
    r1 = r3 + r9;
    r8 = 5.00000000000000000e-01;
    r27 = fmaxf(r27, r6);
    r0 = r27 * r27;
    r15 = r22 * r27;
    r12 = fmaxf(r6, r1);
    r16 = sqrtf(r12);
    r15 = fmaf(r10, r0, r16 * r15);
    r15 = r1 <= r0 ? r1 : r15;
    r16 = 2.50000000000000000e+00;
    r31 = 1.0 / r0;
    r31 = fmaf(r1, r31, r28);
    r28 = logf(r31);
    r28 = r28 * r0;
    r15 = r2 < r16 ? r28 : r15;
    r28 = 1.50000000000000000e+00;
    r31 = sqrtf(r31);
    r31 = r10 + r31;
    r31 = r22 * r31;
    r31 = r31 * r0;
    r15 = r2 < r28 ? r31 : r15;
    r15 = r2 < r8 ? r1 : r15;
    r15 = fmaxf(r14, r15);
    r15 = r29 * r15;
    r12 = 1.0 / r12;
    r15 = r15 * r12;
    r15 = sqrtf(r15);
    r15 = r1 <= r6 ? r21 : r15;
    r15 = r15 * r15;
    r9 = fmaf(r15, r9, r15 * r3);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r9);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void PinholeScore(float* pose,
                  unsigned int pose_num_alloc,
                  SharedIndex* pose_indices,
                  float* calib,
                  unsigned int calib_num_alloc,
                  SharedIndex* calib_indices,
                  float* point,
                  unsigned int point_num_alloc,
                  SharedIndex* point_indices,
                  float* pixel,
                  unsigned int pixel_num_alloc,
                  float* weight_loss,
                  unsigned int weight_loss_num_alloc,
                  float* const out_rTr,
                  size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeScoreKernel<<<n_blocks, 1024>>>(pose,
                                         pose_num_alloc,
                                         pose_indices,
                                         calib,
                                         calib_num_alloc,
                                         calib_indices,
                                         point,
                                         point_num_alloc,
                                         point_indices,
                                         pixel,
                                         pixel_num_alloc,
                                         weight_loss,
                                         weight_loss_num_alloc,
                                         out_rTr,
                                         problem_size);
}

}  // namespace caspar