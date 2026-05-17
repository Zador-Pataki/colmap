#include "kernel_pinhole_split_fixed_focal_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedFocalScoreKernel(float* pose,
                                      unsigned int pose_num_alloc,
                                      SharedIndex* pose_indices,
                                      float* principal_point,
                                      unsigned int principal_point_num_alloc,
                                      SharedIndex* principal_point_indices,
                                      float* point,
                                      unsigned int point_num_alloc,
                                      SharedIndex* point_indices,
                                      float* pixel,
                                      unsigned int pixel_num_alloc,
                                      float* weight_loss,
                                      unsigned int weight_loss_num_alloc,
                                      float* focal,
                                      unsigned int focal_num_alloc,
                                      float* const out_rTr,
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
  LoadShared<2, float, float>(principal_point,
                              0 * principal_point_num_alloc,
                              principal_point_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared,
                       principal_point_indices_loc[threadIdx.x].target,
                       r4,
                       r5);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r6, r7);
    r8 = -1.00000000000000000e+00;
    r6 = fmaf(r6, r8, r4);
    ReadIdx2<1024, float, float, float2>(
        focal, 0 * focal_num_alloc, global_thread_idx, r4, r9);
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r10,
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
    r25 = fmaf(r14, r25, r10);
    r10 = r17 * r19;
    r26 = r18 * r23;
    r10 = fmaf(r22, r10, r26);
    r27 = r18 * r18;
    r27 = r20 * r27;
    r28 = 1.00000000000000000e+00;
    r29 = r17 * r17;
    r29 = fmaf(r20, r29, r28);
    r30 = r27 + r29;
    r25 = fmaf(r15, r10, r25);
    r25 = fmaf(r13, r30, r25);
    r30 = r4 * r25;
    r10 = 9.99999999999999955e-07;
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
    r29 = fmaf(r10, r29, r23);
    r29 = 1.0 / r29;
    r6 = fmaf(r29, r30, r6);
    r7 = fmaf(r7, r8, r5);
    r5 = r18 * r19;
    r5 = fmaf(r22, r5, r24);
    r5 = fmaf(r13, r5, r11);
    r21 = fmaf(r16, r21, r31);
    r27 = r28 + r27;
    r27 = r27 + r12;
    r5 = fmaf(r15, r21, r5);
    r5 = fmaf(r14, r27, r5);
    r27 = r9 * r5;
    r7 = fmaf(r29, r27, r7);
    r3 = fmaf(r3, r7, r2 * r6);
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
    r7 = fmaf(r1, r7, r0 * r6);
    r7 = r7 * r7;
    r1 = r3 + r7;
    r6 = 5.00000000000000000e-01;
    r27 = fmaxf(r27, r10);
    r0 = r27 * r27;
    r15 = r22 * r27;
    r12 = fmaxf(r10, r1);
    r16 = sqrtf(r12);
    r15 = fmaf(r8, r0, r16 * r15);
    r15 = r1 <= r0 ? r1 : r15;
    r16 = 2.50000000000000000e+00;
    r31 = 1.0 / r0;
    r31 = fmaf(r1, r31, r28);
    r28 = logf(r31);
    r28 = r28 * r0;
    r15 = r2 < r16 ? r28 : r15;
    r28 = 1.50000000000000000e+00;
    r31 = sqrtf(r31);
    r31 = r8 + r31;
    r31 = r22 * r31;
    r31 = r31 * r0;
    r15 = r2 < r28 ? r31 : r15;
    r15 = r2 < r6 ? r1 : r15;
    r15 = fmaxf(r14, r15);
    r15 = r29 * r15;
    r12 = 1.0 / r12;
    r15 = r15 * r12;
    r15 = sqrtf(r15);
    r15 = r1 <= r10 ? r21 : r15;
    r15 = r15 * r15;
    r7 = fmaf(r15, r7, r15 * r3);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r7);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void PinholeSplitFixedFocalScore(float* pose,
                                 unsigned int pose_num_alloc,
                                 SharedIndex* pose_indices,
                                 float* principal_point,
                                 unsigned int principal_point_num_alloc,
                                 SharedIndex* principal_point_indices,
                                 float* point,
                                 unsigned int point_num_alloc,
                                 SharedIndex* point_indices,
                                 float* pixel,
                                 unsigned int pixel_num_alloc,
                                 float* weight_loss,
                                 unsigned int weight_loss_num_alloc,
                                 float* focal,
                                 unsigned int focal_num_alloc,
                                 float* const out_rTr,
                                 size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitFixedFocalScoreKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      principal_point,
      principal_point_num_alloc,
      principal_point_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      weight_loss,
      weight_loss_num_alloc,
      focal,
      focal_num_alloc,
      out_rTr,
      problem_size);
}

}  // namespace caspar