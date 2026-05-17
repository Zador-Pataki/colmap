#include "kernel_pinhole_log_depth_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeLogDepthScoreKernel(float* pose,
                               unsigned int pose_num_alloc,
                               SharedIndex* pose_indices,
                               float* scale,
                               unsigned int scale_num_alloc,
                               SharedIndex* scale_indices,
                               float* point,
                               unsigned int point_num_alloc,
                               SharedIndex* point_indices,
                               float* log_depth,
                               unsigned int log_depth_num_alloc,
                               float* loss,
                               unsigned int loss_num_alloc,
                               float* const out_rTr,
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

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17;

  if (global_thread_idx < problem_size) {
    r0 = 0.00000000000000000e+00;
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>(
        (float*)inout_shared, pose_indices_loc[threadIdx.x].target, r1, r2, r3);
  };
  __syncthreads();
  LoadShared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_indices_loc[threadIdx.x].target,
                       r2,
                       r1,
                       r4);
  };
  __syncthreads();
  LoadShared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r5,
                       r6,
                       r7,
                       r8);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r9 = r5 * r8;
    r10 = 2.00000000000000000e+00;
    r7 = r7 * r10;
    r9 = fmaf(r6, r7, r10 * r9);
    r9 = fmaf(r1, r9, r3);
    r1 = -2.00000000000000000e+00;
    r3 = r6 * r1;
    r7 = fmaf(r8, r3, r5 * r7);
    r11 = 1.00000000000000000e+00;
    r3 = fmaf(r6, r3, r11);
    r6 = r5 * r5;
    r3 = fmaf(r1, r6, r3);
    r9 = fmaf(r2, r7, r9);
    r9 = fmaf(r4, r3, r9);
  };
  LoadShared<1, float, float>(
      scale, 0 * scale_num_alloc, scale_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>(
        (float*)inout_shared, scale_indices_loc[threadIdx.x].target, r3);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r4 = -1.00000000000000000e+00;
    ReadIdx1<1024, float, float, float>(
        log_depth, 0 * log_depth_num_alloc, global_thread_idx, r7);
    r7 = fmaf(r7, r4, r3 * r4);
    r3 = 9.99999999999999955e-07;
    r2 = fmaxf(r9, r3);
    r2 = logf(r2);
    r7 = r7 + r2;
    r7 = r0 < r9 ? r7 : r0;
    r7 = r7 * r7;
    ReadIdx3<1024, float, float, float4>(
        loss, 0 * loss_num_alloc, global_thread_idx, r9, r2, r6);
    r6 = fmaxf(r6, r0);
    r1 = sqrtf(r6);
    r12 = 5.00000000000000000e-01;
    r2 = fmaxf(r2, r3);
    r13 = r2 * r2;
    r14 = r10 * r2;
    r15 = fmaxf(r3, r7);
    r16 = sqrtf(r15);
    r14 = fmaf(r16, r14, r4 * r13);
    r14 = r7 <= r13 ? r7 : r14;
    r16 = 2.50000000000000000e+00;
    r17 = 1.0 / r13;
    r17 = fmaf(r17, r7, r11);
    r11 = logf(r17);
    r11 = r11 * r13;
    r14 = r9 < r16 ? r11 : r14;
    r11 = 1.50000000000000000e+00;
    r17 = sqrtf(r17);
    r17 = r4 + r17;
    r17 = r10 * r17;
    r17 = r17 * r13;
    r14 = r9 < r11 ? r17 : r14;
    r14 = r9 < r12 ? r7 : r14;
    r14 = fmaxf(r0, r14);
    r14 = r6 * r14;
    r15 = 1.0 / r15;
    r14 = r14 * r15;
    r14 = sqrtf(r14);
    r14 = r7 <= r3 ? r1 : r14;
    r14 = r14 * r14;
    r14 = r7 * r14;
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r14);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void PinholeLogDepthScore(float* pose,
                          unsigned int pose_num_alloc,
                          SharedIndex* pose_indices,
                          float* scale,
                          unsigned int scale_num_alloc,
                          SharedIndex* scale_indices,
                          float* point,
                          unsigned int point_num_alloc,
                          SharedIndex* point_indices,
                          float* log_depth,
                          unsigned int log_depth_num_alloc,
                          float* loss,
                          unsigned int loss_num_alloc,
                          float* const out_rTr,
                          size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeLogDepthScoreKernel<<<n_blocks, 1024>>>(pose,
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
                                                 out_rTr,
                                                 problem_size);
}

}  // namespace caspar