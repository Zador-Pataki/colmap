#include "kernel_pinhole_log_depth_fixed_rotation_fixed_scale_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeLogDepthFixedRotationFixedScaleScoreKernel(
        float* rotation,
        unsigned int rotation_num_alloc,
        float* translation,
        unsigned int translation_num_alloc,
        SharedIndex* translation_indices,
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* log_depth,
        unsigned int log_depth_num_alloc,
        float* loss,
        unsigned int loss_num_alloc,
        float* scale,
        unsigned int scale_num_alloc,
        float* const out_rTr,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex translation_indices_loc[1024];
  translation_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? translation_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18;

  if (global_thread_idx < problem_size) {
    r0 = 0.00000000000000000e+00;
  };
  LoadShared<3, float, float>(translation,
                              0 * translation_num_alloc,
                              translation_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       translation_indices_loc[threadIdx.x].target,
                       r1,
                       r2,
                       r3);
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
  if (global_thread_idx < problem_size) {
    r5 = 1.00000000000000000e+00;
    r6 = -2.00000000000000000e+00;
    ReadIdx4<1024, float, float, float4>(
        rotation, 0 * rotation_num_alloc, global_thread_idx, r7, r8, r9, r10);
    r11 = r8 * r8;
    r11 = fmaf(r6, r11, r5);
    r12 = r7 * r7;
    r11 = fmaf(r6, r12, r11);
    r11 = fmaf(r4, r11, r3);
    r4 = r8 * r10;
    r3 = 2.00000000000000000e+00;
    r9 = r9 * r3;
    r4 = fmaf(r7, r9, r6 * r4);
    r6 = r7 * r10;
    r9 = fmaf(r8, r9, r3 * r6);
    r11 = fmaf(r2, r4, r11);
    r11 = fmaf(r1, r9, r11);
    r9 = 9.99999999999999955e-07;
    r1 = fmaxf(r9, r11);
    r1 = logf(r1);
    ReadIdx1<1024, float, float, float>(
        scale, 0 * scale_num_alloc, global_thread_idx, r4);
    r2 = -1.00000000000000000e+00;
    r4 = fmaf(r4, r2, r1);
    ReadIdx1<1024, float, float, float>(
        log_depth, 0 * log_depth_num_alloc, global_thread_idx, r1);
    r4 = fmaf(r1, r2, r4);
    r4 = r0 < r11 ? r4 : r0;
    r4 = r4 * r4;
    ReadIdx3<1024, float, float, float4>(
        loss, 0 * loss_num_alloc, global_thread_idx, r11, r1, r6);
    r6 = fmaxf(r6, r0);
    r12 = sqrtf(r6);
    r13 = 5.00000000000000000e-01;
    r1 = fmaxf(r1, r9);
    r14 = r1 * r1;
    r15 = r3 * r1;
    r16 = fmaxf(r9, r4);
    r17 = sqrtf(r16);
    r15 = fmaf(r17, r15, r2 * r14);
    r15 = r4 <= r14 ? r4 : r15;
    r17 = 2.50000000000000000e+00;
    r18 = 1.0 / r14;
    r18 = fmaf(r18, r4, r5);
    r5 = logf(r18);
    r5 = r5 * r14;
    r15 = r11 < r17 ? r5 : r15;
    r5 = 1.50000000000000000e+00;
    r18 = sqrtf(r18);
    r18 = r2 + r18;
    r18 = r3 * r18;
    r18 = r18 * r14;
    r15 = r11 < r5 ? r18 : r15;
    r15 = r11 < r13 ? r4 : r15;
    r15 = fmaxf(r0, r15);
    r15 = r6 * r15;
    r16 = 1.0 / r16;
    r15 = r15 * r16;
    r15 = sqrtf(r15);
    r15 = r4 <= r9 ? r12 : r15;
    r15 = r15 * r15;
    r15 = r4 * r15;
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r15);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void PinholeLogDepthFixedRotationFixedScaleScore(
    float* rotation,
    unsigned int rotation_num_alloc,
    float* translation,
    unsigned int translation_num_alloc,
    SharedIndex* translation_indices,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* log_depth,
    unsigned int log_depth_num_alloc,
    float* loss,
    unsigned int loss_num_alloc,
    float* scale,
    unsigned int scale_num_alloc,
    float* const out_rTr,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeLogDepthFixedRotationFixedScaleScoreKernel<<<n_blocks, 1024>>>(
      rotation,
      rotation_num_alloc,
      translation,
      translation_num_alloc,
      translation_indices,
      point,
      point_num_alloc,
      point_indices,
      log_depth,
      log_depth_num_alloc,
      loss,
      loss_num_alloc,
      scale,
      scale_num_alloc,
      out_rTr,
      problem_size);
}

}  // namespace caspar