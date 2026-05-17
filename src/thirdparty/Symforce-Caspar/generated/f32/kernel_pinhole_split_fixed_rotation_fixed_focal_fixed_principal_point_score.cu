#include "kernel_pinhole_split_fixed_rotation_fixed_focal_fixed_principal_point_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedRotationFixedFocalFixedPrincipalPointScoreKernel(
        float* rotation,
        unsigned int rotation_num_alloc,
        float* translation,
        unsigned int translation_num_alloc,
        SharedIndex* translation_indices,
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* weight_loss,
        unsigned int weight_loss_num_alloc,
        float* focal,
        unsigned int focal_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
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
    ReadIdx2<1024, float, float, float2>(principal_point,
                                         0 * principal_point_num_alloc,
                                         global_thread_idx,
                                         r4,
                                         r5);
    ReadIdx2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r6, r7);
    r8 = -1.00000000000000000e+00;
    r6 = fmaf(r6, r8, r4);
    ReadIdx2<1024, float, float, float2>(
        focal, 0 * focal_num_alloc, global_thread_idx, r4, r9);
  };
  LoadShared<3, float, float>(translation,
                              0 * translation_num_alloc,
                              translation_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       translation_indices_loc[threadIdx.x].target,
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
  if (global_thread_idx < problem_size) {
    r16 = -2.00000000000000000e+00;
    ReadIdx4<1024, float, float, float4>(rotation,
                                         0 * rotation_num_alloc,
                                         global_thread_idx,
                                         r17,
                                         r18,
                                         r19,
                                         r20);
    r21 = r19 * r19;
    r21 = r16 * r21;
    r22 = 1.00000000000000000e+00;
    r23 = r18 * r18;
    r23 = fmaf(r16, r23, r22);
    r24 = r21 + r23;
    r24 = fmaf(r13, r24, r10);
    r10 = r18 * r20;
    r25 = 2.00000000000000000e+00;
    r26 = r17 * r25;
    r27 = r19 * r26;
    r10 = fmaf(r25, r10, r27);
    r28 = r20 * r16;
    r29 = r18 * r26;
    r30 = fmaf(r19, r28, r29);
    r24 = fmaf(r15, r10, r24);
    r24 = fmaf(r14, r30, r24);
    r30 = r4 * r24;
    r10 = 9.99999999999999955e-07;
    r31 = r17 * r17;
    r31 = r31 * r16;
    r23 = r31 + r23;
    r23 = fmaf(r15, r23, r12);
    r27 = fmaf(r18, r28, r27);
    r12 = r18 * r19;
    r12 = r12 * r25;
    r26 = fmaf(r20, r26, r12);
    r23 = fmaf(r13, r27, r23);
    r23 = fmaf(r14, r26, r23);
    r26 = copysign(1.0, r23);
    r26 = fmaf(r10, r26, r23);
    r26 = 1.0 / r26;
    r6 = fmaf(r26, r30, r6);
    r7 = fmaf(r7, r8, r5);
    r21 = r22 + r21;
    r21 = r21 + r31;
    r21 = fmaf(r14, r21, r11);
    r28 = fmaf(r17, r28, r12);
    r17 = r19 * r20;
    r17 = fmaf(r25, r17, r29);
    r21 = fmaf(r15, r28, r21);
    r21 = fmaf(r13, r17, r21);
    r17 = r9 * r21;
    r7 = fmaf(r26, r17, r7);
    r1 = fmaf(r1, r7, r0 * r6);
    r1 = r1 * r1;
    ReadIdx3<1024, float, float, float4>(weight_loss,
                                         4 * weight_loss_num_alloc,
                                         global_thread_idx,
                                         r0,
                                         r17,
                                         r26);
    r13 = 0.00000000000000000e+00;
    r26 = fmaxf(r26, r13);
    r28 = sqrtf(r26);
    r7 = fmaf(r3, r7, r2 * r6);
    r7 = r7 * r7;
    r3 = r1 + r7;
    r6 = 5.00000000000000000e-01;
    r17 = fmaxf(r17, r10);
    r2 = r17 * r17;
    r15 = r25 * r17;
    r29 = fmaxf(r10, r3);
    r12 = sqrtf(r29);
    r15 = fmaf(r12, r15, r8 * r2);
    r15 = r3 <= r2 ? r3 : r15;
    r12 = 2.50000000000000000e+00;
    r14 = 1.0 / r2;
    r14 = fmaf(r3, r14, r22);
    r22 = logf(r14);
    r22 = r22 * r2;
    r15 = r0 < r12 ? r22 : r15;
    r22 = 1.50000000000000000e+00;
    r14 = sqrtf(r14);
    r14 = r8 + r14;
    r14 = r25 * r14;
    r14 = r14 * r2;
    r15 = r0 < r22 ? r14 : r15;
    r15 = r0 < r6 ? r3 : r15;
    r15 = fmaxf(r13, r15);
    r15 = r26 * r15;
    r29 = 1.0 / r29;
    r15 = r15 * r29;
    r15 = sqrtf(r15);
    r15 = r3 <= r10 ? r28 : r15;
    r15 = r15 * r15;
    r7 = fmaf(r15, r7, r15 * r1);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r7);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void PinholeSplitFixedRotationFixedFocalFixedPrincipalPointScore(
    float* rotation,
    unsigned int rotation_num_alloc,
    float* translation,
    unsigned int translation_num_alloc,
    SharedIndex* translation_indices,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* weight_loss,
    unsigned int weight_loss_num_alloc,
    float* focal,
    unsigned int focal_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    float* const out_rTr,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitFixedRotationFixedFocalFixedPrincipalPointScoreKernel<<<n_blocks,
                                                                      1024>>>(
      rotation,
      rotation_num_alloc,
      translation,
      translation_num_alloc,
      translation_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      weight_loss,
      weight_loss_num_alloc,
      focal,
      focal_num_alloc,
      principal_point,
      principal_point_num_alloc,
      out_rTr,
      problem_size);
}

}  // namespace caspar