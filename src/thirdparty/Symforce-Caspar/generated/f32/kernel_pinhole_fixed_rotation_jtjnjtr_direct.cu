#include "kernel_pinhole_fixed_rotation_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeFixedRotationJtjnjtrDirectKernel(
        float* translation_njtr,
        unsigned int translation_njtr_num_alloc,
        SharedIndex* translation_njtr_indices,
        float* translation_jac,
        unsigned int translation_jac_num_alloc,
        float* calib_njtr,
        unsigned int calib_njtr_num_alloc,
        SharedIndex* calib_njtr_indices,
        float* calib_jac,
        unsigned int calib_jac_num_alloc,
        float* point_njtr,
        unsigned int point_njtr_num_alloc,
        SharedIndex* point_njtr_indices,
        float* point_jac,
        unsigned int point_jac_num_alloc,
        float* const out_translation_njtr,
        unsigned int out_translation_njtr_num_alloc,
        float* const out_calib_njtr,
        unsigned int out_calib_njtr_num_alloc,
        float* const out_point_njtr,
        unsigned int out_point_njtr_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex translation_njtr_indices_loc[1024];
  translation_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? translation_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex calib_njtr_indices_loc[1024];
  calib_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex point_njtr_indices_loc[1024];
  point_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(translation_jac,
                                         0 * translation_jac_num_alloc,
                                         global_thread_idx,
                                         r0,
                                         r1,
                                         r2,
                                         r3);
  };
  LoadShared<4, float, float>(calib_njtr,
                              0 * calib_njtr_num_alloc,
                              calib_njtr_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       calib_njtr_indices_loc[threadIdx.x].target,
                       r4,
                       r5,
                       r6,
                       r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(calib_jac,
                                         4 * calib_jac_num_alloc,
                                         global_thread_idx,
                                         r8,
                                         r9,
                                         r10,
                                         r11);
    r12 = fmaf(r7, r10, r6 * r8);
    ReadIdx4<1024, float, float, float4>(calib_jac,
                                         0 * calib_jac_num_alloc,
                                         global_thread_idx,
                                         r13,
                                         r14,
                                         r15,
                                         r16);
    r12 = fmaf(r4, r13, r12);
    r12 = fmaf(r5, r15, r12);
  };
  LoadShared<3, float, float>(point_njtr,
                              0 * point_njtr_num_alloc,
                              point_njtr_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_njtr_indices_loc[threadIdx.x].target,
                       r17,
                       r18,
                       r19);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(
        point_jac, 4 * point_jac_num_alloc, global_thread_idx, r20, r21);
    ReadIdx4<1024, float, float, float4>(point_jac,
                                         0 * point_jac_num_alloc,
                                         global_thread_idx,
                                         r22,
                                         r23,
                                         r24,
                                         r25);
    r26 = fmaf(r18, r24, r19 * r20);
    r26 = fmaf(r17, r22, r26);
    r27 = r12 + r26;
    r6 = fmaf(r6, r9, r7 * r11);
    r6 = fmaf(r4, r14, r6);
    r6 = fmaf(r5, r16, r6);
    r18 = fmaf(r18, r25, r19 * r21);
    r18 = fmaf(r17, r23, r18);
    r17 = r6 + r18;
    r19 = fmaf(r1, r17, r0 * r27);
    r5 = fmaf(r3, r17, r2 * r27);
    ReadIdx2<1024, float, float, float2>(translation_jac,
                                         4 * translation_jac_num_alloc,
                                         global_thread_idx,
                                         r4,
                                         r7);
    r17 = fmaf(r7, r17, r4 * r27);
    WriteSum3<float, float>((float*)inout_shared, r19, r5, r17);
  };
  FlushSumShared<3, float>(out_translation_njtr,
                           0 * out_translation_njtr_num_alloc,
                           translation_njtr_indices_loc,
                           (float*)inout_shared);
  LoadShared<3, float, float>(translation_njtr,
                              0 * translation_njtr_num_alloc,
                              translation_njtr_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       translation_njtr_indices_loc[threadIdx.x].target,
                       r17,
                       r5,
                       r19);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r2 = fmaf(r5, r2, r17 * r0);
    r2 = fmaf(r19, r4, r2);
    r26 = r26 + r2;
    r3 = fmaf(r5, r3, r17 * r1);
    r3 = fmaf(r19, r7, r3);
    r18 = r18 + r3;
    r14 = fmaf(r14, r18, r13 * r26);
    r16 = fmaf(r16, r18, r15 * r26);
    r9 = fmaf(r9, r18, r8 * r26);
    r18 = fmaf(r11, r18, r10 * r26);
    WriteSum4<float, float>((float*)inout_shared, r14, r16, r9, r18);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           0 * out_calib_njtr_num_alloc,
                           calib_njtr_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r2 = r12 + r2;
    r3 = r6 + r3;
    r23 = fmaf(r23, r3, r22 * r2);
    r25 = fmaf(r25, r3, r24 * r2);
    r3 = fmaf(r21, r3, r20 * r2);
    WriteSum3<float, float>((float*)inout_shared, r23, r25, r3);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_njtr_indices_loc,
                           (float*)inout_shared);
}

void PinholeFixedRotationJtjnjtrDirect(
    float* translation_njtr,
    unsigned int translation_njtr_num_alloc,
    SharedIndex* translation_njtr_indices,
    float* translation_jac,
    unsigned int translation_jac_num_alloc,
    float* calib_njtr,
    unsigned int calib_njtr_num_alloc,
    SharedIndex* calib_njtr_indices,
    float* calib_jac,
    unsigned int calib_jac_num_alloc,
    float* point_njtr,
    unsigned int point_njtr_num_alloc,
    SharedIndex* point_njtr_indices,
    float* point_jac,
    unsigned int point_jac_num_alloc,
    float* const out_translation_njtr,
    unsigned int out_translation_njtr_num_alloc,
    float* const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc,
    float* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeFixedRotationJtjnjtrDirectKernel<<<n_blocks, 1024>>>(
      translation_njtr,
      translation_njtr_num_alloc,
      translation_njtr_indices,
      translation_jac,
      translation_jac_num_alloc,
      calib_njtr,
      calib_njtr_num_alloc,
      calib_njtr_indices,
      calib_jac,
      calib_jac_num_alloc,
      point_njtr,
      point_njtr_num_alloc,
      point_njtr_indices,
      point_jac,
      point_jac_num_alloc,
      out_translation_njtr,
      out_translation_njtr_num_alloc,
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar