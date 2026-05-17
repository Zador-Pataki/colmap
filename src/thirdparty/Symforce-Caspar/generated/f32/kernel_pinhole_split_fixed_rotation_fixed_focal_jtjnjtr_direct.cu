#include "kernel_pinhole_split_fixed_rotation_fixed_focal_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedRotationFixedFocalJtjnjtrDirectKernel(
        float* translation_njtr,
        unsigned int translation_njtr_num_alloc,
        SharedIndex* translation_njtr_indices,
        float* translation_jac,
        unsigned int translation_jac_num_alloc,
        float* principal_point_njtr,
        unsigned int principal_point_njtr_num_alloc,
        SharedIndex* principal_point_njtr_indices,
        float* principal_point_jac,
        unsigned int principal_point_jac_num_alloc,
        float* point_njtr,
        unsigned int point_njtr_num_alloc,
        SharedIndex* point_njtr_indices,
        float* point_jac,
        unsigned int point_jac_num_alloc,
        float* const out_translation_njtr,
        unsigned int out_translation_njtr_num_alloc,
        float* const out_principal_point_njtr,
        unsigned int out_principal_point_njtr_num_alloc,
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

  __shared__ SharedIndex principal_point_njtr_indices_loc[1024];
  principal_point_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? principal_point_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex point_njtr_indices_loc[1024];
  point_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(translation_jac,
                                         0 * translation_jac_num_alloc,
                                         global_thread_idx,
                                         r0,
                                         r1,
                                         r2,
                                         r3);
  };
  LoadShared<3, float, float>(point_njtr,
                              0 * point_njtr_num_alloc,
                              point_njtr_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_njtr_indices_loc[threadIdx.x].target,
                       r4,
                       r5,
                       r6);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(
        point_jac, 4 * point_jac_num_alloc, global_thread_idx, r7, r8);
    ReadIdx4<1024, float, float, float4>(point_jac,
                                         0 * point_jac_num_alloc,
                                         global_thread_idx,
                                         r9,
                                         r10,
                                         r11,
                                         r12);
    r13 = fmaf(r5, r12, r6 * r8);
    r13 = fmaf(r4, r10, r13);
  };
  LoadShared<2, float, float>(principal_point_njtr,
                              0 * principal_point_njtr_num_alloc,
                              principal_point_njtr_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared,
                       principal_point_njtr_indices_loc[threadIdx.x].target,
                       r14,
                       r15);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(principal_point_jac,
                                         0 * principal_point_jac_num_alloc,
                                         global_thread_idx,
                                         r16,
                                         r17,
                                         r18,
                                         r19);
    r20 = fmaf(r15, r19, r14 * r17);
    r21 = r13 + r20;
    r5 = fmaf(r5, r11, r6 * r7);
    r5 = fmaf(r4, r9, r5);
    r15 = fmaf(r15, r18, r14 * r16);
    r14 = r5 + r15;
    r4 = fmaf(r0, r14, r1 * r21);
    r6 = fmaf(r2, r14, r3 * r21);
    ReadIdx2<1024, float, float, float2>(translation_jac,
                                         4 * translation_jac_num_alloc,
                                         global_thread_idx,
                                         r22,
                                         r23);
    r14 = fmaf(r22, r14, r23 * r21);
    WriteSum3<float, float>((float*)inout_shared, r4, r6, r14);
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
                       r14,
                       r6,
                       r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r3 = fmaf(r6, r3, r14 * r1);
    r3 = fmaf(r4, r23, r3);
    r13 = r13 + r3;
    r2 = fmaf(r6, r2, r14 * r0);
    r2 = fmaf(r4, r22, r2);
    r5 = r5 + r2;
    r16 = fmaf(r16, r5, r17 * r13);
    r5 = fmaf(r18, r5, r19 * r13);
    WriteSum2<float, float>((float*)inout_shared, r16, r5);
  };
  FlushSumShared<2, float>(out_principal_point_njtr,
                           0 * out_principal_point_njtr_num_alloc,
                           principal_point_njtr_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r15 = r2 + r15;
    r20 = r3 + r20;
    r10 = fmaf(r10, r20, r9 * r15);
    r12 = fmaf(r12, r20, r11 * r15);
    r20 = fmaf(r8, r20, r7 * r15);
    WriteSum3<float, float>((float*)inout_shared, r10, r12, r20);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_njtr_indices_loc,
                           (float*)inout_shared);
}

void PinholeSplitFixedRotationFixedFocalJtjnjtrDirect(
    float* translation_njtr,
    unsigned int translation_njtr_num_alloc,
    SharedIndex* translation_njtr_indices,
    float* translation_jac,
    unsigned int translation_jac_num_alloc,
    float* principal_point_njtr,
    unsigned int principal_point_njtr_num_alloc,
    SharedIndex* principal_point_njtr_indices,
    float* principal_point_jac,
    unsigned int principal_point_jac_num_alloc,
    float* point_njtr,
    unsigned int point_njtr_num_alloc,
    SharedIndex* point_njtr_indices,
    float* point_jac,
    unsigned int point_jac_num_alloc,
    float* const out_translation_njtr,
    unsigned int out_translation_njtr_num_alloc,
    float* const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    float* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitFixedRotationFixedFocalJtjnjtrDirectKernel<<<n_blocks, 1024>>>(
      translation_njtr,
      translation_njtr_num_alloc,
      translation_njtr_indices,
      translation_jac,
      translation_jac_num_alloc,
      principal_point_njtr,
      principal_point_njtr_num_alloc,
      principal_point_njtr_indices,
      principal_point_jac,
      principal_point_jac_num_alloc,
      point_njtr,
      point_njtr_num_alloc,
      point_njtr_indices,
      point_jac,
      point_jac_num_alloc,
      out_translation_njtr,
      out_translation_njtr_num_alloc,
      out_principal_point_njtr,
      out_principal_point_njtr_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar