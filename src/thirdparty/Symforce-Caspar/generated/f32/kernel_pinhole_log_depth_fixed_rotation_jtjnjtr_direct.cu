#include "kernel_pinhole_log_depth_fixed_rotation_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeLogDepthFixedRotationJtjnjtrDirectKernel(
        float* translation_njtr,
        unsigned int translation_njtr_num_alloc,
        SharedIndex* translation_njtr_indices,
        float* translation_jac,
        unsigned int translation_jac_num_alloc,
        float* scale_njtr,
        unsigned int scale_njtr_num_alloc,
        SharedIndex* scale_njtr_indices,
        float* scale_jac,
        unsigned int scale_jac_num_alloc,
        float* point_njtr,
        unsigned int point_njtr_num_alloc,
        SharedIndex* point_njtr_indices,
        float* point_jac,
        unsigned int point_jac_num_alloc,
        float* const out_translation_njtr,
        unsigned int out_translation_njtr_num_alloc,
        float* const out_scale_njtr,
        unsigned int out_scale_njtr_num_alloc,
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

  __shared__ SharedIndex scale_njtr_indices_loc[1024];
  scale_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? scale_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex point_njtr_indices_loc[1024];
  point_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9;

  if (global_thread_idx < problem_size) {
    r0 = 0.00000000000000000e+00;
    ReadIdx1<1024, float, float, float>(
        translation_jac, 0 * translation_jac_num_alloc, global_thread_idx, r1);
  };
  LoadShared<1, float, float>(scale_njtr,
                              0 * scale_njtr_num_alloc,
                              scale_njtr_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>(
        (float*)inout_shared, scale_njtr_indices_loc[threadIdx.x].target, r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx1<1024, float, float, float>(
        scale_jac, 0 * scale_jac_num_alloc, global_thread_idx, r3);
    r2 = r2 * r3;
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
    ReadIdx3<1024, float, float, float4>(
        point_jac, 0 * point_jac_num_alloc, global_thread_idx, r7, r8, r9);
    r5 = fmaf(r5, r8, r6 * r9);
    r5 = fmaf(r4, r7, r5);
    r4 = r2 + r5;
    r4 = r1 * r4;
    WriteSum3<float, float>((float*)inout_shared, r0, r0, r4);
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
                       r4,
                       r0,
                       r6);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r1 = r6 * r1;
    r5 = r1 + r5;
    r5 = r3 * r5;
    WriteSum1<float, float>((float*)inout_shared, r5);
  };
  FlushSumShared<1, float>(out_scale_njtr,
                           0 * out_scale_njtr_num_alloc,
                           scale_njtr_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r2 + r1;
    r7 = r7 * r1;
    r8 = r8 * r1;
    r1 = r9 * r1;
    WriteSum3<float, float>((float*)inout_shared, r7, r8, r1);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_njtr_indices_loc,
                           (float*)inout_shared);
}

void PinholeLogDepthFixedRotationJtjnjtrDirect(
    float* translation_njtr,
    unsigned int translation_njtr_num_alloc,
    SharedIndex* translation_njtr_indices,
    float* translation_jac,
    unsigned int translation_jac_num_alloc,
    float* scale_njtr,
    unsigned int scale_njtr_num_alloc,
    SharedIndex* scale_njtr_indices,
    float* scale_jac,
    unsigned int scale_jac_num_alloc,
    float* point_njtr,
    unsigned int point_njtr_num_alloc,
    SharedIndex* point_njtr_indices,
    float* point_jac,
    unsigned int point_jac_num_alloc,
    float* const out_translation_njtr,
    unsigned int out_translation_njtr_num_alloc,
    float* const out_scale_njtr,
    unsigned int out_scale_njtr_num_alloc,
    float* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeLogDepthFixedRotationJtjnjtrDirectKernel<<<n_blocks, 1024>>>(
      translation_njtr,
      translation_njtr_num_alloc,
      translation_njtr_indices,
      translation_jac,
      translation_jac_num_alloc,
      scale_njtr,
      scale_njtr_num_alloc,
      scale_njtr_indices,
      scale_jac,
      scale_jac_num_alloc,
      point_njtr,
      point_njtr_num_alloc,
      point_njtr_indices,
      point_jac,
      point_jac_num_alloc,
      out_translation_njtr,
      out_translation_njtr_num_alloc,
      out_scale_njtr,
      out_scale_njtr_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar