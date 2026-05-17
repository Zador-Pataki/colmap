#include "kernel_pinhole_log_depth_fixed_point_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeLogDepthFixedPointJtjnjtrDirectKernel(
        float* pose_njtr,
        unsigned int pose_njtr_num_alloc,
        SharedIndex* pose_njtr_indices,
        float* pose_jac,
        unsigned int pose_jac_num_alloc,
        float* scale_njtr,
        unsigned int scale_njtr_num_alloc,
        SharedIndex* scale_njtr_indices,
        float* scale_jac,
        unsigned int scale_jac_num_alloc,
        float* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        float* const out_scale_njtr,
        unsigned int out_scale_njtr_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_njtr_indices_loc[1024];
  pose_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex scale_njtr_indices_loc[1024];
  scale_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? scale_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9;

  if (global_thread_idx < problem_size) {
    r0 = 0.00000000000000000e+00;
    ReadIdx4<1024, float, float, float4>(
        pose_jac, 0 * pose_jac_num_alloc, global_thread_idx, r1, r2, r3, r4);
  };
  LoadShared<1, float, float>(scale_njtr,
                              0 * scale_njtr_num_alloc,
                              scale_njtr_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>(
        (float*)inout_shared, scale_njtr_indices_loc[threadIdx.x].target, r5);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx1<1024, float, float, float>(
        scale_jac, 0 * scale_jac_num_alloc, global_thread_idx, r6);
    r5 = r5 * r6;
    r7 = r1 * r5;
    r8 = r2 * r5;
    r9 = r3 * r5;
    WriteSum4<float, float>((float*)inout_shared, r7, r8, r9, r0);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_njtr_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r5 = r4 * r5;
    WriteSum2<float, float>((float*)inout_shared, r0, r5);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_njtr_indices_loc,
                           (float*)inout_shared);
  LoadShared<2, float, float>(pose_njtr,
                              4 * pose_njtr_num_alloc,
                              pose_njtr_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared,
                       pose_njtr_indices_loc[threadIdx.x].target,
                       r5,
                       r0);
  };
  __syncthreads();
  LoadShared<4, float, float>(pose_njtr,
                              0 * pose_njtr_num_alloc,
                              pose_njtr_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_njtr_indices_loc[threadIdx.x].target,
                       r5,
                       r9,
                       r8,
                       r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r1 = fmaf(r5, r1, r0 * r4);
    r1 = fmaf(r8, r3, r1);
    r1 = fmaf(r9, r2, r1);
    r1 = r6 * r1;
    WriteSum1<float, float>((float*)inout_shared, r1);
  };
  FlushSumShared<1, float>(out_scale_njtr,
                           0 * out_scale_njtr_num_alloc,
                           scale_njtr_indices_loc,
                           (float*)inout_shared);
}

void PinholeLogDepthFixedPointJtjnjtrDirect(
    float* pose_njtr,
    unsigned int pose_njtr_num_alloc,
    SharedIndex* pose_njtr_indices,
    float* pose_jac,
    unsigned int pose_jac_num_alloc,
    float* scale_njtr,
    unsigned int scale_njtr_num_alloc,
    SharedIndex* scale_njtr_indices,
    float* scale_jac,
    unsigned int scale_jac_num_alloc,
    float* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    float* const out_scale_njtr,
    unsigned int out_scale_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeLogDepthFixedPointJtjnjtrDirectKernel<<<n_blocks, 1024>>>(
      pose_njtr,
      pose_njtr_num_alloc,
      pose_njtr_indices,
      pose_jac,
      pose_jac_num_alloc,
      scale_njtr,
      scale_njtr_num_alloc,
      scale_njtr_indices,
      scale_jac,
      scale_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_scale_njtr,
      out_scale_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar