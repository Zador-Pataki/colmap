#include "kernel_pinhole_fixed_pose_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeFixedPoseJtjnjtrDirectKernel(float* calib_njtr,
                                        unsigned int calib_njtr_num_alloc,
                                        SharedIndex* calib_njtr_indices,
                                        float* calib_jac,
                                        unsigned int calib_jac_num_alloc,
                                        float* point_njtr,
                                        unsigned int point_njtr_num_alloc,
                                        SharedIndex* point_njtr_indices,
                                        float* point_jac,
                                        unsigned int point_jac_num_alloc,
                                        float* const out_calib_njtr,
                                        unsigned int out_calib_njtr_num_alloc,
                                        float* const out_point_njtr,
                                        unsigned int out_point_njtr_num_alloc,
                                        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

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
      r16, r17, r18;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(
        calib_jac, 0 * calib_jac_num_alloc, global_thread_idx, r0, r1, r2, r3);
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
    r5 = fmaf(r5, r11, r6 * r7);
    r5 = fmaf(r4, r9, r5);
    r4 = fmaf(r0, r5, r1 * r13);
    r6 = fmaf(r2, r5, r3 * r13);
    ReadIdx4<1024, float, float, float4>(calib_jac,
                                         4 * calib_jac_num_alloc,
                                         global_thread_idx,
                                         r14,
                                         r15,
                                         r16,
                                         r17);
    r18 = fmaf(r14, r5, r15 * r13);
    r5 = fmaf(r16, r5, r17 * r13);
    WriteSum4<float, float>((float*)inout_shared, r4, r6, r18, r5);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           0 * out_calib_njtr_num_alloc,
                           calib_njtr_indices_loc,
                           (float*)inout_shared);
  LoadShared<4, float, float>(calib_njtr,
                              0 * calib_njtr_num_alloc,
                              calib_njtr_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       calib_njtr_indices_loc[threadIdx.x].target,
                       r5,
                       r18,
                       r6,
                       r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r15 = fmaf(r6, r15, r4 * r17);
    r15 = fmaf(r5, r1, r15);
    r15 = fmaf(r18, r3, r15);
    r16 = fmaf(r4, r16, r6 * r14);
    r16 = fmaf(r5, r0, r16);
    r16 = fmaf(r18, r2, r16);
    r9 = fmaf(r9, r16, r10 * r15);
    r11 = fmaf(r11, r16, r12 * r15);
    r16 = fmaf(r7, r16, r8 * r15);
    WriteSum3<float, float>((float*)inout_shared, r9, r11, r16);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_njtr_indices_loc,
                           (float*)inout_shared);
}

void PinholeFixedPoseJtjnjtrDirect(float* calib_njtr,
                                   unsigned int calib_njtr_num_alloc,
                                   SharedIndex* calib_njtr_indices,
                                   float* calib_jac,
                                   unsigned int calib_jac_num_alloc,
                                   float* point_njtr,
                                   unsigned int point_njtr_num_alloc,
                                   SharedIndex* point_njtr_indices,
                                   float* point_jac,
                                   unsigned int point_jac_num_alloc,
                                   float* const out_calib_njtr,
                                   unsigned int out_calib_njtr_num_alloc,
                                   float* const out_point_njtr,
                                   unsigned int out_point_njtr_num_alloc,
                                   size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeFixedPoseJtjnjtrDirectKernel<<<n_blocks, 1024>>>(
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
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar