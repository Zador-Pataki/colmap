#include "kernel_pinhole_fixed_point_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeFixedPointJtjnjtrDirectKernel(float* pose_njtr,
                                         unsigned int pose_njtr_num_alloc,
                                         SharedIndex* pose_njtr_indices,
                                         float* pose_jac,
                                         unsigned int pose_jac_num_alloc,
                                         float* calib_njtr,
                                         unsigned int calib_njtr_num_alloc,
                                         SharedIndex* calib_njtr_indices,
                                         float* calib_jac,
                                         unsigned int calib_jac_num_alloc,
                                         float* const out_pose_njtr,
                                         unsigned int out_pose_njtr_num_alloc,
                                         float* const out_calib_njtr,
                                         unsigned int out_calib_njtr_num_alloc,
                                         size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_njtr_indices_loc[1024];
  pose_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex calib_njtr_indices_loc[1024];
  calib_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(
        pose_jac, 0 * pose_jac_num_alloc, global_thread_idx, r0, r1, r2, r3);
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
    r12 = fmaf(r6, r9, r7 * r11);
    ReadIdx4<1024, float, float, float4>(calib_jac,
                                         0 * calib_jac_num_alloc,
                                         global_thread_idx,
                                         r13,
                                         r14,
                                         r15,
                                         r16);
    r12 = fmaf(r4, r14, r12);
    r12 = fmaf(r5, r16, r12);
    r7 = fmaf(r7, r10, r6 * r8);
    r7 = fmaf(r4, r13, r7);
    r7 = fmaf(r5, r15, r7);
    r5 = fmaf(r0, r7, r1 * r12);
    r4 = fmaf(r2, r7, r3 * r12);
    ReadIdx4<1024, float, float, float4>(
        pose_jac, 4 * pose_jac_num_alloc, global_thread_idx, r6, r17, r18, r19);
    r20 = fmaf(r6, r7, r17 * r12);
    r21 = fmaf(r18, r7, r19 * r12);
    WriteSum4<float, float>((float*)inout_shared, r5, r4, r20, r21);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_njtr_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(
        pose_jac, 8 * pose_jac_num_alloc, global_thread_idx, r21, r20, r4, r5);
    r22 = fmaf(r21, r7, r20 * r12);
    r12 = fmaf(r5, r12, r4 * r7);
    WriteSum2<float, float>((float*)inout_shared, r22, r12);
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
                       r12,
                       r22);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r21 = fmaf(r12, r21, r22 * r4);
  };
  LoadShared<4, float, float>(pose_njtr,
                              0 * pose_njtr_num_alloc,
                              pose_njtr_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_njtr_indices_loc[threadIdx.x].target,
                       r4,
                       r7,
                       r23,
                       r24);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r21 = fmaf(r23, r6, r21);
    r21 = fmaf(r24, r18, r21);
    r21 = fmaf(r4, r0, r21);
    r21 = fmaf(r7, r2, r21);
    r19 = fmaf(r24, r19, r22 * r5);
    r19 = fmaf(r23, r17, r19);
    r19 = fmaf(r4, r1, r19);
    r19 = fmaf(r7, r3, r19);
    r19 = fmaf(r12, r20, r19);
    r14 = fmaf(r14, r19, r13 * r21);
    r16 = fmaf(r16, r19, r15 * r21);
    r9 = fmaf(r9, r19, r8 * r21);
    r19 = fmaf(r11, r19, r10 * r21);
    WriteSum4<float, float>((float*)inout_shared, r14, r16, r9, r19);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           0 * out_calib_njtr_num_alloc,
                           calib_njtr_indices_loc,
                           (float*)inout_shared);
}

void PinholeFixedPointJtjnjtrDirect(float* pose_njtr,
                                    unsigned int pose_njtr_num_alloc,
                                    SharedIndex* pose_njtr_indices,
                                    float* pose_jac,
                                    unsigned int pose_jac_num_alloc,
                                    float* calib_njtr,
                                    unsigned int calib_njtr_num_alloc,
                                    SharedIndex* calib_njtr_indices,
                                    float* calib_jac,
                                    unsigned int calib_jac_num_alloc,
                                    float* const out_pose_njtr,
                                    unsigned int out_pose_njtr_num_alloc,
                                    float* const out_calib_njtr,
                                    unsigned int out_calib_njtr_num_alloc,
                                    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeFixedPointJtjnjtrDirectKernel<<<n_blocks, 1024>>>(
      pose_njtr,
      pose_njtr_num_alloc,
      pose_njtr_indices,
      pose_jac,
      pose_jac_num_alloc,
      calib_njtr,
      calib_njtr_num_alloc,
      calib_njtr_indices,
      calib_jac,
      calib_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar