#include "kernel_pinhole_split_intrinsics_prior_fixed_focal_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitIntrinsicsPriorFixedFocalScoreKernel(
        float* principal_point,
        unsigned int principal_point_num_alloc,
        SharedIndex* principal_point_indices,
        float* prior,
        unsigned int prior_num_alloc,
        float* inv_std,
        unsigned int inv_std_num_alloc,
        float* focal,
        unsigned int focal_num_alloc,
        float* const out_rTr,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ SharedIndex principal_point_indices_loc[1024];
  principal_point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? principal_point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9;
  LoadShared<2, float, float>(principal_point,
                              0 * principal_point_num_alloc,
                              principal_point_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared,
                       principal_point_indices_loc[threadIdx.x].target,
                       r0,
                       r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(
        prior, 0 * prior_num_alloc, global_thread_idx, r2, r3, r4, r5);
    r6 = -1.00000000000000000e+00;
    r4 = fmaf(r4, r6, r0);
    r4 = r4 * r4;
    ReadIdx4<1024, float, float, float4>(
        inv_std, 0 * inv_std_num_alloc, global_thread_idx, r0, r7, r8, r9);
    r8 = r8 * r8;
    r5 = fmaf(r5, r6, r1);
    r5 = r5 * r5;
    r9 = r9 * r9;
    r9 = fmaf(r5, r9, r4 * r8);
    ReadIdx2<1024, float, float, float2>(
        focal, 0 * focal_num_alloc, global_thread_idx, r5, r8);
    r3 = fmaf(r3, r6, r8);
    r3 = r3 * r3;
    r7 = r7 * r7;
    r6 = fmaf(r2, r6, r5);
    r6 = r6 * r6;
    r0 = r0 * r0;
    r9 = fmaf(r3, r7, r9);
    r9 = fmaf(r6, r0, r9);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r9);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void PinholeSplitIntrinsicsPriorFixedFocalScore(
    float* principal_point,
    unsigned int principal_point_num_alloc,
    SharedIndex* principal_point_indices,
    float* prior,
    unsigned int prior_num_alloc,
    float* inv_std,
    unsigned int inv_std_num_alloc,
    float* focal,
    unsigned int focal_num_alloc,
    float* const out_rTr,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitIntrinsicsPriorFixedFocalScoreKernel<<<n_blocks, 1024>>>(
      principal_point,
      principal_point_num_alloc,
      principal_point_indices,
      prior,
      prior_num_alloc,
      inv_std,
      inv_std_num_alloc,
      focal,
      focal_num_alloc,
      out_rTr,
      problem_size);
}

}  // namespace caspar