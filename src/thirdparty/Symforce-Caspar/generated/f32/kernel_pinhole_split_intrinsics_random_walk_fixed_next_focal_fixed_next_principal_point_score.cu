#include "kernel_pinhole_split_intrinsics_random_walk_fixed_next_focal_fixed_next_principal_point_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitIntrinsicsRandomWalkFixedNextFocalFixedNextPrincipalPointScoreKernel(
        float* prev_focal,
        unsigned int prev_focal_num_alloc,
        SharedIndex* prev_focal_indices,
        float* prev_principal_point,
        unsigned int prev_principal_point_num_alloc,
        SharedIndex* prev_principal_point_indices,
        float* inv_std,
        unsigned int inv_std_num_alloc,
        float* next_focal,
        unsigned int next_focal_num_alloc,
        float* next_principal_point,
        unsigned int next_principal_point_num_alloc,
        float* const out_rTr,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ SharedIndex prev_focal_indices_loc[1024];
  prev_focal_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? prev_focal_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex prev_principal_point_indices_loc[1024];
  prev_principal_point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? prev_principal_point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(next_principal_point,
                                         0 * next_principal_point_num_alloc,
                                         global_thread_idx,
                                         r0,
                                         r1);
  };
  LoadShared<2, float, float>(prev_principal_point,
                              0 * prev_principal_point_num_alloc,
                              prev_principal_point_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared,
                       prev_principal_point_indices_loc[threadIdx.x].target,
                       r2,
                       r3);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r4 = -1.00000000000000000e+00;
    r2 = fmaf(r2, r4, r0);
    r2 = r2 * r2;
    ReadIdx4<1024, float, float, float4>(
        inv_std, 0 * inv_std_num_alloc, global_thread_idx, r0, r5, r6, r7);
    r6 = r6 * r6;
    r3 = fmaf(r3, r4, r1);
    r3 = r3 * r3;
    r7 = r7 * r7;
    r7 = fmaf(r3, r7, r2 * r6);
    ReadIdx2<1024, float, float, float2>(
        next_focal, 0 * next_focal_num_alloc, global_thread_idx, r3, r6);
  };
  LoadShared<2, float, float>(prev_focal,
                              0 * prev_focal_num_alloc,
                              prev_focal_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared,
                       prev_focal_indices_loc[threadIdx.x].target,
                       r2,
                       r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r2 = fmaf(r2, r4, r3);
    r2 = r2 * r2;
    r0 = r0 * r0;
    r4 = fmaf(r1, r4, r6);
    r4 = r4 * r4;
    r5 = r5 * r5;
    r7 = fmaf(r2, r0, r7);
    r7 = fmaf(r4, r5, r7);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r7);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void PinholeSplitIntrinsicsRandomWalkFixedNextFocalFixedNextPrincipalPointScore(
    float* prev_focal,
    unsigned int prev_focal_num_alloc,
    SharedIndex* prev_focal_indices,
    float* prev_principal_point,
    unsigned int prev_principal_point_num_alloc,
    SharedIndex* prev_principal_point_indices,
    float* inv_std,
    unsigned int inv_std_num_alloc,
    float* next_focal,
    unsigned int next_focal_num_alloc,
    float* next_principal_point,
    unsigned int next_principal_point_num_alloc,
    float* const out_rTr,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitIntrinsicsRandomWalkFixedNextFocalFixedNextPrincipalPointScoreKernel<<<
      n_blocks,
      1024>>>(prev_focal,
              prev_focal_num_alloc,
              prev_focal_indices,
              prev_principal_point,
              prev_principal_point_num_alloc,
              prev_principal_point_indices,
              inv_std,
              inv_std_num_alloc,
              next_focal,
              next_focal_num_alloc,
              next_principal_point,
              next_principal_point_num_alloc,
              out_rTr,
              problem_size);
}

}  // namespace caspar