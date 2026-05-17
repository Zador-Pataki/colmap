#include "kernel_pinhole_split_intrinsics_prior_fixed_principal_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitIntrinsicsPriorFixedPrincipalPointResJacKernel(
        float* focal,
        unsigned int focal_num_alloc,
        SharedIndex* focal_indices,
        float* prior,
        unsigned int prior_num_alloc,
        float* inv_std,
        unsigned int inv_std_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_focal_njtr,
        unsigned int out_focal_njtr_num_alloc,
        float* const out_focal_precond_diag,
        unsigned int out_focal_precond_diag_num_alloc,
        float* const out_focal_precond_tril,
        unsigned int out_focal_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ SharedIndex focal_indices_loc[1024];
  focal_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(
        inv_std, 0 * inv_std_num_alloc, global_thread_idx, r0, r1, r2, r3);
  };
  LoadShared<2, float, float>(
      focal, 0 * focal_num_alloc, focal_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>(
        (float*)inout_shared, focal_indices_loc[threadIdx.x].target, r4, r5);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(
        prior, 0 * prior_num_alloc, global_thread_idx, r6, r7, r8, r9);
    r10 = -1.00000000000000000e+00;
    r6 = fmaf(r6, r10, r4);
    r6 = r0 * r6;
    r7 = fmaf(r7, r10, r5);
    r7 = r1 * r7;
    ReadIdx2<1024, float, float, float2>(principal_point,
                                         0 * principal_point_num_alloc,
                                         global_thread_idx,
                                         r5,
                                         r4);
    r8 = fmaf(r8, r10, r5);
    r8 = r2 * r8;
    r9 = fmaf(r9, r10, r4);
    r9 = r3 * r9;
    WriteIdx4<1024, float, float, float4>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r6, r7, r8, r9);
    r9 = r0 * r10;
    r9 = r9 * r6;
    r10 = r1 * r10;
    r10 = r10 * r7;
    WriteSum2<float, float>((float*)inout_shared, r9, r10);
  };
  FlushSumShared<2, float>(out_focal_njtr,
                           0 * out_focal_njtr_num_alloc,
                           focal_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = r0 * r0;
    r1 = r1 * r1;
    WriteSum2<float, float>((float*)inout_shared, r0, r1);
  };
  FlushSumShared<2, float>(out_focal_precond_diag,
                           0 * out_focal_precond_diag_num_alloc,
                           focal_indices_loc,
                           (float*)inout_shared);
}

void PinholeSplitIntrinsicsPriorFixedPrincipalPointResJac(
    float* focal,
    unsigned int focal_num_alloc,
    SharedIndex* focal_indices,
    float* prior,
    unsigned int prior_num_alloc,
    float* inv_std,
    unsigned int inv_std_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_focal_njtr,
    unsigned int out_focal_njtr_num_alloc,
    float* const out_focal_precond_diag,
    unsigned int out_focal_precond_diag_num_alloc,
    float* const out_focal_precond_tril,
    unsigned int out_focal_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitIntrinsicsPriorFixedPrincipalPointResJacKernel<<<n_blocks,
                                                               1024>>>(
      focal,
      focal_num_alloc,
      focal_indices,
      prior,
      prior_num_alloc,
      inv_std,
      inv_std_num_alloc,
      principal_point,
      principal_point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_focal_njtr,
      out_focal_njtr_num_alloc,
      out_focal_precond_diag,
      out_focal_precond_diag_num_alloc,
      out_focal_precond_tril,
      out_focal_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar