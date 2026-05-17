#include "kernel_pinhole_split_intrinsics_random_walk_fixed_next_focal_fixed_next_principal_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitIntrinsicsRandomWalkFixedNextFocalFixedNextPrincipalPointResJacKernel(
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
        float* out_res,
        unsigned int out_res_num_alloc,
        float* out_prev_focal_jac,
        unsigned int out_prev_focal_jac_num_alloc,
        float* const out_prev_focal_njtr,
        unsigned int out_prev_focal_njtr_num_alloc,
        float* const out_prev_focal_precond_diag,
        unsigned int out_prev_focal_precond_diag_num_alloc,
        float* const out_prev_focal_precond_tril,
        unsigned int out_prev_focal_precond_tril_num_alloc,
        float* out_prev_principal_point_jac,
        unsigned int out_prev_principal_point_jac_num_alloc,
        float* const out_prev_principal_point_njtr,
        unsigned int out_prev_principal_point_njtr_num_alloc,
        float* const out_prev_principal_point_precond_diag,
        unsigned int out_prev_principal_point_precond_diag_num_alloc,
        float* const out_prev_principal_point_precond_tril,
        unsigned int out_prev_principal_point_precond_tril_num_alloc,
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

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(
        inv_std, 0 * inv_std_num_alloc, global_thread_idx, r0, r1, r2, r3);
    ReadIdx2<1024, float, float, float2>(
        next_focal, 0 * next_focal_num_alloc, global_thread_idx, r4, r5);
  };
  LoadShared<2, float, float>(prev_focal,
                              0 * prev_focal_num_alloc,
                              prev_focal_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared,
                       prev_focal_indices_loc[threadIdx.x].target,
                       r6,
                       r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r8 = -1.00000000000000000e+00;
    r6 = fmaf(r6, r8, r4);
    r6 = r0 * r6;
    r7 = fmaf(r7, r8, r5);
    r7 = r1 * r7;
    ReadIdx2<1024, float, float, float2>(next_principal_point,
                                         0 * next_principal_point_num_alloc,
                                         global_thread_idx,
                                         r5,
                                         r4);
  };
  LoadShared<2, float, float>(prev_principal_point,
                              0 * prev_principal_point_num_alloc,
                              prev_principal_point_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared,
                       prev_principal_point_indices_loc[threadIdx.x].target,
                       r9,
                       r10);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r9 = fmaf(r9, r8, r5);
    r9 = r2 * r9;
    r10 = fmaf(r10, r8, r4);
    r10 = r3 * r10;
    WriteIdx4<1024, float, float, float4>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r6, r7, r9, r10);
    r4 = r0 * r8;
    r5 = r1 * r8;
    WriteIdx2<1024, float, float, float2>(out_prev_focal_jac,
                                          0 * out_prev_focal_jac_num_alloc,
                                          global_thread_idx,
                                          r4,
                                          r5);
    r6 = r0 * r6;
    r7 = r1 * r7;
    WriteSum2<float, float>((float*)inout_shared, r6, r7);
  };
  FlushSumShared<2, float>(out_prev_focal_njtr,
                           0 * out_prev_focal_njtr_num_alloc,
                           prev_focal_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = r0 * r0;
    r1 = r1 * r1;
    WriteSum2<float, float>((float*)inout_shared, r0, r1);
  };
  FlushSumShared<2, float>(out_prev_focal_precond_diag,
                           0 * out_prev_focal_precond_diag_num_alloc,
                           prev_focal_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r2 * r8;
    r8 = r3 * r8;
    WriteIdx2<1024, float, float, float2>(
        out_prev_principal_point_jac,
        0 * out_prev_principal_point_jac_num_alloc,
        global_thread_idx,
        r1,
        r8);
    r9 = r2 * r9;
    r10 = r3 * r10;
    WriteSum2<float, float>((float*)inout_shared, r9, r10);
  };
  FlushSumShared<2, float>(out_prev_principal_point_njtr,
                           0 * out_prev_principal_point_njtr_num_alloc,
                           prev_principal_point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r2 = r2 * r2;
    r3 = r3 * r3;
    WriteSum2<float, float>((float*)inout_shared, r2, r3);
  };
  FlushSumShared<2, float>(out_prev_principal_point_precond_diag,
                           0 * out_prev_principal_point_precond_diag_num_alloc,
                           prev_principal_point_indices_loc,
                           (float*)inout_shared);
}

void PinholeSplitIntrinsicsRandomWalkFixedNextFocalFixedNextPrincipalPointResJac(
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
    float* out_res,
    unsigned int out_res_num_alloc,
    float* out_prev_focal_jac,
    unsigned int out_prev_focal_jac_num_alloc,
    float* const out_prev_focal_njtr,
    unsigned int out_prev_focal_njtr_num_alloc,
    float* const out_prev_focal_precond_diag,
    unsigned int out_prev_focal_precond_diag_num_alloc,
    float* const out_prev_focal_precond_tril,
    unsigned int out_prev_focal_precond_tril_num_alloc,
    float* out_prev_principal_point_jac,
    unsigned int out_prev_principal_point_jac_num_alloc,
    float* const out_prev_principal_point_njtr,
    unsigned int out_prev_principal_point_njtr_num_alloc,
    float* const out_prev_principal_point_precond_diag,
    unsigned int out_prev_principal_point_precond_diag_num_alloc,
    float* const out_prev_principal_point_precond_tril,
    unsigned int out_prev_principal_point_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitIntrinsicsRandomWalkFixedNextFocalFixedNextPrincipalPointResJacKernel<<<
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
              out_res,
              out_res_num_alloc,
              out_prev_focal_jac,
              out_prev_focal_jac_num_alloc,
              out_prev_focal_njtr,
              out_prev_focal_njtr_num_alloc,
              out_prev_focal_precond_diag,
              out_prev_focal_precond_diag_num_alloc,
              out_prev_focal_precond_tril,
              out_prev_focal_precond_tril_num_alloc,
              out_prev_principal_point_jac,
              out_prev_principal_point_jac_num_alloc,
              out_prev_principal_point_njtr,
              out_prev_principal_point_njtr_num_alloc,
              out_prev_principal_point_precond_diag,
              out_prev_principal_point_precond_diag_num_alloc,
              out_prev_principal_point_precond_tril,
              out_prev_principal_point_precond_tril_num_alloc,
              problem_size);
}

}  // namespace caspar