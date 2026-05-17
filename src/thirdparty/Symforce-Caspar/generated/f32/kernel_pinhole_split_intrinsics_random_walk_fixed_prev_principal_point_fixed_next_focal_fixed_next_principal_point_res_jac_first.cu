#include "kernel_pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_focal_fixed_next_principal_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalFixedNextPrincipalPointResJacFirstKernel(
        float* prev_focal,
        unsigned int prev_focal_num_alloc,
        SharedIndex* prev_focal_indices,
        float* inv_std,
        unsigned int inv_std_num_alloc,
        float* prev_principal_point,
        unsigned int prev_principal_point_num_alloc,
        float* next_focal,
        unsigned int next_focal_num_alloc,
        float* next_principal_point,
        unsigned int next_principal_point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_rTr,
        float* const out_prev_focal_njtr,
        unsigned int out_prev_focal_njtr_num_alloc,
        float* const out_prev_focal_precond_diag,
        unsigned int out_prev_focal_precond_diag_num_alloc,
        float* const out_prev_focal_precond_tril,
        unsigned int out_prev_focal_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ SharedIndex prev_focal_indices_loc[1024];
  prev_focal_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? prev_focal_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12;

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
    r4 = r0 * r6;
    r7 = fmaf(r7, r8, r5);
    r5 = r1 * r7;
    ReadIdx2<1024, float, float, float2>(next_principal_point,
                                         0 * next_principal_point_num_alloc,
                                         global_thread_idx,
                                         r9,
                                         r10);
    ReadIdx2<1024, float, float, float2>(prev_principal_point,
                                         0 * prev_principal_point_num_alloc,
                                         global_thread_idx,
                                         r11,
                                         r12);
    r11 = fmaf(r11, r8, r9);
    r9 = r2 * r11;
    r8 = fmaf(r12, r8, r10);
    r12 = r3 * r8;
    WriteIdx4<1024, float, float, float4>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r4, r5, r9, r12);
    r5 = r2 * r11;
    r4 = r3 * r8;
    r4 = fmaf(r12, r4, r9 * r5);
    r0 = r0 * r0;
    r5 = r6 * r0;
    r1 = r1 * r1;
    r12 = r7 * r1;
    r4 = fmaf(r6, r5, r4);
    r4 = fmaf(r7, r12, r4);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r4);
  if (global_thread_idx < problem_size) {
    WriteSum2<float, float>((float*)inout_shared, r5, r12);
  };
  FlushSumShared<2, float>(out_prev_focal_njtr,
                           0 * out_prev_focal_njtr_num_alloc,
                           prev_focal_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<float, float>((float*)inout_shared, r0, r1);
  };
  FlushSumShared<2, float>(out_prev_focal_precond_diag,
                           0 * out_prev_focal_precond_diag_num_alloc,
                           prev_focal_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalFixedNextPrincipalPointResJacFirst(
    float* prev_focal,
    unsigned int prev_focal_num_alloc,
    SharedIndex* prev_focal_indices,
    float* inv_std,
    unsigned int inv_std_num_alloc,
    float* prev_principal_point,
    unsigned int prev_principal_point_num_alloc,
    float* next_focal,
    unsigned int next_focal_num_alloc,
    float* next_principal_point,
    unsigned int next_principal_point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
    float* const out_prev_focal_njtr,
    unsigned int out_prev_focal_njtr_num_alloc,
    float* const out_prev_focal_precond_diag,
    unsigned int out_prev_focal_precond_diag_num_alloc,
    float* const out_prev_focal_precond_tril,
    unsigned int out_prev_focal_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalFixedNextPrincipalPointResJacFirstKernel<<<
      n_blocks,
      1024>>>(prev_focal,
              prev_focal_num_alloc,
              prev_focal_indices,
              inv_std,
              inv_std_num_alloc,
              prev_principal_point,
              prev_principal_point_num_alloc,
              next_focal,
              next_focal_num_alloc,
              next_principal_point,
              next_principal_point_num_alloc,
              out_res,
              out_res_num_alloc,
              out_rTr,
              out_prev_focal_njtr,
              out_prev_focal_njtr_num_alloc,
              out_prev_focal_precond_diag,
              out_prev_focal_precond_diag_num_alloc,
              out_prev_focal_precond_tril,
              out_prev_focal_precond_tril_num_alloc,
              problem_size);
}

}  // namespace caspar