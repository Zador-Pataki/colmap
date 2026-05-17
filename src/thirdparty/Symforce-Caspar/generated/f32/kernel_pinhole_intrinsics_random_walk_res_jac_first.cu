#include "kernel_pinhole_intrinsics_random_walk_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeIntrinsicsRandomWalkResJacFirstKernel(
        float* prev_calib,
        unsigned int prev_calib_num_alloc,
        SharedIndex* prev_calib_indices,
        float* next_calib,
        unsigned int next_calib_num_alloc,
        SharedIndex* next_calib_indices,
        float* inv_std,
        unsigned int inv_std_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_rTr,
        float* out_prev_calib_jac,
        unsigned int out_prev_calib_jac_num_alloc,
        float* const out_prev_calib_njtr,
        unsigned int out_prev_calib_njtr_num_alloc,
        float* const out_prev_calib_precond_diag,
        unsigned int out_prev_calib_precond_diag_num_alloc,
        float* const out_prev_calib_precond_tril,
        unsigned int out_prev_calib_precond_tril_num_alloc,
        float* out_next_calib_jac,
        unsigned int out_next_calib_jac_num_alloc,
        float* const out_next_calib_njtr,
        unsigned int out_next_calib_njtr_num_alloc,
        float* const out_next_calib_precond_diag,
        unsigned int out_next_calib_precond_diag_num_alloc,
        float* const out_next_calib_precond_tril,
        unsigned int out_next_calib_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex prev_calib_indices_loc[1024];
  prev_calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? prev_calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex next_calib_indices_loc[1024];
  next_calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? next_calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(
        inv_std, 0 * inv_std_num_alloc, global_thread_idx, r0, r1, r2, r3);
  };
  LoadShared<4, float, float>(next_calib,
                              0 * next_calib_num_alloc,
                              next_calib_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       next_calib_indices_loc[threadIdx.x].target,
                       r4,
                       r5,
                       r6,
                       r7);
  };
  __syncthreads();
  LoadShared<4, float, float>(prev_calib,
                              0 * prev_calib_num_alloc,
                              prev_calib_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       prev_calib_indices_loc[threadIdx.x].target,
                       r8,
                       r9,
                       r10,
                       r11);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r12 = -1.00000000000000000e+00;
    r8 = fmaf(r8, r12, r4);
    r4 = r0 * r8;
    r9 = fmaf(r9, r12, r5);
    r5 = r1 * r9;
    r10 = fmaf(r10, r12, r6);
    r6 = r2 * r10;
    r11 = fmaf(r11, r12, r7);
    r7 = r3 * r11;
    WriteIdx4<1024, float, float, float4>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r4, r5, r6, r7);
    r7 = r3 * r3;
    r6 = r11 * r7;
    r5 = r1 * r1;
    r4 = r9 * r5;
    r9 = fmaf(r9, r4, r11 * r6);
    r11 = r0 * r0;
    r13 = r8 * r11;
    r14 = r2 * r2;
    r15 = r10 * r14;
    r9 = fmaf(r8, r13, r9);
    r9 = fmaf(r10, r15, r9);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r9);
  if (global_thread_idx < problem_size) {
    r9 = r0 * r12;
    r10 = r1 * r12;
    r8 = r2 * r12;
    r16 = r3 * r12;
    WriteIdx4<1024, float, float, float4>(out_prev_calib_jac,
                                          0 * out_prev_calib_jac_num_alloc,
                                          global_thread_idx,
                                          r9,
                                          r10,
                                          r8,
                                          r16);
    WriteSum4<float, float>((float*)inout_shared, r13, r4, r15, r6);
  };
  FlushSumShared<4, float>(out_prev_calib_njtr,
                           0 * out_prev_calib_njtr_num_alloc,
                           prev_calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum4<float, float>((float*)inout_shared, r11, r5, r14, r7);
  };
  FlushSumShared<4, float>(out_prev_calib_precond_diag,
                           0 * out_prev_calib_precond_diag_num_alloc,
                           prev_calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteIdx4<1024, float, float, float4>(out_next_calib_jac,
                                          0 * out_next_calib_jac_num_alloc,
                                          global_thread_idx,
                                          r0,
                                          r1,
                                          r2,
                                          r3);
    r13 = r12 * r13;
    r4 = r12 * r4;
    r15 = r12 * r15;
    r6 = r12 * r6;
    WriteSum4<float, float>((float*)inout_shared, r13, r4, r15, r6);
  };
  FlushSumShared<4, float>(out_next_calib_njtr,
                           0 * out_next_calib_njtr_num_alloc,
                           next_calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum4<float, float>((float*)inout_shared, r11, r5, r14, r7);
  };
  FlushSumShared<4, float>(out_next_calib_precond_diag,
                           0 * out_next_calib_precond_diag_num_alloc,
                           next_calib_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void PinholeIntrinsicsRandomWalkResJacFirst(
    float* prev_calib,
    unsigned int prev_calib_num_alloc,
    SharedIndex* prev_calib_indices,
    float* next_calib,
    unsigned int next_calib_num_alloc,
    SharedIndex* next_calib_indices,
    float* inv_std,
    unsigned int inv_std_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
    float* out_prev_calib_jac,
    unsigned int out_prev_calib_jac_num_alloc,
    float* const out_prev_calib_njtr,
    unsigned int out_prev_calib_njtr_num_alloc,
    float* const out_prev_calib_precond_diag,
    unsigned int out_prev_calib_precond_diag_num_alloc,
    float* const out_prev_calib_precond_tril,
    unsigned int out_prev_calib_precond_tril_num_alloc,
    float* out_next_calib_jac,
    unsigned int out_next_calib_jac_num_alloc,
    float* const out_next_calib_njtr,
    unsigned int out_next_calib_njtr_num_alloc,
    float* const out_next_calib_precond_diag,
    unsigned int out_next_calib_precond_diag_num_alloc,
    float* const out_next_calib_precond_tril,
    unsigned int out_next_calib_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeIntrinsicsRandomWalkResJacFirstKernel<<<n_blocks, 1024>>>(
      prev_calib,
      prev_calib_num_alloc,
      prev_calib_indices,
      next_calib,
      next_calib_num_alloc,
      next_calib_indices,
      inv_std,
      inv_std_num_alloc,
      out_res,
      out_res_num_alloc,
      out_rTr,
      out_prev_calib_jac,
      out_prev_calib_jac_num_alloc,
      out_prev_calib_njtr,
      out_prev_calib_njtr_num_alloc,
      out_prev_calib_precond_diag,
      out_prev_calib_precond_diag_num_alloc,
      out_prev_calib_precond_tril,
      out_prev_calib_precond_tril_num_alloc,
      out_next_calib_jac,
      out_next_calib_jac_num_alloc,
      out_next_calib_njtr,
      out_next_calib_njtr_num_alloc,
      out_next_calib_precond_diag,
      out_next_calib_precond_diag_num_alloc,
      out_next_calib_precond_tril,
      out_next_calib_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar