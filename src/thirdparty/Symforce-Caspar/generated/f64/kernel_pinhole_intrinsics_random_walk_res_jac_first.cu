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
        double* prev_calib,
        unsigned int prev_calib_num_alloc,
        SharedIndex* prev_calib_indices,
        double* next_calib,
        unsigned int next_calib_num_alloc,
        SharedIndex* next_calib_indices,
        double* inv_std,
        unsigned int inv_std_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* const out_rTr,
        double* out_prev_calib_jac,
        unsigned int out_prev_calib_jac_num_alloc,
        double* const out_prev_calib_njtr,
        unsigned int out_prev_calib_njtr_num_alloc,
        double* const out_prev_calib_precond_diag,
        unsigned int out_prev_calib_precond_diag_num_alloc,
        double* const out_prev_calib_precond_tril,
        unsigned int out_prev_calib_precond_tril_num_alloc,
        double* out_next_calib_jac,
        unsigned int out_next_calib_jac_num_alloc,
        double* const out_next_calib_njtr,
        unsigned int out_next_calib_njtr_num_alloc,
        double* const out_next_calib_precond_diag,
        unsigned int out_next_calib_precond_diag_num_alloc,
        double* const out_next_calib_precond_tril,
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

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        inv_std, 0 * inv_std_num_alloc, global_thread_idx, r0, r1);
  };
  LoadShared<2, double, double>(next_calib,
                                0 * next_calib_num_alloc,
                                next_calib_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        next_calib_indices_loc[threadIdx.x].target,
                        r2,
                        r3);
  };
  __syncthreads();
  LoadShared<2, double, double>(prev_calib,
                                0 * prev_calib_num_alloc,
                                prev_calib_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        prev_calib_indices_loc[threadIdx.x].target,
                        r4,
                        r5);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r6 = -1.00000000000000000e+00;
    r4 = fma(r4, r6, r2);
    r2 = r0 * r4;
    r5 = fma(r5, r6, r3);
    r3 = r1 * r5;
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    ReadIdx2<1024, double, double, double2>(
        inv_std, 2 * inv_std_num_alloc, global_thread_idx, r3, r2);
  };
  LoadShared<2, double, double>(next_calib,
                                2 * next_calib_num_alloc,
                                next_calib_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        next_calib_indices_loc[threadIdx.x].target,
                        r7,
                        r8);
  };
  __syncthreads();
  LoadShared<2, double, double>(prev_calib,
                                2 * prev_calib_num_alloc,
                                prev_calib_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        prev_calib_indices_loc[threadIdx.x].target,
                        r9,
                        r10);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r9 = fma(r9, r6, r7);
    r7 = r3 * r9;
    r10 = fma(r10, r6, r8);
    r8 = r2 * r10;
    WriteIdx2<1024, double, double, double2>(
        out_res, 2 * out_res_num_alloc, global_thread_idx, r7, r8);
    r8 = r2 * r2;
    r7 = r10 * r8;
    r11 = r1 * r1;
    r12 = r5 * r11;
    r5 = fma(r5, r12, r10 * r7);
    r10 = r0 * r0;
    r13 = r4 * r10;
    r14 = r3 * r3;
    r15 = r9 * r14;
    r5 = fma(r4, r13, r5);
    r5 = fma(r9, r15, r5);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r5);
  if (global_thread_idx < problem_size) {
    r5 = r0 * r6;
    r9 = r1 * r6;
    WriteIdx2<1024, double, double, double2>(out_prev_calib_jac,
                                             0 * out_prev_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r5,
                                             r9);
    r9 = r3 * r6;
    r5 = r2 * r6;
    WriteIdx2<1024, double, double, double2>(out_prev_calib_jac,
                                             2 * out_prev_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r9,
                                             r5);
    WriteSum2<double, double>((double*)inout_shared, r13, r12);
  };
  FlushSumShared<2, double>(out_prev_calib_njtr,
                            0 * out_prev_calib_njtr_num_alloc,
                            prev_calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r15, r7);
  };
  FlushSumShared<2, double>(out_prev_calib_njtr,
                            2 * out_prev_calib_njtr_num_alloc,
                            prev_calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r10, r11);
  };
  FlushSumShared<2, double>(out_prev_calib_precond_diag,
                            0 * out_prev_calib_precond_diag_num_alloc,
                            prev_calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r14, r8);
  };
  FlushSumShared<2, double>(out_prev_calib_precond_diag,
                            2 * out_prev_calib_precond_diag_num_alloc,
                            prev_calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteIdx2<1024, double, double, double2>(out_next_calib_jac,
                                             0 * out_next_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r0,
                                             r1);
    WriteIdx2<1024, double, double, double2>(out_next_calib_jac,
                                             2 * out_next_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r3,
                                             r2);
    r13 = r6 * r13;
    r12 = r6 * r12;
    WriteSum2<double, double>((double*)inout_shared, r13, r12);
  };
  FlushSumShared<2, double>(out_next_calib_njtr,
                            0 * out_next_calib_njtr_num_alloc,
                            next_calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r15 = r6 * r15;
    r7 = r6 * r7;
    WriteSum2<double, double>((double*)inout_shared, r15, r7);
  };
  FlushSumShared<2, double>(out_next_calib_njtr,
                            2 * out_next_calib_njtr_num_alloc,
                            next_calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r10, r11);
  };
  FlushSumShared<2, double>(out_next_calib_precond_diag,
                            0 * out_next_calib_precond_diag_num_alloc,
                            next_calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r14, r8);
  };
  FlushSumShared<2, double>(out_next_calib_precond_diag,
                            2 * out_next_calib_precond_diag_num_alloc,
                            next_calib_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void PinholeIntrinsicsRandomWalkResJacFirst(
    double* prev_calib,
    unsigned int prev_calib_num_alloc,
    SharedIndex* prev_calib_indices,
    double* next_calib,
    unsigned int next_calib_num_alloc,
    SharedIndex* next_calib_indices,
    double* inv_std,
    unsigned int inv_std_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_rTr,
    double* out_prev_calib_jac,
    unsigned int out_prev_calib_jac_num_alloc,
    double* const out_prev_calib_njtr,
    unsigned int out_prev_calib_njtr_num_alloc,
    double* const out_prev_calib_precond_diag,
    unsigned int out_prev_calib_precond_diag_num_alloc,
    double* const out_prev_calib_precond_tril,
    unsigned int out_prev_calib_precond_tril_num_alloc,
    double* out_next_calib_jac,
    unsigned int out_next_calib_jac_num_alloc,
    double* const out_next_calib_njtr,
    unsigned int out_next_calib_njtr_num_alloc,
    double* const out_next_calib_precond_diag,
    unsigned int out_next_calib_precond_diag_num_alloc,
    double* const out_next_calib_precond_tril,
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