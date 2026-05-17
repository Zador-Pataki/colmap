#include "kernel_pinhole_split_intrinsics_random_walk_fixed_next_principal_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitIntrinsicsRandomWalkFixedNextPrincipalPointResJacKernel(
        double* prev_focal,
        unsigned int prev_focal_num_alloc,
        SharedIndex* prev_focal_indices,
        double* prev_principal_point,
        unsigned int prev_principal_point_num_alloc,
        SharedIndex* prev_principal_point_indices,
        double* next_focal,
        unsigned int next_focal_num_alloc,
        SharedIndex* next_focal_indices,
        double* inv_std,
        unsigned int inv_std_num_alloc,
        double* next_principal_point,
        unsigned int next_principal_point_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* out_prev_focal_jac,
        unsigned int out_prev_focal_jac_num_alloc,
        double* const out_prev_focal_njtr,
        unsigned int out_prev_focal_njtr_num_alloc,
        double* const out_prev_focal_precond_diag,
        unsigned int out_prev_focal_precond_diag_num_alloc,
        double* const out_prev_focal_precond_tril,
        unsigned int out_prev_focal_precond_tril_num_alloc,
        double* out_prev_principal_point_jac,
        unsigned int out_prev_principal_point_jac_num_alloc,
        double* const out_prev_principal_point_njtr,
        unsigned int out_prev_principal_point_njtr_num_alloc,
        double* const out_prev_principal_point_precond_diag,
        unsigned int out_prev_principal_point_precond_diag_num_alloc,
        double* const out_prev_principal_point_precond_tril,
        unsigned int out_prev_principal_point_precond_tril_num_alloc,
        double* out_next_focal_jac,
        unsigned int out_next_focal_jac_num_alloc,
        double* const out_next_focal_njtr,
        unsigned int out_next_focal_njtr_num_alloc,
        double* const out_next_focal_precond_diag,
        unsigned int out_next_focal_precond_diag_num_alloc,
        double* const out_next_focal_precond_tril,
        unsigned int out_next_focal_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

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
  __shared__ SharedIndex next_focal_indices_loc[1024];
  next_focal_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? next_focal_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        inv_std, 0 * inv_std_num_alloc, global_thread_idx, r0, r1);
  };
  LoadShared<2, double, double>(next_focal,
                                0 * next_focal_num_alloc,
                                next_focal_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        next_focal_indices_loc[threadIdx.x].target,
                        r2,
                        r3);
  };
  __syncthreads();
  LoadShared<2, double, double>(prev_focal,
                                0 * prev_focal_num_alloc,
                                prev_focal_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        prev_focal_indices_loc[threadIdx.x].target,
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
    ReadIdx2<1024, double, double, double2>(next_principal_point,
                                            0 * next_principal_point_num_alloc,
                                            global_thread_idx,
                                            r7,
                                            r8);
  };
  LoadShared<2, double, double>(prev_principal_point,
                                0 * prev_principal_point_num_alloc,
                                prev_principal_point_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        prev_principal_point_indices_loc[threadIdx.x].target,
                        r9,
                        r10);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r9 = fma(r9, r6, r7);
    r9 = r3 * r9;
    r10 = fma(r10, r6, r8);
    r10 = r2 * r10;
    WriteIdx2<1024, double, double, double2>(
        out_res, 2 * out_res_num_alloc, global_thread_idx, r9, r10);
    r8 = r0 * r6;
    r7 = r1 * r6;
    WriteIdx2<1024, double, double, double2>(out_prev_focal_jac,
                                             0 * out_prev_focal_jac_num_alloc,
                                             global_thread_idx,
                                             r8,
                                             r7);
    r7 = r0 * r0;
    r4 = r4 * r7;
    r8 = r1 * r1;
    r5 = r5 * r8;
    WriteSum2<double, double>((double*)inout_shared, r4, r5);
  };
  FlushSumShared<2, double>(out_prev_focal_njtr,
                            0 * out_prev_focal_njtr_num_alloc,
                            prev_focal_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r7, r8);
  };
  FlushSumShared<2, double>(out_prev_focal_precond_diag,
                            0 * out_prev_focal_precond_diag_num_alloc,
                            prev_focal_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = r3 * r6;
    r12 = r2 * r6;
    WriteIdx2<1024, double, double, double2>(
        out_prev_principal_point_jac,
        0 * out_prev_principal_point_jac_num_alloc,
        global_thread_idx,
        r11,
        r12);
    r9 = r3 * r9;
    r10 = r2 * r10;
    WriteSum2<double, double>((double*)inout_shared, r9, r10);
  };
  FlushSumShared<2, double>(out_prev_principal_point_njtr,
                            0 * out_prev_principal_point_njtr_num_alloc,
                            prev_principal_point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r3 = r3 * r3;
    r2 = r2 * r2;
    WriteSum2<double, double>((double*)inout_shared, r3, r2);
  };
  FlushSumShared<2, double>(out_prev_principal_point_precond_diag,
                            0 * out_prev_principal_point_precond_diag_num_alloc,
                            prev_principal_point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteIdx2<1024, double, double, double2>(out_next_focal_jac,
                                             0 * out_next_focal_jac_num_alloc,
                                             global_thread_idx,
                                             r0,
                                             r1);
    r4 = r6 * r4;
    r5 = r6 * r5;
    WriteSum2<double, double>((double*)inout_shared, r4, r5);
  };
  FlushSumShared<2, double>(out_next_focal_njtr,
                            0 * out_next_focal_njtr_num_alloc,
                            next_focal_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r7, r8);
  };
  FlushSumShared<2, double>(out_next_focal_precond_diag,
                            0 * out_next_focal_precond_diag_num_alloc,
                            next_focal_indices_loc,
                            (double*)inout_shared);
}

void PinholeSplitIntrinsicsRandomWalkFixedNextPrincipalPointResJac(
    double* prev_focal,
    unsigned int prev_focal_num_alloc,
    SharedIndex* prev_focal_indices,
    double* prev_principal_point,
    unsigned int prev_principal_point_num_alloc,
    SharedIndex* prev_principal_point_indices,
    double* next_focal,
    unsigned int next_focal_num_alloc,
    SharedIndex* next_focal_indices,
    double* inv_std,
    unsigned int inv_std_num_alloc,
    double* next_principal_point,
    unsigned int next_principal_point_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* out_prev_focal_jac,
    unsigned int out_prev_focal_jac_num_alloc,
    double* const out_prev_focal_njtr,
    unsigned int out_prev_focal_njtr_num_alloc,
    double* const out_prev_focal_precond_diag,
    unsigned int out_prev_focal_precond_diag_num_alloc,
    double* const out_prev_focal_precond_tril,
    unsigned int out_prev_focal_precond_tril_num_alloc,
    double* out_prev_principal_point_jac,
    unsigned int out_prev_principal_point_jac_num_alloc,
    double* const out_prev_principal_point_njtr,
    unsigned int out_prev_principal_point_njtr_num_alloc,
    double* const out_prev_principal_point_precond_diag,
    unsigned int out_prev_principal_point_precond_diag_num_alloc,
    double* const out_prev_principal_point_precond_tril,
    unsigned int out_prev_principal_point_precond_tril_num_alloc,
    double* out_next_focal_jac,
    unsigned int out_next_focal_jac_num_alloc,
    double* const out_next_focal_njtr,
    unsigned int out_next_focal_njtr_num_alloc,
    double* const out_next_focal_precond_diag,
    unsigned int out_next_focal_precond_diag_num_alloc,
    double* const out_next_focal_precond_tril,
    unsigned int out_next_focal_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitIntrinsicsRandomWalkFixedNextPrincipalPointResJacKernel<<<
      n_blocks,
      1024>>>(prev_focal,
              prev_focal_num_alloc,
              prev_focal_indices,
              prev_principal_point,
              prev_principal_point_num_alloc,
              prev_principal_point_indices,
              next_focal,
              next_focal_num_alloc,
              next_focal_indices,
              inv_std,
              inv_std_num_alloc,
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
              out_next_focal_jac,
              out_next_focal_jac_num_alloc,
              out_next_focal_njtr,
              out_next_focal_njtr_num_alloc,
              out_next_focal_precond_diag,
              out_next_focal_precond_diag_num_alloc,
              out_next_focal_precond_tril,
              out_next_focal_precond_tril_num_alloc,
              problem_size);
}

}  // namespace caspar