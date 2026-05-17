#include "kernel_pinhole_split_intrinsics_prior_fixed_focal_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitIntrinsicsPriorFixedFocalResJacFirstKernel(
        double* principal_point,
        unsigned int principal_point_num_alloc,
        SharedIndex* principal_point_indices,
        double* prior,
        unsigned int prior_num_alloc,
        double* inv_std,
        unsigned int inv_std_num_alloc,
        double* focal,
        unsigned int focal_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* const out_rTr,
        double* const out_principal_point_njtr,
        unsigned int out_principal_point_njtr_num_alloc,
        double* const out_principal_point_precond_diag,
        unsigned int out_principal_point_precond_diag_num_alloc,
        double* const out_principal_point_precond_tril,
        unsigned int out_principal_point_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex principal_point_indices_loc[1024];
  principal_point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? principal_point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        inv_std, 0 * inv_std_num_alloc, global_thread_idx, r0, r1);
    ReadIdx2<1024, double, double, double2>(
        focal, 0 * focal_num_alloc, global_thread_idx, r2, r3);
    ReadIdx2<1024, double, double, double2>(
        prior, 0 * prior_num_alloc, global_thread_idx, r4, r5);
    r6 = -1.00000000000000000e+00;
    r4 = fma(r4, r6, r2);
    r2 = r0 * r4;
    r5 = fma(r5, r6, r3);
    r3 = r1 * r5;
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    ReadIdx2<1024, double, double, double2>(
        inv_std, 2 * inv_std_num_alloc, global_thread_idx, r7, r8);
  };
  LoadShared<2, double, double>(principal_point,
                                0 * principal_point_num_alloc,
                                principal_point_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        principal_point_indices_loc[threadIdx.x].target,
                        r9,
                        r10);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        prior, 2 * prior_num_alloc, global_thread_idx, r11, r12);
    r11 = fma(r11, r6, r9);
    r9 = r7 * r11;
    r12 = fma(r12, r6, r10);
    r10 = r8 * r12;
    WriteIdx2<1024, double, double, double2>(
        out_res, 2 * out_res_num_alloc, global_thread_idx, r9, r10);
    r9 = r7 * r9;
    r10 = r8 * r10;
    r12 = fma(r12, r10, r11 * r9);
    r11 = r1 * r5;
    r12 = fma(r3, r11, r12);
    r3 = r0 * r4;
    r12 = fma(r2, r3, r12);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r12);
  if (global_thread_idx < problem_size) {
    r9 = r6 * r9;
    r10 = r6 * r10;
    WriteSum2<double, double>((double*)inout_shared, r9, r10);
  };
  FlushSumShared<2, double>(out_principal_point_njtr,
                            0 * out_principal_point_njtr_num_alloc,
                            principal_point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = r7 * r7;
    r8 = r8 * r8;
    WriteSum2<double, double>((double*)inout_shared, r7, r8);
  };
  FlushSumShared<2, double>(out_principal_point_precond_diag,
                            0 * out_principal_point_precond_diag_num_alloc,
                            principal_point_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void PinholeSplitIntrinsicsPriorFixedFocalResJacFirst(
    double* principal_point,
    unsigned int principal_point_num_alloc,
    SharedIndex* principal_point_indices,
    double* prior,
    unsigned int prior_num_alloc,
    double* inv_std,
    unsigned int inv_std_num_alloc,
    double* focal,
    unsigned int focal_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_rTr,
    double* const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    double* const out_principal_point_precond_diag,
    unsigned int out_principal_point_precond_diag_num_alloc,
    double* const out_principal_point_precond_tril,
    unsigned int out_principal_point_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitIntrinsicsPriorFixedFocalResJacFirstKernel<<<n_blocks, 1024>>>(
      principal_point,
      principal_point_num_alloc,
      principal_point_indices,
      prior,
      prior_num_alloc,
      inv_std,
      inv_std_num_alloc,
      focal,
      focal_num_alloc,
      out_res,
      out_res_num_alloc,
      out_rTr,
      out_principal_point_njtr,
      out_principal_point_njtr_num_alloc,
      out_principal_point_precond_diag,
      out_principal_point_precond_diag_num_alloc,
      out_principal_point_precond_tril,
      out_principal_point_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar