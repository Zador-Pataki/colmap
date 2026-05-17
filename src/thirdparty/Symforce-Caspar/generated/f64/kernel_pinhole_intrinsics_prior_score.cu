#include "kernel_pinhole_intrinsics_prior_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeIntrinsicsPriorScoreKernel(double* calib,
                                      unsigned int calib_num_alloc,
                                      SharedIndex* calib_indices,
                                      double* prior,
                                      unsigned int prior_num_alloc,
                                      double* inv_std,
                                      unsigned int inv_std_num_alloc,
                                      double* const out_rTr,
                                      size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex calib_indices_loc[1024];
  calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        inv_std, 2 * inv_std_num_alloc, global_thread_idx, r0, r1);
    r0 = r0 * r0;
  };
  LoadShared<2, double, double>(
      calib, 2 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r2, r3);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        prior, 2 * prior_num_alloc, global_thread_idx, r4, r5);
    r6 = -1.00000000000000000e+00;
    r4 = fma(r4, r6, r2);
    r4 = r4 * r4;
    r1 = r1 * r1;
    r5 = fma(r5, r6, r3);
    r5 = r5 * r5;
    r5 = fma(r1, r5, r0 * r4);
    ReadIdx2<1024, double, double, double2>(
        inv_std, 0 * inv_std_num_alloc, global_thread_idx, r1, r4);
    r1 = r1 * r1;
  };
  LoadShared<2, double, double>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r0, r3);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        prior, 0 * prior_num_alloc, global_thread_idx, r2, r7);
    r2 = fma(r2, r6, r0);
    r2 = r2 * r2;
    r4 = r4 * r4;
    r6 = fma(r7, r6, r3);
    r6 = r6 * r6;
    r5 = fma(r1, r2, r5);
    r5 = fma(r4, r6, r5);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r5);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void PinholeIntrinsicsPriorScore(double* calib,
                                 unsigned int calib_num_alloc,
                                 SharedIndex* calib_indices,
                                 double* prior,
                                 unsigned int prior_num_alloc,
                                 double* inv_std,
                                 unsigned int inv_std_num_alloc,
                                 double* const out_rTr,
                                 size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeIntrinsicsPriorScoreKernel<<<n_blocks, 1024>>>(calib,
                                                        calib_num_alloc,
                                                        calib_indices,
                                                        prior,
                                                        prior_num_alloc,
                                                        inv_std,
                                                        inv_std_num_alloc,
                                                        out_rTr,
                                                        problem_size);
}

}  // namespace caspar