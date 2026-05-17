#include "kernel_pinhole_intrinsics_random_walk_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeIntrinsicsRandomWalkScoreKernel(double* prev_calib,
                                           unsigned int prev_calib_num_alloc,
                                           SharedIndex* prev_calib_indices,
                                           double* next_calib,
                                           unsigned int next_calib_num_alloc,
                                           SharedIndex* next_calib_indices,
                                           double* inv_std,
                                           unsigned int inv_std_num_alloc,
                                           double* const out_rTr,
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

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10;
  LoadShared<2, double, double>(next_calib,
                                2 * next_calib_num_alloc,
                                next_calib_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        next_calib_indices_loc[threadIdx.x].target,
                        r0,
                        r1);
  };
  __syncthreads();
  LoadShared<2, double, double>(prev_calib,
                                2 * prev_calib_num_alloc,
                                prev_calib_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        prev_calib_indices_loc[threadIdx.x].target,
                        r2,
                        r3);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r4 = -1.00000000000000000e+00;
    r3 = fma(r3, r4, r1);
    r3 = r3 * r3;
    ReadIdx2<1024, double, double, double2>(
        inv_std, 2 * inv_std_num_alloc, global_thread_idx, r1, r5);
    r5 = r5 * r5;
  };
  LoadShared<2, double, double>(next_calib,
                                0 * next_calib_num_alloc,
                                next_calib_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        next_calib_indices_loc[threadIdx.x].target,
                        r6,
                        r7);
  };
  __syncthreads();
  LoadShared<2, double, double>(prev_calib,
                                0 * prev_calib_num_alloc,
                                prev_calib_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        prev_calib_indices_loc[threadIdx.x].target,
                        r8,
                        r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r9 = fma(r9, r4, r7);
    r9 = r9 * r9;
    ReadIdx2<1024, double, double, double2>(
        inv_std, 0 * inv_std_num_alloc, global_thread_idx, r7, r10);
    r10 = r10 * r10;
    r10 = fma(r9, r10, r3 * r5);
    r8 = fma(r8, r4, r6);
    r8 = r8 * r8;
    r7 = r7 * r7;
    r4 = fma(r2, r4, r0);
    r4 = r4 * r4;
    r1 = r1 * r1;
    r10 = fma(r8, r7, r10);
    r10 = fma(r4, r1, r10);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r10);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void PinholeIntrinsicsRandomWalkScore(double* prev_calib,
                                      unsigned int prev_calib_num_alloc,
                                      SharedIndex* prev_calib_indices,
                                      double* next_calib,
                                      unsigned int next_calib_num_alloc,
                                      SharedIndex* next_calib_indices,
                                      double* inv_std,
                                      unsigned int inv_std_num_alloc,
                                      double* const out_rTr,
                                      size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeIntrinsicsRandomWalkScoreKernel<<<n_blocks, 1024>>>(
      prev_calib,
      prev_calib_num_alloc,
      prev_calib_indices,
      next_calib,
      next_calib_num_alloc,
      next_calib_indices,
      inv_std,
      inv_std_num_alloc,
      out_rTr,
      problem_size);
}

}  // namespace caspar