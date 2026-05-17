#include "kernel_pinhole_intrinsics_random_walk_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeIntrinsicsRandomWalkJtjnjtrDirectKernel(
        double* prev_calib_njtr,
        unsigned int prev_calib_njtr_num_alloc,
        SharedIndex* prev_calib_njtr_indices,
        double* prev_calib_jac,
        unsigned int prev_calib_jac_num_alloc,
        double* next_calib_njtr,
        unsigned int next_calib_njtr_num_alloc,
        SharedIndex* next_calib_njtr_indices,
        double* next_calib_jac,
        unsigned int next_calib_jac_num_alloc,
        double* const out_prev_calib_njtr,
        unsigned int out_prev_calib_njtr_num_alloc,
        double* const out_next_calib_njtr,
        unsigned int out_next_calib_njtr_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex prev_calib_njtr_indices_loc[1024];
  prev_calib_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? prev_calib_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex next_calib_njtr_indices_loc[1024];
  next_calib_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? next_calib_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7;
  LoadShared<2, double, double>(next_calib_njtr,
                                0 * next_calib_njtr_num_alloc,
                                next_calib_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        next_calib_njtr_indices_loc[threadIdx.x].target,
                        r0,
                        r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(prev_calib_jac,
                                            0 * prev_calib_jac_num_alloc,
                                            global_thread_idx,
                                            r2,
                                            r3);
    ReadIdx2<1024, double, double, double2>(next_calib_jac,
                                            0 * next_calib_jac_num_alloc,
                                            global_thread_idx,
                                            r4,
                                            r5);
    r4 = r2 * r4;
    r0 = r0 * r4;
    r5 = r3 * r5;
    r1 = r1 * r5;
    WriteSum2<double, double>((double*)inout_shared, r0, r1);
  };
  FlushSumShared<2, double>(out_prev_calib_njtr,
                            0 * out_prev_calib_njtr_num_alloc,
                            prev_calib_njtr_indices_loc,
                            (double*)inout_shared);
  LoadShared<2, double, double>(next_calib_njtr,
                                2 * next_calib_njtr_num_alloc,
                                next_calib_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        next_calib_njtr_indices_loc[threadIdx.x].target,
                        r1,
                        r0);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(prev_calib_jac,
                                            2 * prev_calib_jac_num_alloc,
                                            global_thread_idx,
                                            r3,
                                            r2);
    ReadIdx2<1024, double, double, double2>(next_calib_jac,
                                            2 * next_calib_jac_num_alloc,
                                            global_thread_idx,
                                            r6,
                                            r7);
    r6 = r3 * r6;
    r1 = r1 * r6;
    r7 = r2 * r7;
    r0 = r0 * r7;
    WriteSum2<double, double>((double*)inout_shared, r1, r0);
  };
  FlushSumShared<2, double>(out_prev_calib_njtr,
                            2 * out_prev_calib_njtr_num_alloc,
                            prev_calib_njtr_indices_loc,
                            (double*)inout_shared);
  LoadShared<2, double, double>(prev_calib_njtr,
                                0 * prev_calib_njtr_num_alloc,
                                prev_calib_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        prev_calib_njtr_indices_loc[threadIdx.x].target,
                        r0,
                        r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r4 = r0 * r4;
    r5 = r1 * r5;
    WriteSum2<double, double>((double*)inout_shared, r4, r5);
  };
  FlushSumShared<2, double>(out_next_calib_njtr,
                            0 * out_next_calib_njtr_num_alloc,
                            next_calib_njtr_indices_loc,
                            (double*)inout_shared);
  LoadShared<2, double, double>(prev_calib_njtr,
                                2 * prev_calib_njtr_num_alloc,
                                prev_calib_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        prev_calib_njtr_indices_loc[threadIdx.x].target,
                        r5,
                        r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r6 = r5 * r6;
    r7 = r4 * r7;
    WriteSum2<double, double>((double*)inout_shared, r6, r7);
  };
  FlushSumShared<2, double>(out_next_calib_njtr,
                            2 * out_next_calib_njtr_num_alloc,
                            next_calib_njtr_indices_loc,
                            (double*)inout_shared);
}

void PinholeIntrinsicsRandomWalkJtjnjtrDirect(
    double* prev_calib_njtr,
    unsigned int prev_calib_njtr_num_alloc,
    SharedIndex* prev_calib_njtr_indices,
    double* prev_calib_jac,
    unsigned int prev_calib_jac_num_alloc,
    double* next_calib_njtr,
    unsigned int next_calib_njtr_num_alloc,
    SharedIndex* next_calib_njtr_indices,
    double* next_calib_jac,
    unsigned int next_calib_jac_num_alloc,
    double* const out_prev_calib_njtr,
    unsigned int out_prev_calib_njtr_num_alloc,
    double* const out_next_calib_njtr,
    unsigned int out_next_calib_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeIntrinsicsRandomWalkJtjnjtrDirectKernel<<<n_blocks, 1024>>>(
      prev_calib_njtr,
      prev_calib_njtr_num_alloc,
      prev_calib_njtr_indices,
      prev_calib_jac,
      prev_calib_jac_num_alloc,
      next_calib_njtr,
      next_calib_njtr_num_alloc,
      next_calib_njtr_indices,
      next_calib_jac,
      next_calib_jac_num_alloc,
      out_prev_calib_njtr,
      out_prev_calib_njtr_num_alloc,
      out_next_calib_njtr,
      out_next_calib_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar