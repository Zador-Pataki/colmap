#include "kernel_DepthScale_update_r_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    DepthScaleUpdateRFirstKernel(double* DepthScale_r_k,
                                 unsigned int DepthScale_r_k_num_alloc,
                                 double* DepthScale_w,
                                 unsigned int DepthScale_w_num_alloc,
                                 const double* const negalpha,
                                 double* out_DepthScale_r_kp1,
                                 unsigned int out_DepthScale_r_kp1_num_alloc,
                                 double* const out_DepthScale_r_0_norm2_tot,
                                 double* const out_DepthScale_r_kp1_norm2_tot,
                                 size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ double out_DepthScale_r_0_norm2_tot_local[1];

  __shared__ double out_DepthScale_r_kp1_norm2_tot_local[1];

  double r0, r1, r2;

  if (global_thread_idx < problem_size) {
    ReadIdx1<1024, double, double, double>(
        DepthScale_r_k, 0 * DepthScale_r_k_num_alloc, global_thread_idx, r0);
    ReadIdx1<1024, double, double, double>(
        DepthScale_w, 0 * DepthScale_w_num_alloc, global_thread_idx, r1);
  };
  LoadUnique<1, double, double>(negalpha, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, 0, r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r2 = fma(r1, r2, r0);
    WriteIdx1<1024, double, double, double>(out_DepthScale_r_kp1,
                                            0 * out_DepthScale_r_kp1_num_alloc,
                                            global_thread_idx,
                                            r2);
    r0 = r0 * r0;
  };
  SumStore<double>(out_DepthScale_r_0_norm2_tot_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r0);
  if (global_thread_idx < problem_size) {
    r2 = r2 * r2;
  };
  SumStore<double>(out_DepthScale_r_kp1_norm2_tot_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r2);
  SumFlushFinal<double>(
      out_DepthScale_r_0_norm2_tot_local, out_DepthScale_r_0_norm2_tot, 1);
  SumFlushFinal<double>(
      out_DepthScale_r_kp1_norm2_tot_local, out_DepthScale_r_kp1_norm2_tot, 1);
}

void DepthScaleUpdateRFirst(double* DepthScale_r_k,
                            unsigned int DepthScale_r_k_num_alloc,
                            double* DepthScale_w,
                            unsigned int DepthScale_w_num_alloc,
                            const double* const negalpha,
                            double* out_DepthScale_r_kp1,
                            unsigned int out_DepthScale_r_kp1_num_alloc,
                            double* const out_DepthScale_r_0_norm2_tot,
                            double* const out_DepthScale_r_kp1_norm2_tot,
                            size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  DepthScaleUpdateRFirstKernel<<<n_blocks, 1024>>>(
      DepthScale_r_k,
      DepthScale_r_k_num_alloc,
      DepthScale_w,
      DepthScale_w_num_alloc,
      negalpha,
      out_DepthScale_r_kp1,
      out_DepthScale_r_kp1_num_alloc,
      out_DepthScale_r_0_norm2_tot,
      out_DepthScale_r_kp1_norm2_tot,
      problem_size);
}

}  // namespace caspar