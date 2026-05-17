#include "kernel_DepthScale_alpha_denominator_or_beta_numerator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    DepthScaleAlphaDenominatorOrBetaNumeratorKernel(
        double* DepthScale_p_kp1,
        unsigned int DepthScale_p_kp1_num_alloc,
        double* DepthScale_w,
        unsigned int DepthScale_w_num_alloc,
        double* const DepthScale_out,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[256];

  __shared__ double DepthScale_out_local[1];

  double r0, r1;

  if (global_thread_idx < problem_size) {
    ReadIdx1<1024, double, double, double>(DepthScale_p_kp1,
                                           0 * DepthScale_p_kp1_num_alloc,
                                           global_thread_idx,
                                           r0);
    ReadIdx1<1024, double, double, double>(
        DepthScale_w, 0 * DepthScale_w_num_alloc, global_thread_idx, r1);
    r1 = r0 * r1;
  };
  SumStore<double>(DepthScale_out_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r1);
  SumFlushFinal<double>(DepthScale_out_local, DepthScale_out, 1);
}

void DepthScaleAlphaDenominatorOrBetaNumerator(
    double* DepthScale_p_kp1,
    unsigned int DepthScale_p_kp1_num_alloc,
    double* DepthScale_w,
    unsigned int DepthScale_w_num_alloc,
    double* const DepthScale_out,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  DepthScaleAlphaDenominatorOrBetaNumeratorKernel<<<n_blocks, 1024>>>(
      DepthScale_p_kp1,
      DepthScale_p_kp1_num_alloc,
      DepthScale_w,
      DepthScale_w_num_alloc,
      DepthScale_out,
      problem_size);
}

}  // namespace caspar