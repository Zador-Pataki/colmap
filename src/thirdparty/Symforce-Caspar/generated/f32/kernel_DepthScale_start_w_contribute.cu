#include "kernel_DepthScale_start_w_contribute.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) DepthScaleStartWContributeKernel(
    float* DepthScale_precond_diag,
    unsigned int DepthScale_precond_diag_num_alloc,
    const float* const diag,
    float* DepthScale_p,
    unsigned int DepthScale_p_num_alloc,
    float* out_DepthScale_w,
    unsigned int out_DepthScale_w_num_alloc,
    size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  float r0, r1;

  if (global_thread_idx < problem_size) {
    ReadIdx1<1024, float, float, float>(DepthScale_precond_diag,
                                        0 * DepthScale_precond_diag_num_alloc,
                                        global_thread_idx,
                                        r0);
  };
  LoadUnique<1, float, float>(diag, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float*)inout_shared, 0, r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r1 = r0 * r1;
    ReadIdx1<1024, float, float, float>(
        DepthScale_p, 0 * DepthScale_p_num_alloc, global_thread_idx, r0);
    r1 = r1 * r0;
    AddIdx1<1024, float, float, float>(out_DepthScale_w,
                                       0 * out_DepthScale_w_num_alloc,
                                       global_thread_idx,
                                       r1);
  };
}

void DepthScaleStartWContribute(float* DepthScale_precond_diag,
                                unsigned int DepthScale_precond_diag_num_alloc,
                                const float* const diag,
                                float* DepthScale_p,
                                unsigned int DepthScale_p_num_alloc,
                                float* out_DepthScale_w,
                                unsigned int out_DepthScale_w_num_alloc,
                                size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  DepthScaleStartWContributeKernel<<<n_blocks, 1024>>>(
      DepthScale_precond_diag,
      DepthScale_precond_diag_num_alloc,
      diag,
      DepthScale_p,
      DepthScale_p_num_alloc,
      out_DepthScale_w,
      out_DepthScale_w_num_alloc,
      problem_size);
}

}  // namespace caspar