#include "kernel_scale_prior_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ScalePriorJtjnjtrDirectKernel(double* scale_njtr,
                                  unsigned int scale_njtr_num_alloc,
                                  SharedIndex* scale_njtr_indices,
                                  double* scale_jac,
                                  unsigned int scale_jac_num_alloc,
                                  double* const out_scale_njtr,
                                  unsigned int out_scale_njtr_num_alloc,
                                  size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ SharedIndex scale_njtr_indices_loc[1024];
  scale_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? scale_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
}

void ScalePriorJtjnjtrDirect(double* scale_njtr,
                             unsigned int scale_njtr_num_alloc,
                             SharedIndex* scale_njtr_indices,
                             double* scale_jac,
                             unsigned int scale_jac_num_alloc,
                             double* const out_scale_njtr,
                             unsigned int out_scale_njtr_num_alloc,
                             size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ScalePriorJtjnjtrDirectKernel<<<n_blocks, 1024>>>(scale_njtr,
                                                    scale_njtr_num_alloc,
                                                    scale_njtr_indices,
                                                    scale_jac,
                                                    scale_jac_num_alloc,
                                                    out_scale_njtr,
                                                    out_scale_njtr_num_alloc,
                                                    problem_size);
}

}  // namespace caspar