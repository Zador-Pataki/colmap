#include "kernel_DepthScale_update_p.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    DepthScaleUpdatePKernel(float* DepthScale_z,
                            unsigned int DepthScale_z_num_alloc,
                            float* DepthScale_p_k,
                            unsigned int DepthScale_p_k_num_alloc,
                            const float* const beta,
                            float* out_DepthScale_p_kp1,
                            unsigned int out_DepthScale_p_kp1_num_alloc,
                            size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  float r0, r1, r2;

  if (global_thread_idx < problem_size) {
    ReadIdx1<1024, float, float, float>(
        DepthScale_p_k, 0 * DepthScale_p_k_num_alloc, global_thread_idx, r0);
    ReadIdx1<1024, float, float, float>(
        DepthScale_z, 0 * DepthScale_z_num_alloc, global_thread_idx, r1);
  };
  LoadUnique<1, float, float>(beta, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float*)inout_shared, 0, r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r2 = fmaf(r0, r2, r1);
    WriteIdx1<1024, float, float, float>(out_DepthScale_p_kp1,
                                         0 * out_DepthScale_p_kp1_num_alloc,
                                         global_thread_idx,
                                         r2);
  };
}

void DepthScaleUpdateP(float* DepthScale_z,
                       unsigned int DepthScale_z_num_alloc,
                       float* DepthScale_p_k,
                       unsigned int DepthScale_p_k_num_alloc,
                       const float* const beta,
                       float* out_DepthScale_p_kp1,
                       unsigned int out_DepthScale_p_kp1_num_alloc,
                       size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  DepthScaleUpdatePKernel<<<n_blocks, 1024>>>(DepthScale_z,
                                              DepthScale_z_num_alloc,
                                              DepthScale_p_k,
                                              DepthScale_p_k_num_alloc,
                                              beta,
                                              out_DepthScale_p_kp1,
                                              out_DepthScale_p_kp1_num_alloc,
                                              problem_size);
}

}  // namespace caspar