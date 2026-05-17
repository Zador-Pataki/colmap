#include "kernel_pinhole_log_depth_fixed_pose_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeLogDepthFixedPoseJtjnjtrDirectKernel(
        double* scale_njtr,
        unsigned int scale_njtr_num_alloc,
        SharedIndex* scale_njtr_indices,
        double* scale_jac,
        unsigned int scale_jac_num_alloc,
        double* point_njtr,
        unsigned int point_njtr_num_alloc,
        SharedIndex* point_njtr_indices,
        double* point_jac,
        unsigned int point_jac_num_alloc,
        double* const out_scale_njtr,
        unsigned int out_scale_njtr_num_alloc,
        double* const out_point_njtr,
        unsigned int out_point_njtr_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex scale_njtr_indices_loc[1024];
  scale_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? scale_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex point_njtr_indices_loc[1024];
  point_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6;

  if (global_thread_idx < problem_size) {
    ReadIdx1<1024, double, double, double>(
        scale_jac, 0 * scale_jac_num_alloc, global_thread_idx, r0);
  };
  LoadShared<1, double, double>(point_njtr,
                                2 * point_njtr_num_alloc,
                                point_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_njtr_indices_loc[threadIdx.x].target, r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx1<1024, double, double, double>(
        point_jac, 2 * point_jac_num_alloc, global_thread_idx, r2);
  };
  LoadShared<2, double, double>(point_njtr,
                                0 * point_njtr_num_alloc,
                                point_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        point_njtr_indices_loc[threadIdx.x].target,
                        r3,
                        r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        point_jac, 0 * point_jac_num_alloc, global_thread_idx, r5, r6);
    r4 = fma(r4, r6, r1 * r2);
    r4 = fma(r3, r5, r4);
    r4 = r0 * r4;
    WriteSum1<double, double>((double*)inout_shared, r4);
  };
  FlushSumShared<1, double>(out_scale_njtr,
                            0 * out_scale_njtr_num_alloc,
                            scale_njtr_indices_loc,
                            (double*)inout_shared);
  LoadShared<1, double, double>(scale_njtr,
                                0 * scale_njtr_num_alloc,
                                scale_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, scale_njtr_indices_loc[threadIdx.x].target, r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r0 = r4 * r0;
    r5 = r5 * r0;
    r6 = r6 * r0;
    WriteSum2<double, double>((double*)inout_shared, r5, r6);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = r2 * r0;
    WriteSum1<double, double>((double*)inout_shared, r0);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_njtr_indices_loc,
                            (double*)inout_shared);
}

void PinholeLogDepthFixedPoseJtjnjtrDirect(
    double* scale_njtr,
    unsigned int scale_njtr_num_alloc,
    SharedIndex* scale_njtr_indices,
    double* scale_jac,
    unsigned int scale_jac_num_alloc,
    double* point_njtr,
    unsigned int point_njtr_num_alloc,
    SharedIndex* point_njtr_indices,
    double* point_jac,
    unsigned int point_jac_num_alloc,
    double* const out_scale_njtr,
    unsigned int out_scale_njtr_num_alloc,
    double* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeLogDepthFixedPoseJtjnjtrDirectKernel<<<n_blocks, 1024>>>(
      scale_njtr,
      scale_njtr_num_alloc,
      scale_njtr_indices,
      scale_jac,
      scale_jac_num_alloc,
      point_njtr,
      point_njtr_num_alloc,
      point_njtr_indices,
      point_jac,
      point_jac_num_alloc,
      out_scale_njtr,
      out_scale_njtr_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar