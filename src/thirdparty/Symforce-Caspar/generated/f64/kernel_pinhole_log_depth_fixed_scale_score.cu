#include "kernel_pinhole_log_depth_fixed_scale_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeLogDepthFixedScaleScoreKernel(double* pose,
                                         unsigned int pose_num_alloc,
                                         SharedIndex* pose_indices,
                                         double* point,
                                         unsigned int point_num_alloc,
                                         SharedIndex* point_indices,
                                         double* log_depth,
                                         unsigned int log_depth_num_alloc,
                                         double* loss,
                                         unsigned int loss_num_alloc,
                                         double* scale,
                                         unsigned int scale_num_alloc,
                                         double* const out_rTr,
                                         size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17;

  if (global_thread_idx < problem_size) {
    r0 = 0.00000000000000000e+00;
  };
  LoadShared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r1);
  };
  __syncthreads();
  LoadShared<2, double, double>(
      point, 0 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r2, r3);
  };
  __syncthreads();
  LoadShared<2, double, double>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r4, r5);
  };
  __syncthreads();
  LoadShared<2, double, double>(
      pose, 2 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r6, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r8 = r4 * r7;
    r9 = 2.00000000000000000e+00;
    r6 = r6 * r9;
    r8 = fma(r5, r6, r9 * r8);
    r8 = fma(r3, r8, r1);
    r3 = -2.00000000000000000e+00;
    r1 = r5 * r3;
    r6 = fma(r7, r1, r4 * r6);
  };
  LoadShared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r10);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r11 = 1.00000000000000000e+00;
    r1 = fma(r5, r1, r11);
    r5 = r4 * r4;
    r1 = fma(r3, r5, r1);
    r8 = fma(r2, r6, r8);
    r8 = fma(r10, r1, r8);
    ReadIdx1<1024, double, double, double>(
        scale, 0 * scale_num_alloc, global_thread_idx, r1);
    r10 = -1.00000000000000000e+00;
    ReadIdx1<1024, double, double, double>(
        log_depth, 0 * log_depth_num_alloc, global_thread_idx, r6);
    r6 = fma(r6, r10, r1 * r10);
    r1 = 1.00000000000000008e-15;
    r2 = fmax(r8, r1);
    r2 = log(r2);
    r6 = r6 + r2;
    r6 = r0 < r8 ? r6 : r0;
    r6 = r6 * r6;
    ReadIdx1<1024, double, double, double>(
        loss, 2 * loss_num_alloc, global_thread_idx, r8);
    r8 = fmax(r8, r0);
    r2 = sqrt(r8);
    ReadIdx2<1024, double, double, double2>(
        loss, 0 * loss_num_alloc, global_thread_idx, r5, r3);
    r12 = 5.00000000000000000e-01;
    r3 = fmax(r3, r1);
    r13 = r3 * r3;
    r14 = r9 * r3;
    r15 = fmax(r1, r6);
    r16 = sqrt(r15);
    r14 = fma(r16, r14, r10 * r13);
    r14 = r6 <= r13 ? r6 : r14;
    r16 = 2.50000000000000000e+00;
    r17 = 1.0 / r13;
    r17 = fma(r17, r6, r11);
    r11 = log(r17);
    r11 = r11 * r13;
    r14 = r5 < r16 ? r11 : r14;
    r11 = 1.50000000000000000e+00;
    r17 = sqrt(r17);
    r17 = r10 + r17;
    r17 = r9 * r17;
    r17 = r17 * r13;
    r14 = r5 < r11 ? r17 : r14;
    r14 = r5 < r12 ? r6 : r14;
    r14 = fmax(r0, r14);
    r14 = r8 * r14;
    r15 = 1.0 / r15;
    r14 = r14 * r15;
    r14 = sqrt(r14);
    r14 = r6 <= r1 ? r2 : r14;
    r14 = r14 * r14;
    r14 = r6 * r14;
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r14);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void PinholeLogDepthFixedScaleScore(double* pose,
                                    unsigned int pose_num_alloc,
                                    SharedIndex* pose_indices,
                                    double* point,
                                    unsigned int point_num_alloc,
                                    SharedIndex* point_indices,
                                    double* log_depth,
                                    unsigned int log_depth_num_alloc,
                                    double* loss,
                                    unsigned int loss_num_alloc,
                                    double* scale,
                                    unsigned int scale_num_alloc,
                                    double* const out_rTr,
                                    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeLogDepthFixedScaleScoreKernel<<<n_blocks, 1024>>>(pose,
                                                           pose_num_alloc,
                                                           pose_indices,
                                                           point,
                                                           point_num_alloc,
                                                           point_indices,
                                                           log_depth,
                                                           log_depth_num_alloc,
                                                           loss,
                                                           loss_num_alloc,
                                                           scale,
                                                           scale_num_alloc,
                                                           out_rTr,
                                                           problem_size);
}

}  // namespace caspar