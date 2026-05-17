#include "kernel_pinhole_log_depth_fixed_rotation_fixed_point_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeLogDepthFixedRotationFixedPointScoreKernel(
        double* rotation,
        unsigned int rotation_num_alloc,
        double* translation,
        unsigned int translation_num_alloc,
        SharedIndex* translation_indices,
        double* scale,
        unsigned int scale_num_alloc,
        SharedIndex* scale_indices,
        double* log_depth,
        unsigned int log_depth_num_alloc,
        double* loss,
        unsigned int loss_num_alloc,
        double* point,
        unsigned int point_num_alloc,
        double* const out_rTr,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex translation_indices_loc[1024];
  translation_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? translation_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex scale_indices_loc[1024];
  scale_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? scale_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18;

  if (global_thread_idx < problem_size) {
    r0 = 0.00000000000000000e+00;
  };
  LoadShared<1, double, double>(translation,
                                2 * translation_num_alloc,
                                translation_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, translation_indices_loc[threadIdx.x].target, r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx1<1024, double, double, double>(
        point, 2 * point_num_alloc, global_thread_idx, r2);
    r3 = 1.00000000000000000e+00;
    r4 = -2.00000000000000000e+00;
    ReadIdx2<1024, double, double, double2>(
        rotation, 0 * rotation_num_alloc, global_thread_idx, r5, r6);
    r7 = r6 * r6;
    r7 = fma(r4, r7, r3);
    r8 = r5 * r5;
    r7 = fma(r4, r8, r7);
    r7 = fma(r2, r7, r1);
    ReadIdx2<1024, double, double, double2>(
        point, 0 * point_num_alloc, global_thread_idx, r2, r1);
    ReadIdx2<1024, double, double, double2>(
        rotation, 2 * rotation_num_alloc, global_thread_idx, r8, r9);
    r10 = r6 * r9;
    r11 = 2.00000000000000000e+00;
    r8 = r8 * r11;
    r10 = fma(r5, r8, r4 * r10);
    r4 = r5 * r9;
    r8 = fma(r6, r8, r11 * r4);
    r7 = fma(r2, r10, r7);
    r7 = fma(r1, r8, r7);
    r8 = 1.00000000000000008e-15;
    r1 = fmax(r8, r7);
    r1 = log(r1);
  };
  LoadShared<1, double, double>(
      scale, 0 * scale_num_alloc, scale_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, scale_indices_loc[threadIdx.x].target, r10);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r2 = -1.00000000000000000e+00;
    r10 = fma(r10, r2, r1);
    ReadIdx1<1024, double, double, double>(
        log_depth, 0 * log_depth_num_alloc, global_thread_idx, r1);
    r10 = fma(r1, r2, r10);
    r10 = r0 < r7 ? r10 : r0;
    r10 = r10 * r10;
    ReadIdx1<1024, double, double, double>(
        loss, 2 * loss_num_alloc, global_thread_idx, r7);
    r7 = fmax(r7, r0);
    r1 = sqrt(r7);
    ReadIdx2<1024, double, double, double2>(
        loss, 0 * loss_num_alloc, global_thread_idx, r4, r12);
    r13 = 5.00000000000000000e-01;
    r12 = fmax(r12, r8);
    r14 = r12 * r12;
    r15 = r11 * r12;
    r16 = fmax(r8, r10);
    r17 = sqrt(r16);
    r15 = fma(r17, r15, r2 * r14);
    r15 = r10 <= r14 ? r10 : r15;
    r17 = 2.50000000000000000e+00;
    r18 = 1.0 / r14;
    r18 = fma(r18, r10, r3);
    r3 = log(r18);
    r3 = r3 * r14;
    r15 = r4 < r17 ? r3 : r15;
    r3 = 1.50000000000000000e+00;
    r18 = sqrt(r18);
    r18 = r2 + r18;
    r18 = r11 * r18;
    r18 = r18 * r14;
    r15 = r4 < r3 ? r18 : r15;
    r15 = r4 < r13 ? r10 : r15;
    r15 = fmax(r0, r15);
    r15 = r7 * r15;
    r16 = 1.0 / r16;
    r15 = r15 * r16;
    r15 = sqrt(r15);
    r15 = r10 <= r8 ? r1 : r15;
    r15 = r15 * r15;
    r15 = r10 * r15;
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r15);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void PinholeLogDepthFixedRotationFixedPointScore(
    double* rotation,
    unsigned int rotation_num_alloc,
    double* translation,
    unsigned int translation_num_alloc,
    SharedIndex* translation_indices,
    double* scale,
    unsigned int scale_num_alloc,
    SharedIndex* scale_indices,
    double* log_depth,
    unsigned int log_depth_num_alloc,
    double* loss,
    unsigned int loss_num_alloc,
    double* point,
    unsigned int point_num_alloc,
    double* const out_rTr,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeLogDepthFixedRotationFixedPointScoreKernel<<<n_blocks, 1024>>>(
      rotation,
      rotation_num_alloc,
      translation,
      translation_num_alloc,
      translation_indices,
      scale,
      scale_num_alloc,
      scale_indices,
      log_depth,
      log_depth_num_alloc,
      loss,
      loss_num_alloc,
      point,
      point_num_alloc,
      out_rTr,
      problem_size);
}

}  // namespace caspar