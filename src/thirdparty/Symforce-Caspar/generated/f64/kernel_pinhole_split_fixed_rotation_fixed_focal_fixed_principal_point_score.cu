#include "kernel_pinhole_split_fixed_rotation_fixed_focal_fixed_principal_point_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedRotationFixedFocalFixedPrincipalPointScoreKernel(
        double* rotation,
        unsigned int rotation_num_alloc,
        double* translation,
        unsigned int translation_num_alloc,
        SharedIndex* translation_indices,
        double* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* weight_loss,
        unsigned int weight_loss_num_alloc,
        double* focal,
        unsigned int focal_num_alloc,
        double* principal_point,
        unsigned int principal_point_num_alloc,
        double* const out_rTr,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex translation_indices_loc[1024];
  translation_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? translation_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 0 * weight_loss_num_alloc, global_thread_idx, r0, r1);
    ReadIdx2<1024, double, double, double2>(principal_point,
                                            0 * principal_point_num_alloc,
                                            global_thread_idx,
                                            r2,
                                            r3);
    ReadIdx2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r4, r5);
    r6 = -1.00000000000000000e+00;
    r5 = fma(r5, r6, r3);
    ReadIdx2<1024, double, double, double2>(
        focal, 0 * focal_num_alloc, global_thread_idx, r3, r7);
  };
  LoadShared<2, double, double>(translation,
                                0 * translation_num_alloc,
                                translation_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        translation_indices_loc[threadIdx.x].target,
                        r8,
                        r9);
  };
  __syncthreads();
  LoadShared<2, double, double>(
      point, 0 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r10, r11);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r12 = -2.00000000000000000e+00;
    ReadIdx2<1024, double, double, double2>(
        rotation, 2 * rotation_num_alloc, global_thread_idx, r13, r14);
    r15 = r13 * r13;
    r15 = r12 * r15;
    r16 = 1.00000000000000000e+00;
    ReadIdx2<1024, double, double, double2>(
        rotation, 0 * rotation_num_alloc, global_thread_idx, r17, r18);
    r19 = r17 * r17;
    r19 = fma(r12, r19, r16);
    r20 = r15 + r19;
    r20 = fma(r11, r20, r9);
  };
  LoadShared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r21 = 2.00000000000000000e+00;
    r22 = r18 * r21;
    r23 = r13 * r22;
    r24 = r14 * r12;
    r25 = fma(r17, r24, r23);
    r26 = r13 * r14;
    r27 = r17 * r22;
    r26 = fma(r21, r26, r27);
    r20 = fma(r9, r25, r20);
    r20 = fma(r10, r26, r20);
    r26 = r7 * r20;
    r25 = 1.00000000000000008e-15;
  };
  LoadShared<1, double, double>(translation,
                                2 * translation_num_alloc,
                                translation_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared,
                        translation_indices_loc[threadIdx.x].target,
                        r28);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r29 = r18 * r18;
    r29 = r29 * r12;
    r19 = r29 + r19;
    r19 = fma(r9, r19, r28);
    r28 = r17 * r13;
    r28 = r28 * r21;
    r18 = fma(r18, r24, r28);
    r12 = r17 * r14;
    r12 = fma(r21, r12, r23);
    r19 = fma(r10, r18, r19);
    r19 = fma(r11, r12, r19);
    r12 = copysign(1.0, r19);
    r12 = fma(r25, r12, r19);
    r12 = 1.0 / r12;
    r5 = fma(r12, r26, r5);
    r4 = fma(r4, r6, r2);
    r15 = r16 + r15;
    r15 = r15 + r29;
    r15 = fma(r10, r15, r8);
    r22 = fma(r14, r22, r28);
    r24 = fma(r13, r24, r27);
    r15 = fma(r9, r22, r15);
    r15 = fma(r11, r24, r15);
    r24 = r3 * r15;
    r4 = fma(r12, r24, r4);
    r0 = fma(r0, r4, r1 * r5);
    r0 = r0 * r0;
    ReadIdx1<1024, double, double, double>(
        weight_loss, 6 * weight_loss_num_alloc, global_thread_idx, r1);
    r24 = 0.00000000000000000e+00;
    r1 = fmax(r1, r24);
    r12 = sqrt(r1);
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 2 * weight_loss_num_alloc, global_thread_idx, r11, r22);
    r4 = fma(r11, r4, r22 * r5);
    r4 = r4 * r4;
    r11 = r0 + r4;
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 4 * weight_loss_num_alloc, global_thread_idx, r5, r22);
    r9 = 5.00000000000000000e-01;
    r22 = fmax(r22, r25);
    r27 = r22 * r22;
    r28 = r21 * r22;
    r10 = fmax(r25, r11);
    r8 = sqrt(r10);
    r28 = fma(r8, r28, r6 * r27);
    r28 = r11 <= r27 ? r11 : r28;
    r8 = 2.50000000000000000e+00;
    r29 = 1.0 / r27;
    r29 = fma(r11, r29, r16);
    r16 = log(r29);
    r16 = r16 * r27;
    r28 = r5 < r8 ? r16 : r28;
    r16 = 1.50000000000000000e+00;
    r29 = sqrt(r29);
    r29 = r6 + r29;
    r29 = r21 * r29;
    r29 = r29 * r27;
    r28 = r5 < r16 ? r29 : r28;
    r28 = r5 < r9 ? r11 : r28;
    r28 = fmax(r24, r28);
    r28 = r1 * r28;
    r10 = 1.0 / r10;
    r28 = r28 * r10;
    r28 = sqrt(r28);
    r28 = r11 <= r25 ? r12 : r28;
    r28 = r28 * r28;
    r4 = fma(r28, r4, r28 * r0);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r4);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void PinholeSplitFixedRotationFixedFocalFixedPrincipalPointScore(
    double* rotation,
    unsigned int rotation_num_alloc,
    double* translation,
    unsigned int translation_num_alloc,
    SharedIndex* translation_indices,
    double* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* weight_loss,
    unsigned int weight_loss_num_alloc,
    double* focal,
    unsigned int focal_num_alloc,
    double* principal_point,
    unsigned int principal_point_num_alloc,
    double* const out_rTr,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitFixedRotationFixedFocalFixedPrincipalPointScoreKernel<<<n_blocks,
                                                                      1024>>>(
      rotation,
      rotation_num_alloc,
      translation,
      translation_num_alloc,
      translation_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      weight_loss,
      weight_loss_num_alloc,
      focal,
      focal_num_alloc,
      principal_point,
      principal_point_num_alloc,
      out_rTr,
      problem_size);
}

}  // namespace caspar