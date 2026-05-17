#include "kernel_pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedPoseFixedFocalFixedPrincipalPointResJacKernel(
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* weight_loss,
        unsigned int weight_loss_num_alloc,
        float* pose,
        unsigned int pose_num_alloc,
        float* focal,
        unsigned int focal_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_point_njtr,
        unsigned int out_point_njtr_num_alloc,
        float* const out_point_precond_diag,
        unsigned int out_point_precond_diag_num_alloc,
        float* const out_point_precond_tril,
        unsigned int out_point_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(weight_loss,
                                         0 * weight_loss_num_alloc,
                                         global_thread_idx,
                                         r0,
                                         r1,
                                         r2,
                                         r3);
    ReadIdx2<1024, float, float, float2>(principal_point,
                                         0 * principal_point_num_alloc,
                                         global_thread_idx,
                                         r4,
                                         r5);
    ReadIdx2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r6, r7);
    r8 = -1.00000000000000000e+00;
    r6 = fmaf(r6, r8, r4);
    r4 = 9.99999999999999955e-07;
    ReadIdx3<1024, float, float, float4>(
        pose, 4 * pose_num_alloc, global_thread_idx, r9, r10, r11);
  };
  LoadShared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_indices_loc[threadIdx.x].target,
                       r12,
                       r13,
                       r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(
        pose, 0 * pose_num_alloc, global_thread_idx, r15, r16, r17, r18);
    r19 = r16 * r17;
    r20 = 2.00000000000000000e+00;
    r19 = r19 * r20;
    r21 = r15 * r20;
    r22 = fmaf(r18, r21, r19);
    r11 = fmaf(r13, r22, r11);
    r23 = r17 * r21;
    r24 = -2.00000000000000000e+00;
    r25 = r18 * r24;
    r26 = fmaf(r16, r25, r23);
    r27 = r15 * r15;
    r27 = r27 * r24;
    r28 = 1.00000000000000000e+00;
    r29 = r16 * r16;
    r29 = fmaf(r24, r29, r28);
    r30 = r27 + r29;
    r11 = fmaf(r12, r26, r11);
    r11 = fmaf(r14, r30, r11);
    r31 = copysign(1.0, r11);
    r31 = fmaf(r4, r31, r11);
    r11 = 1.0 / r31;
    ReadIdx2<1024, float, float, float2>(
        focal, 0 * focal_num_alloc, global_thread_idx, r32, r33);
    r21 = r16 * r21;
    r34 = fmaf(r17, r25, r21);
    r9 = fmaf(r13, r34, r9);
    r35 = r16 * r18;
    r35 = fmaf(r20, r35, r23);
    r23 = r17 * r17;
    r23 = r24 * r23;
    r29 = r23 + r29;
    r9 = fmaf(r14, r35, r9);
    r9 = fmaf(r12, r29, r9);
    r9 = r32 * r9;
    r6 = fmaf(r11, r9, r6);
    r7 = fmaf(r7, r8, r5);
    r5 = r17 * r18;
    r5 = fmaf(r20, r5, r21);
    r12 = fmaf(r12, r5, r10);
    r25 = fmaf(r15, r25, r19);
    r23 = r28 + r23;
    r23 = r23 + r27;
    r12 = fmaf(r14, r25, r12);
    r12 = fmaf(r13, r23, r12);
    r12 = r33 * r12;
    r7 = fmaf(r11, r12, r7);
    r13 = fmaf(r1, r7, r0 * r6);
    ReadIdx3<1024, float, float, float4>(weight_loss,
                                         4 * weight_loss_num_alloc,
                                         global_thread_idx,
                                         r14,
                                         r27,
                                         r15);
    r19 = 0.00000000000000000e+00;
    r15 = fmaxf(r15, r19);
    r10 = sqrtf(r15);
    r7 = fmaf(r3, r7, r2 * r6);
    r6 = fmaf(r13, r13, r7 * r7);
    r21 = 5.00000000000000000e-01;
    r27 = fmaxf(r27, r4);
    r24 = r27 * r27;
    r36 = r20 * r27;
    r37 = fmaxf(r4, r6);
    r38 = sqrtf(r37);
    r36 = fmaf(r8, r24, r38 * r36);
    r36 = r6 <= r24 ? r6 : r36;
    r38 = 2.50000000000000000e+00;
    r39 = 1.0 / r24;
    r39 = fmaf(r6, r39, r28);
    r40 = logf(r39);
    r40 = r40 * r24;
    r36 = r14 < r38 ? r40 : r36;
    r40 = 1.50000000000000000e+00;
    r41 = sqrtf(r39);
    r41 = r8 + r41;
    r41 = r20 * r41;
    r41 = r41 * r24;
    r36 = r14 < r40 ? r41 : r36;
    r36 = r14 < r21 ? r6 : r36;
    r41 = fmaxf(r19, r36);
    r42 = 1.0 / r37;
    r42 = r15 * r42;
    r43 = r41 * r42;
    r44 = sqrtf(r43);
    r44 = r6 <= r4 ? r10 : r44;
    r10 = r13 * r44;
    r45 = r7 * r44;
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r10, r45);
    r45 = r8 * r7;
    r46 = r8 * r7;
    r47 = 2.50000000000000000e-01;
    r48 = r6 <= r24 ? r19 : r19;
    r48 = r14 < r38 ? r19 : r48;
    r48 = r14 < r40 ? r19 : r48;
    r48 = r14 < r21 ? r19 : r48;
    r48 = r47 * r48;
    r43 = rsqrtf(r43);
    r36 = copysign(1.0, r36);
    r36 = r28 + r36;
    r42 = r36 * r42;
    r48 = r48 * r43;
    r48 = r48 * r42;
    r48 = r6 <= r4 ? r19 : r48;
    r46 = r46 * r48;
    r36 = r20 * r7;
    r47 = r32 * r29;
    r31 = r31 * r31;
    r31 = 1.0 / r31;
    r31 = r8 * r31;
    r9 = r31 * r9;
    r47 = fmaf(r26, r9, r11 * r47);
    r49 = r26 * r31;
    r50 = r33 * r5;
    r50 = fmaf(r11, r50, r12 * r49);
    r49 = fmaf(r3, r50, r2 * r47);
    r51 = r20 * r13;
    r50 = fmaf(r1, r50, r0 * r47);
    r51 = fmaf(r50, r51, r49 * r36);
    r41 = r15 * r41;
    r15 = -5.00000000000000000e-01;
    r36 = -9.99999999999999955e-07;
    r36 = r36 + r6;
    r36 = copysign(1.0, r36);
    r36 = r28 + r36;
    r28 = r37 * r37;
    r28 = 1.0 / r28;
    r41 = r41 * r15;
    r41 = r41 * r36;
    r41 = r41 * r28;
    r28 = r21 * r27;
    r37 = rsqrtf(r37);
    r28 = r28 * r36;
    r28 = r28 * r37;
    r37 = r51 * r28;
    r37 = r6 <= r24 ? r51 : r37;
    r36 = 1.0 / r39;
    r15 = r51 * r36;
    r37 = r14 < r38 ? r15 : r37;
    r39 = rsqrtf(r39);
    r15 = r51 * r39;
    r37 = r14 < r40 ? r15 : r37;
    r37 = r14 < r21 ? r51 : r37;
    r15 = r21 * r37;
    r15 = fmaf(r42, r15, r51 * r41);
    r15 = r21 * r15;
    r15 = r15 * r43;
    r15 = r6 <= r4 ? r19 : r15;
    r51 = fmaf(r7, r15, r46);
    r51 = fmaf(r44, r49, r51);
    r45 = r45 * r44;
    r49 = r8 * r13;
    r49 = r49 * r48;
    r15 = fmaf(r13, r15, r49);
    r15 = fmaf(r44, r50, r15);
    r10 = r8 * r10;
    r45 = fmaf(r15, r10, r51 * r45);
    r50 = r8 * r7;
    r48 = r20 * r13;
    r47 = r32 * r34;
    r47 = fmaf(r11, r47, r22 * r9);
    r52 = r33 * r23;
    r53 = r22 * r31;
    r53 = fmaf(r12, r53, r11 * r52);
    r52 = fmaf(r1, r53, r0 * r47);
    r54 = r20 * r7;
    r53 = fmaf(r3, r53, r2 * r47);
    r54 = fmaf(r53, r54, r52 * r48);
    r48 = r54 * r28;
    r48 = r6 <= r24 ? r54 : r48;
    r47 = r54 * r36;
    r48 = r14 < r38 ? r47 : r48;
    r47 = r54 * r39;
    r48 = r14 < r40 ? r47 : r48;
    r48 = r14 < r21 ? r54 : r48;
    r47 = r21 * r48;
    r54 = fmaf(r54, r41, r42 * r47);
    r54 = r21 * r54;
    r54 = r54 * r43;
    r54 = r6 <= r4 ? r19 : r54;
    r47 = fmaf(r7, r54, r46);
    r47 = fmaf(r44, r53, r47);
    r50 = r50 * r44;
    r54 = fmaf(r13, r54, r49);
    r54 = fmaf(r44, r52, r54);
    r50 = fmaf(r54, r10, r47 * r50);
    r52 = r8 * r7;
    r53 = r20 * r7;
    r55 = r32 * r35;
    r55 = fmaf(r11, r55, r30 * r9);
    r9 = r30 * r31;
    r56 = r33 * r25;
    r56 = fmaf(r11, r56, r12 * r9);
    r3 = fmaf(r3, r56, r2 * r55);
    r2 = r20 * r13;
    r56 = fmaf(r1, r56, r0 * r55);
    r2 = fmaf(r56, r2, r3 * r53);
    r28 = r2 * r28;
    r28 = r6 <= r24 ? r2 : r28;
    r36 = r2 * r36;
    r28 = r14 < r38 ? r36 : r28;
    r39 = r2 * r39;
    r28 = r14 < r40 ? r39 : r28;
    r28 = r14 < r21 ? r2 : r28;
    r14 = r21 * r28;
    r14 = fmaf(r42, r14, r2 * r41);
    r14 = r21 * r14;
    r14 = r14 * r43;
    r14 = r6 <= r4 ? r19 : r14;
    r46 = fmaf(r7, r14, r46);
    r46 = fmaf(r44, r3, r46);
    r52 = r52 * r44;
    r14 = fmaf(r13, r14, r49);
    r14 = fmaf(r44, r56, r14);
    r10 = fmaf(r14, r10, r46 * r52);
    WriteSum3<float, float>((float*)inout_shared, r45, r50, r10);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r10 = fmaf(r51, r51, r15 * r15);
    r50 = fmaf(r47, r47, r54 * r54);
    r45 = fmaf(r14, r14, r46 * r46);
    WriteSum3<float, float>((float*)inout_shared, r10, r50, r45);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r45 = fmaf(r51, r47, r15 * r54);
    r15 = fmaf(r15, r14, r51 * r46);
    r14 = fmaf(r54, r14, r47 * r46);
    WriteSum3<float, float>((float*)inout_shared, r45, r15, r14);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
}

void PinholeSplitFixedPoseFixedFocalFixedPrincipalPointResJac(
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* weight_loss,
    unsigned int weight_loss_num_alloc,
    float* pose,
    unsigned int pose_num_alloc,
    float* focal,
    unsigned int focal_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    float* const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    float* const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitFixedPoseFixedFocalFixedPrincipalPointResJacKernel<<<n_blocks,
                                                                   1024>>>(
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      weight_loss,
      weight_loss_num_alloc,
      pose,
      pose_num_alloc,
      focal,
      focal_num_alloc,
      principal_point,
      principal_point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      out_point_precond_diag,
      out_point_precond_diag_num_alloc,
      out_point_precond_tril,
      out_point_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar