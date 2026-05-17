#include "kernel_pinhole_split_fixed_pose_fixed_principal_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedPoseFixedPrincipalPointResJacFirstKernel(
        float* focal,
        unsigned int focal_num_alloc,
        SharedIndex* focal_indices,
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* weight_loss,
        unsigned int weight_loss_num_alloc,
        float* pose,
        unsigned int pose_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_rTr,
        float* out_focal_jac,
        unsigned int out_focal_jac_num_alloc,
        float* const out_focal_njtr,
        unsigned int out_focal_njtr_num_alloc,
        float* const out_focal_precond_diag,
        unsigned int out_focal_precond_diag_num_alloc,
        float* const out_focal_precond_tril,
        unsigned int out_focal_precond_tril_num_alloc,
        float* out_point_jac,
        unsigned int out_point_jac_num_alloc,
        float* const out_point_njtr,
        unsigned int out_point_njtr_num_alloc,
        float* const out_point_precond_diag,
        unsigned int out_point_precond_diag_num_alloc,
        float* const out_point_precond_tril,
        unsigned int out_point_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex focal_indices_loc[1024];
  focal_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60;

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
  };
  LoadShared<2, float, float>(
      focal, 0 * focal_num_alloc, focal_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>(
        (float*)inout_shared, focal_indices_loc[threadIdx.x].target, r4, r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx3<1024, float, float, float4>(
        pose, 4 * pose_num_alloc, global_thread_idx, r10, r11, r12);
  };
  LoadShared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_indices_loc[threadIdx.x].target,
                       r13,
                       r14,
                       r15);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(
        pose, 0 * pose_num_alloc, global_thread_idx, r16, r17, r18, r19);
    r20 = r16 * r17;
    r21 = 2.00000000000000000e+00;
    r20 = r20 * r21;
    r22 = -2.00000000000000000e+00;
    r23 = r19 * r22;
    r24 = fmaf(r18, r23, r20);
    r10 = fmaf(r14, r24, r10);
    r25 = r16 * r18;
    r25 = r25 * r21;
    r26 = r17 * r19;
    r26 = fmaf(r21, r26, r25);
    r27 = r18 * r18;
    r27 = r22 * r27;
    r28 = 1.00000000000000000e+00;
    r29 = r17 * r17;
    r29 = fmaf(r22, r29, r28);
    r30 = r27 + r29;
    r10 = fmaf(r15, r26, r10);
    r10 = fmaf(r13, r30, r10);
    r31 = 9.99999999999999955e-07;
    r32 = r17 * r18;
    r32 = r32 * r21;
    r33 = r16 * r19;
    r33 = fmaf(r21, r33, r32);
    r12 = fmaf(r14, r33, r12);
    r25 = fmaf(r17, r23, r25);
    r34 = r16 * r16;
    r34 = r22 * r34;
    r29 = r34 + r29;
    r12 = fmaf(r13, r25, r12);
    r12 = fmaf(r15, r29, r12);
    r22 = copysign(1.0, r12);
    r22 = fmaf(r31, r22, r12);
    r12 = 1.0 / r22;
    r35 = r10 * r12;
    r6 = fmaf(r4, r35, r6);
    r7 = fmaf(r7, r8, r5);
    r5 = r18 * r19;
    r5 = fmaf(r21, r5, r20);
    r13 = fmaf(r13, r5, r11);
    r23 = fmaf(r16, r23, r32);
    r27 = r28 + r27;
    r27 = r27 + r34;
    r13 = fmaf(r15, r23, r13);
    r13 = fmaf(r14, r27, r13);
    r14 = r9 * r13;
    r7 = fmaf(r12, r14, r7);
    r15 = fmaf(r1, r7, r0 * r6);
    ReadIdx3<1024, float, float, float4>(weight_loss,
                                         4 * weight_loss_num_alloc,
                                         global_thread_idx,
                                         r34,
                                         r32,
                                         r11);
    r20 = 0.00000000000000000e+00;
    r11 = fmaxf(r11, r20);
    r36 = sqrtf(r11);
    r7 = fmaf(r3, r7, r2 * r6);
    r6 = fmaf(r15, r15, r7 * r7);
    r37 = 5.00000000000000000e-01;
    r32 = fmaxf(r32, r31);
    r38 = r32 * r32;
    r39 = r21 * r32;
    r40 = fmaxf(r31, r6);
    r41 = sqrtf(r40);
    r42 = r8 * r32;
    r42 = fmaf(r32, r42, r41 * r39);
    r42 = r6 <= r38 ? r6 : r42;
    r39 = 2.50000000000000000e+00;
    r41 = r32 * r32;
    r43 = 1.0 / r38;
    r43 = fmaf(r6, r43, r28);
    r44 = logf(r43);
    r41 = r41 * r44;
    r42 = r34 < r39 ? r41 : r42;
    r41 = 1.50000000000000000e+00;
    r44 = r21 * r32;
    r45 = sqrtf(r43);
    r45 = r8 + r45;
    r44 = r44 * r32;
    r44 = r44 * r45;
    r42 = r34 < r41 ? r44 : r42;
    r42 = r34 < r37 ? r6 : r42;
    r44 = fmaxf(r20, r42);
    r45 = 1.0 / r40;
    r45 = r11 * r45;
    r46 = r44 * r45;
    r47 = sqrtf(r46);
    r47 = r6 <= r31 ? r36 : r47;
    r36 = r15 * r47;
    r48 = r7 * r47;
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r36, r48);
    r48 = r7 * r7;
    r48 = r48 * r47;
    r36 = r15 * r15;
    r36 = r36 * r47;
    r36 = fmaf(r47, r36, r47 * r48);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r36);
  if (global_thread_idx < problem_size) {
    r36 = r8 * r15;
    r48 = 2.50000000000000000e-01;
    r49 = r6 <= r38 ? r20 : r20;
    r49 = r34 < r39 ? r20 : r49;
    r49 = r34 < r41 ? r20 : r49;
    r49 = r34 < r37 ? r20 : r49;
    r49 = r48 * r49;
    r46 = rsqrtf(r46);
    r42 = copysign(1.0, r42);
    r42 = r28 + r42;
    r45 = r42 * r45;
    r49 = r49 * r46;
    r49 = r49 * r45;
    r49 = r6 <= r31 ? r20 : r49;
    r36 = r36 * r49;
    r42 = r21 * r15;
    r48 = r0 * r35;
    r50 = r2 * r21;
    r50 = r50 * r7;
    r50 = fmaf(r35, r50, r42 * r48);
    r51 = r37 * r32;
    r52 = -9.99999999999999955e-07;
    r52 = r52 + r6;
    r52 = copysign(1.0, r52);
    r52 = r28 + r52;
    r28 = rsqrtf(r40);
    r51 = r51 * r52;
    r51 = r51 * r28;
    r28 = r50 * r51;
    r28 = r6 <= r38 ? r50 : r28;
    r53 = 1.0 / r43;
    r54 = r50 * r53;
    r28 = r34 < r39 ? r54 : r28;
    r43 = rsqrtf(r43);
    r54 = r50 * r43;
    r28 = r34 < r41 ? r54 : r28;
    r28 = r34 < r37 ? r50 : r28;
    r54 = r37 * r28;
    r44 = r11 * r44;
    r11 = -5.00000000000000000e-01;
    r40 = r40 * r40;
    r40 = 1.0 / r40;
    r44 = r44 * r11;
    r44 = r44 * r52;
    r44 = r44 * r40;
    r50 = fmaf(r50, r44, r45 * r54);
    r50 = r37 * r50;
    r50 = r50 * r46;
    r50 = r6 <= r31 ? r20 : r50;
    r54 = fmaf(r15, r50, r36);
    r54 = fmaf(r47, r48, r54);
    r48 = r8 * r7;
    r48 = r48 * r49;
    r50 = fmaf(r7, r50, r48);
    r49 = r2 * r47;
    r50 = fmaf(r35, r49, r50);
    r49 = r1 * r13;
    r49 = r49 * r12;
    r35 = r3 * r21;
    r35 = r35 * r13;
    r35 = r35 * r7;
    r35 = fmaf(r12, r35, r42 * r49);
    r49 = r35 * r51;
    r49 = r6 <= r38 ? r35 : r49;
    r40 = r35 * r53;
    r49 = r34 < r39 ? r40 : r49;
    r40 = r35 * r43;
    r49 = r34 < r41 ? r40 : r49;
    r49 = r34 < r37 ? r35 : r49;
    r40 = r37 * r49;
    r40 = fmaf(r45, r40, r35 * r44);
    r40 = r37 * r40;
    r40 = r40 * r46;
    r40 = r6 <= r31 ? r20 : r40;
    r35 = fmaf(r15, r40, r36);
    r52 = r1 * r13;
    r52 = r52 * r47;
    r35 = fmaf(r12, r52, r35);
    r40 = fmaf(r7, r40, r48);
    r52 = r3 * r13;
    r52 = r52 * r47;
    r40 = fmaf(r12, r52, r40);
    WriteIdx4<1024, float, float, float4>(out_focal_jac,
                                          0 * out_focal_jac_num_alloc,
                                          global_thread_idx,
                                          r54,
                                          r50,
                                          r35,
                                          r40);
    r52 = r8 * r47;
    r11 = r7 * r52;
    r55 = r15 * r54;
    r55 = fmaf(r52, r55, r50 * r11);
    r56 = r15 * r35;
    r56 = fmaf(r40, r11, r52 * r56);
    WriteSum2<float, float>((float*)inout_shared, r55, r56);
  };
  FlushSumShared<2, float>(out_focal_njtr,
                           0 * out_focal_njtr_num_alloc,
                           focal_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r56 = fmaf(r54, r54, r50 * r50);
    r55 = fmaf(r35, r35, r40 * r40);
    WriteSum2<float, float>((float*)inout_shared, r56, r55);
  };
  FlushSumShared<2, float>(out_focal_precond_diag,
                           0 * out_focal_precond_diag_num_alloc,
                           focal_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = fmaf(r54, r35, r50 * r40);
    WriteSum1<float, float>((float*)inout_shared, r40);
  };
  FlushSumShared<1, float>(out_focal_precond_tril,
                           0 * out_focal_precond_tril_num_alloc,
                           focal_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = r21 * r7;
    r50 = r4 * r30;
    r10 = r4 * r10;
    r22 = r22 * r22;
    r22 = 1.0 / r22;
    r10 = r10 * r8;
    r10 = r10 * r22;
    r50 = fmaf(r25, r10, r12 * r50);
    r55 = r25 * r8;
    r55 = r55 * r22;
    r56 = r9 * r5;
    r56 = fmaf(r12, r56, r14 * r55);
    r55 = fmaf(r3, r56, r2 * r50);
    r56 = fmaf(r1, r56, r0 * r50);
    r40 = fmaf(r56, r42, r55 * r40);
    r50 = r40 * r51;
    r50 = r6 <= r38 ? r40 : r50;
    r57 = r40 * r53;
    r50 = r34 < r39 ? r57 : r50;
    r57 = r40 * r43;
    r50 = r34 < r41 ? r57 : r50;
    r50 = r34 < r37 ? r40 : r50;
    r57 = r37 * r50;
    r57 = fmaf(r45, r57, r40 * r44);
    r57 = r37 * r57;
    r57 = r57 * r46;
    r57 = r6 <= r31 ? r20 : r57;
    r40 = fmaf(r15, r57, r36);
    r40 = fmaf(r47, r56, r40);
    r57 = fmaf(r7, r57, r48);
    r57 = fmaf(r47, r55, r57);
    r55 = r21 * r7;
    r56 = r4 * r24;
    r56 = fmaf(r12, r56, r33 * r10);
    r58 = r9 * r27;
    r59 = r33 * r8;
    r59 = r59 * r22;
    r59 = fmaf(r14, r59, r12 * r58);
    r58 = fmaf(r3, r59, r2 * r56);
    r59 = fmaf(r1, r59, r0 * r56);
    r55 = fmaf(r59, r42, r58 * r55);
    r56 = r55 * r51;
    r56 = r6 <= r38 ? r55 : r56;
    r60 = r55 * r53;
    r56 = r34 < r39 ? r60 : r56;
    r60 = r55 * r43;
    r56 = r34 < r41 ? r60 : r56;
    r56 = r34 < r37 ? r55 : r56;
    r60 = r37 * r56;
    r55 = fmaf(r55, r44, r45 * r60);
    r55 = r37 * r55;
    r55 = r55 * r46;
    r55 = r6 <= r31 ? r20 : r55;
    r60 = fmaf(r15, r55, r36);
    r60 = fmaf(r47, r59, r60);
    r55 = fmaf(r7, r55, r48);
    r55 = fmaf(r47, r58, r55);
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r40,
                                          r57,
                                          r60,
                                          r55);
    r58 = r21 * r7;
    r59 = r4 * r26;
    r59 = fmaf(r12, r59, r29 * r10);
    r10 = r29 * r8;
    r10 = r10 * r22;
    r22 = r9 * r23;
    r22 = fmaf(r12, r22, r14 * r10);
    r10 = fmaf(r3, r22, r2 * r59);
    r22 = fmaf(r1, r22, r0 * r59);
    r42 = fmaf(r22, r42, r10 * r58);
    r51 = r42 * r51;
    r51 = r6 <= r38 ? r42 : r51;
    r53 = r42 * r53;
    r51 = r34 < r39 ? r53 : r51;
    r43 = r42 * r43;
    r51 = r34 < r41 ? r43 : r51;
    r51 = r34 < r37 ? r42 : r51;
    r34 = r37 * r51;
    r34 = fmaf(r45, r34, r42 * r44);
    r34 = r37 * r34;
    r34 = r34 * r46;
    r34 = r6 <= r31 ? r20 : r34;
    r36 = fmaf(r15, r34, r36);
    r36 = fmaf(r47, r22, r36);
    r34 = fmaf(r7, r34, r48);
    r34 = fmaf(r47, r10, r34);
    WriteIdx2<1024, float, float, float2>(out_point_jac,
                                          4 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r36,
                                          r34);
    r10 = r15 * r40;
    r10 = fmaf(r52, r10, r57 * r11);
    r48 = r15 * r60;
    r48 = fmaf(r55, r11, r52 * r48);
    r22 = r15 * r36;
    r11 = fmaf(r34, r11, r52 * r22);
    WriteSum3<float, float>((float*)inout_shared, r10, r48, r11);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = fmaf(r57, r57, r40 * r40);
    r48 = fmaf(r55, r55, r60 * r60);
    r10 = fmaf(r36, r36, r34 * r34);
    WriteSum3<float, float>((float*)inout_shared, r11, r48, r10);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r10 = fmaf(r57, r55, r40 * r60);
    r57 = fmaf(r40, r36, r57 * r34);
    r34 = fmaf(r60, r36, r55 * r34);
    WriteSum3<float, float>((float*)inout_shared, r10, r57, r34);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void PinholeSplitFixedPoseFixedPrincipalPointResJacFirst(
    float* focal,
    unsigned int focal_num_alloc,
    SharedIndex* focal_indices,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* weight_loss,
    unsigned int weight_loss_num_alloc,
    float* pose,
    unsigned int pose_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
    float* out_focal_jac,
    unsigned int out_focal_jac_num_alloc,
    float* const out_focal_njtr,
    unsigned int out_focal_njtr_num_alloc,
    float* const out_focal_precond_diag,
    unsigned int out_focal_precond_diag_num_alloc,
    float* const out_focal_precond_tril,
    unsigned int out_focal_precond_tril_num_alloc,
    float* out_point_jac,
    unsigned int out_point_jac_num_alloc,
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
  PinholeSplitFixedPoseFixedPrincipalPointResJacFirstKernel<<<n_blocks, 1024>>>(
      focal,
      focal_num_alloc,
      focal_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      weight_loss,
      weight_loss_num_alloc,
      pose,
      pose_num_alloc,
      principal_point,
      principal_point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_rTr,
      out_focal_jac,
      out_focal_jac_num_alloc,
      out_focal_njtr,
      out_focal_njtr_num_alloc,
      out_focal_precond_diag,
      out_focal_precond_diag_num_alloc,
      out_focal_precond_tril,
      out_focal_precond_tril_num_alloc,
      out_point_jac,
      out_point_jac_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      out_point_precond_diag,
      out_point_precond_diag_num_alloc,
      out_point_precond_tril,
      out_point_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar