#include "kernel_pinhole_split_fixed_principal_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedPrincipalPointResJacFirstKernel(
        double* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        double* focal,
        unsigned int focal_num_alloc,
        SharedIndex* focal_indices,
        double* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* weight_loss,
        unsigned int weight_loss_num_alloc,
        double* principal_point,
        unsigned int principal_point_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* const out_rTr,
        double* out_pose_jac,
        unsigned int out_pose_jac_num_alloc,
        double* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        double* const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        double* const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
        double* out_focal_jac,
        unsigned int out_focal_jac_num_alloc,
        double* const out_focal_njtr,
        unsigned int out_focal_njtr_num_alloc,
        double* const out_focal_precond_diag,
        unsigned int out_focal_precond_diag_num_alloc,
        double* const out_focal_precond_tril,
        unsigned int out_focal_precond_tril_num_alloc,
        double* out_point_jac,
        unsigned int out_point_jac_num_alloc,
        double* const out_point_njtr,
        unsigned int out_point_njtr_num_alloc,
        double* const out_point_precond_diag,
        unsigned int out_point_precond_diag_num_alloc,
        double* const out_point_precond_tril,
        unsigned int out_point_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
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

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73;

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
  };
  LoadShared<2, double, double>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r3, r7);
  };
  __syncthreads();
  LoadShared<2, double, double>(
      point, 0 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r8, r9);
  };
  __syncthreads();
  LoadShared<2, double, double>(
      pose, 2 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r10, r11);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r12 = r10 * r11;
    r13 = 2.00000000000000000e+00;
    r12 = r12 * r13;
  };
  LoadShared<2, double, double>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r14, r15);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r16 = r14 * r15;
    r16 = r16 * r13;
    r17 = r12 + r16;
    r7 = fma(r8, r17, r7);
  };
  LoadShared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r18);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r19 = r15 * r10;
    r19 = r19 * r13;
    r20 = -2.00000000000000000e+00;
    r21 = r14 * r20;
    r22 = r11 * r21;
    r23 = r19 + r22;
    r24 = r10 * r10;
    r25 = r20 * r24;
    r26 = 1.00000000000000000e+00;
    r27 = fma(r14, r21, r26);
    r28 = r25 + r27;
    r7 = fma(r18, r23, r7);
    r7 = fma(r9, r28, r7);
  };
  LoadShared<2, double, double>(
      focal, 0 * focal_num_alloc, focal_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, focal_indices_loc[threadIdx.x].target, r29, r30);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r31 = 1.00000000000000008e-15;
  };
  LoadShared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r32);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r33 = r14 * r11;
    r33 = r33 * r13;
    r19 = r19 + r33;
    r32 = fma(r9, r19, r32);
    r34 = r14 * r10;
    r34 = r34 * r13;
    r35 = r15 * r11;
    r35 = r35 * r20;
    r36 = r34 + r35;
    r37 = r15 * r15;
    r38 = r20 * r37;
    r27 = r38 + r27;
    r32 = fma(r8, r36, r32);
    r32 = fma(r18, r27, r32);
    r39 = copysign(1.0, r32);
    r39 = fma(r31, r39, r32);
    r32 = 1.0 / r39;
    r40 = r30 * r32;
    r5 = fma(r7, r40, r5);
    r4 = fma(r4, r6, r2);
    r2 = r10 * r11;
    r2 = r2 * r20;
    r16 = r16 + r2;
    r3 = fma(r9, r16, r3);
    r41 = r15 * r11;
    r41 = r41 * r13;
    r34 = r34 + r41;
    r25 = r26 + r25;
    r25 = r25 + r38;
    r3 = fma(r18, r34, r3);
    r3 = fma(r8, r25, r3);
    r38 = r29 * r3;
    r4 = fma(r32, r38, r4);
    r42 = fma(r0, r4, r1 * r5);
    ReadIdx1<1024, double, double, double>(
        weight_loss, 6 * weight_loss_num_alloc, global_thread_idx, r43);
    r44 = 0.00000000000000000e+00;
    r43 = fmax(r43, r44);
    r45 = sqrt(r43);
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 2 * weight_loss_num_alloc, global_thread_idx, r46, r47);
    r4 = fma(r46, r4, r47 * r5);
    r5 = fma(r4, r4, r42 * r42);
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 4 * weight_loss_num_alloc, global_thread_idx, r48, r49);
    r50 = 5.00000000000000000e-01;
    r49 = fmax(r49, r31);
    r51 = r49 * r49;
    r52 = r6 * r49;
    r53 = r13 * r49;
    r54 = fmax(r31, r5);
    r55 = sqrt(r54);
    r53 = fma(r55, r53, r49 * r52);
    r53 = r5 <= r51 ? r5 : r53;
    r52 = 2.50000000000000000e+00;
    r55 = r49 * r49;
    r56 = 1.0 / r51;
    r56 = fma(r5, r56, r26);
    r57 = log(r56);
    r55 = r55 * r57;
    r53 = r48 < r52 ? r55 : r53;
    r55 = 1.50000000000000000e+00;
    r57 = r13 * r49;
    r58 = sqrt(r56);
    r58 = r6 + r58;
    r57 = r57 * r49;
    r57 = r57 * r58;
    r53 = r48 < r55 ? r57 : r53;
    r53 = r48 < r50 ? r5 : r53;
    r57 = fmax(r44, r53);
    r58 = 1.0 / r54;
    r58 = r43 * r58;
    r59 = r57 * r58;
    r60 = sqrt(r59);
    r60 = r5 <= r31 ? r45 : r60;
    r45 = r42 * r60;
    r61 = r4 * r60;
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r45, r61);
    r61 = r42 * r42;
    r61 = r61 * r60;
    r45 = r4 * r4;
    r45 = r45 * r60;
    r45 = fma(r60, r45, r60 * r61);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r45);
  if (global_thread_idx < problem_size) {
    r57 = r43 * r57;
    r43 = -5.00000000000000000e-01;
    r45 = -1.00000000000000008e-15;
    r45 = r45 + r5;
    r45 = copysign(1.0, r45);
    r45 = r26 + r45;
    r61 = r54 * r54;
    r61 = 1.0 / r61;
    r57 = r57 * r43;
    r57 = r57 * r45;
    r57 = r57 * r61;
    r61 = r13 * r4;
    r43 = r15 * r10;
    r43 = r43 * r20;
    r22 = r43 + r22;
    r20 = r6 * r37;
    r62 = r24 + r20;
    r11 = r11 * r11;
    r63 = r14 * r14;
    r63 = r63 * r6;
    r64 = r11 + r63;
    r65 = r62 + r64;
    r65 = fma(r9, r65, r18 * r22);
    r22 = r6 * r65;
    r39 = r39 * r39;
    r39 = 1.0 / r39;
    r22 = r22 * r39;
    r15 = r15 * r21;
    r12 = r12 + r15;
    r12 = fma(r9, r34, r18 * r12);
    r66 = r29 * r12;
    r66 = fma(r32, r66, r38 * r22);
    r14 = r14 * r14;
    r22 = r6 * r11;
    r67 = r14 + r22;
    r62 = r62 + r67;
    r62 = fma(r18, r62, r9 * r23);
    r30 = r30 * r7;
    r30 = r30 * r6;
    r30 = r30 * r39;
    r62 = fma(r65, r30, r62 * r40);
    r68 = fma(r47, r62, r46 * r66);
    r62 = fma(r1, r62, r0 * r66);
    r66 = r13 * r42;
    r61 = fma(r62, r66, r68 * r61);
    r69 = r50 * r49;
    r54 = rsqrt(r54);
    r69 = r69 * r45;
    r69 = r69 * r54;
    r54 = r61 * r69;
    r54 = r5 <= r51 ? r61 : r54;
    r45 = 1.0 / r56;
    r70 = r61 * r45;
    r54 = r48 < r52 ? r70 : r54;
    r56 = rsqrt(r56);
    r70 = r61 * r56;
    r54 = r48 < r55 ? r70 : r54;
    r54 = r48 < r50 ? r61 : r54;
    r70 = r50 * r54;
    r53 = copysign(1.0, r53);
    r53 = r26 + r53;
    r58 = r53 * r58;
    r70 = fma(r58, r70, r61 * r57);
    r70 = r50 * r70;
    r59 = rsqrt(r59);
    r70 = r70 * r59;
    r70 = r5 <= r31 ? r44 : r70;
    r61 = r6 * r42;
    r53 = 2.50000000000000000e-01;
    r26 = r5 <= r51 ? r44 : r44;
    r26 = r48 < r52 ? r44 : r26;
    r26 = r48 < r55 ? r44 : r26;
    r26 = r48 < r50 ? r44 : r26;
    r26 = r53 * r26;
    r26 = r26 * r59;
    r26 = r26 * r58;
    r26 = r5 <= r31 ? r44 : r26;
    r61 = r61 * r26;
    r53 = fma(r42, r70, r61);
    r53 = fma(r60, r62, r53);
    r62 = r6 * r4;
    r62 = r62 * r26;
    r70 = fma(r4, r70, r62);
    r70 = fma(r60, r68, r70);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r53, r70);
    r68 = r13 * r4;
    r26 = r6 * r24;
    r71 = r37 + r26;
    r67 = r67 + r71;
    r67 = fma(r8, r67, r18 * r36);
    r43 = r33 + r43;
    r43 = fma(r8, r43, r18 * r17);
    r43 = fma(r43, r40, r67 * r30);
    r21 = r10 * r21;
    r35 = r35 + r21;
    r11 = r14 + r11;
    r11 = r11 + r20;
    r11 = r11 + r26;
    r11 = fma(r18, r11, r8 * r35);
    r18 = r29 * r11;
    r35 = r6 * r67;
    r35 = r35 * r39;
    r35 = fma(r38, r35, r32 * r18);
    r18 = fma(r46, r35, r47 * r43);
    r35 = fma(r0, r35, r1 * r43);
    r68 = fma(r35, r66, r18 * r68);
    r43 = r68 * r69;
    r43 = r5 <= r51 ? r68 : r43;
    r26 = r68 * r45;
    r43 = r48 < r52 ? r26 : r43;
    r26 = r68 * r56;
    r43 = r48 < r55 ? r26 : r43;
    r43 = r48 < r50 ? r68 : r43;
    r26 = r50 * r43;
    r26 = fma(r58, r26, r68 * r57);
    r26 = r50 * r26;
    r26 = r26 * r59;
    r26 = r5 <= r31 ? r44 : r26;
    r68 = fma(r42, r26, r61);
    r68 = fma(r60, r35, r68);
    r26 = fma(r4, r26, r62);
    r26 = fma(r60, r18, r26);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r68, r26);
    r18 = r13 * r4;
    r21 = r41 + r21;
    r21 = fma(r9, r21, r8 * r19);
    r15 = r2 + r15;
    r71 = r64 + r71;
    r71 = fma(r8, r71, r9 * r15);
    r71 = fma(r71, r40, r21 * r30);
    r15 = r6 * r21;
    r15 = r15 * r39;
    r37 = r24 + r37;
    r37 = r37 + r63;
    r37 = r37 + r22;
    r37 = fma(r9, r37, r8 * r16);
    r9 = r29 * r37;
    r9 = fma(r32, r9, r38 * r15);
    r15 = fma(r46, r9, r47 * r71);
    r9 = fma(r0, r9, r1 * r71);
    r18 = fma(r9, r66, r15 * r18);
    r71 = r18 * r69;
    r71 = r5 <= r51 ? r18 : r71;
    r8 = r18 * r45;
    r71 = r48 < r52 ? r8 : r71;
    r8 = r18 * r56;
    r71 = r48 < r55 ? r8 : r71;
    r71 = r48 < r50 ? r18 : r71;
    r8 = r50 * r71;
    r18 = fma(r18, r57, r58 * r8);
    r18 = r50 * r18;
    r18 = r18 * r59;
    r18 = r5 <= r31 ? r44 : r18;
    r8 = fma(r42, r18, r61);
    r8 = fma(r60, r9, r8);
    r18 = fma(r4, r18, r62);
    r18 = fma(r60, r15, r18);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r8, r18);
    r15 = r32 * r66;
    r9 = r0 * r15;
    r22 = r29 * r46;
    r22 = r22 * r13;
    r22 = r22 * r4;
    r22 = fma(r32, r22, r29 * r9);
    r63 = r22 * r69;
    r63 = r5 <= r51 ? r22 : r63;
    r24 = r22 * r45;
    r63 = r48 < r52 ? r24 : r63;
    r24 = r22 * r56;
    r63 = r48 < r55 ? r24 : r63;
    r63 = r48 < r50 ? r22 : r63;
    r24 = r50 * r63;
    r22 = fma(r22, r57, r58 * r24);
    r22 = r50 * r22;
    r22 = r22 * r59;
    r22 = r5 <= r31 ? r44 : r22;
    r24 = fma(r42, r22, r61);
    r64 = r29 * r0;
    r64 = r64 * r60;
    r24 = fma(r32, r64, r24);
    r22 = fma(r4, r22, r62);
    r64 = r29 * r46;
    r64 = r64 * r60;
    r22 = fma(r32, r64, r22);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r24, r22);
    r64 = r1 * r40;
    r2 = r47 * r13;
    r2 = r2 * r4;
    r2 = fma(r40, r2, r66 * r64);
    r41 = r2 * r69;
    r41 = r5 <= r51 ? r2 : r41;
    r35 = r2 * r45;
    r41 = r48 < r52 ? r35 : r41;
    r35 = r2 * r56;
    r41 = r48 < r55 ? r35 : r41;
    r41 = r48 < r50 ? r2 : r41;
    r35 = r50 * r41;
    r35 = fma(r58, r35, r2 * r57);
    r35 = r50 * r35;
    r35 = r35 * r59;
    r35 = r5 <= r31 ? r44 : r35;
    r2 = fma(r42, r35, r61);
    r2 = fma(r60, r64, r2);
    r35 = fma(r4, r35, r62);
    r64 = r47 * r60;
    r35 = fma(r40, r64, r35);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r2, r35);
    r64 = r13 * r4;
    r20 = r46 * r6;
    r20 = r20 * r39;
    r20 = fma(r47, r30, r38 * r20);
    r14 = r0 * r6;
    r14 = r14 * r39;
    r14 = fma(r1, r30, r38 * r14);
    r64 = fma(r14, r66, r20 * r64);
    r10 = r64 * r69;
    r10 = r5 <= r51 ? r64 : r10;
    r33 = r64 * r45;
    r10 = r48 < r52 ? r33 : r10;
    r33 = r64 * r56;
    r10 = r48 < r55 ? r33 : r10;
    r10 = r48 < r50 ? r64 : r10;
    r33 = r50 * r10;
    r33 = fma(r58, r33, r64 * r57);
    r33 = r50 * r33;
    r33 = r33 * r59;
    r33 = r5 <= r31 ? r44 : r33;
    r64 = fma(r42, r33, r61);
    r64 = fma(r60, r14, r64);
    r33 = fma(r4, r33, r62);
    r33 = fma(r60, r20, r33);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r64, r33);
    r20 = r42 * r53;
    r14 = r6 * r60;
    r72 = r4 * r14;
    r20 = fma(r70, r72, r14 * r20);
    r73 = r42 * r68;
    r73 = fma(r14, r73, r26 * r72);
    WriteSum2<double, double>((double*)inout_shared, r20, r73);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = r42 * r8;
    r73 = fma(r18, r72, r14 * r73);
    r20 = r42 * r24;
    r20 = fma(r14, r20, r22 * r72);
    WriteSum2<double, double>((double*)inout_shared, r73, r20);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r20 = r42 * r2;
    r20 = fma(r35, r72, r14 * r20);
    r73 = r42 * r64;
    r73 = fma(r33, r72, r14 * r73);
    WriteSum2<double, double>((double*)inout_shared, r20, r73);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fma(r70, r70, r53 * r53);
    r20 = fma(r26, r26, r68 * r68);
    WriteSum2<double, double>((double*)inout_shared, r73, r20);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r20 = fma(r8, r8, r18 * r18);
    r73 = fma(r22, r22, r24 * r24);
    WriteSum2<double, double>((double*)inout_shared, r20, r73);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fma(r35, r35, r2 * r2);
    r20 = fma(r33, r33, r64 * r64);
    WriteSum2<double, double>((double*)inout_shared, r73, r20);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r20 = fma(r70, r26, r53 * r68);
    r73 = fma(r70, r18, r53 * r8);
    WriteSum2<double, double>((double*)inout_shared, r20, r73);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fma(r53, r24, r70 * r22);
    r20 = fma(r53, r2, r70 * r35);
    WriteSum2<double, double>((double*)inout_shared, r73, r20);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r70 = fma(r70, r33, r53 * r64);
    r20 = fma(r26, r18, r68 * r8);
    WriteSum2<double, double>((double*)inout_shared, r70, r20);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r20 = fma(r68, r24, r26 * r22);
    r70 = fma(r26, r35, r68 * r2);
    WriteSum2<double, double>((double*)inout_shared, r20, r70);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = fma(r68, r64, r26 * r33);
    r70 = fma(r18, r22, r8 * r24);
    WriteSum2<double, double>((double*)inout_shared, r26, r70);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r70 = fma(r18, r35, r8 * r2);
    r18 = fma(r18, r33, r8 * r64);
    WriteSum2<double, double>((double*)inout_shared, r70, r18);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r18 = fma(r24, r2, r22 * r35);
    r22 = fma(r24, r64, r22 * r33);
    WriteSum2<double, double>((double*)inout_shared, r18, r22);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = fma(r35, r33, r2 * r64);
    WriteSum1<double, double>((double*)inout_shared, r33);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = r46 * r13;
    r33 = r33 * r3;
    r33 = r33 * r4;
    r33 = fma(r32, r33, r3 * r9);
    r9 = r33 * r69;
    r9 = r5 <= r51 ? r33 : r9;
    r35 = r33 * r45;
    r9 = r48 < r52 ? r35 : r9;
    r35 = r33 * r56;
    r9 = r48 < r55 ? r35 : r9;
    r9 = r48 < r50 ? r33 : r9;
    r35 = r50 * r9;
    r35 = fma(r58, r35, r33 * r57);
    r35 = r50 * r35;
    r35 = r35 * r59;
    r35 = r5 <= r31 ? r44 : r35;
    r33 = fma(r42, r35, r61);
    r22 = r0 * r3;
    r22 = r22 * r60;
    r33 = fma(r32, r22, r33);
    r35 = fma(r4, r35, r62);
    r22 = r46 * r3;
    r22 = r22 * r60;
    r35 = fma(r32, r22, r35);
    WriteIdx2<1024, double, double, double2>(out_focal_jac,
                                             0 * out_focal_jac_num_alloc,
                                             global_thread_idx,
                                             r33,
                                             r35);
    r22 = r1 * r7;
    r18 = r47 * r13;
    r18 = r18 * r7;
    r18 = r18 * r4;
    r18 = fma(r32, r18, r15 * r22);
    r22 = r18 * r69;
    r22 = r5 <= r51 ? r18 : r22;
    r15 = r18 * r45;
    r22 = r48 < r52 ? r15 : r22;
    r15 = r18 * r56;
    r22 = r48 < r55 ? r15 : r22;
    r22 = r48 < r50 ? r18 : r22;
    r15 = r50 * r22;
    r15 = fma(r58, r15, r18 * r57);
    r15 = r50 * r15;
    r15 = r15 * r59;
    r15 = r5 <= r31 ? r44 : r15;
    r18 = fma(r42, r15, r61);
    r70 = r1 * r7;
    r70 = r70 * r60;
    r18 = fma(r32, r70, r18);
    r15 = fma(r4, r15, r62);
    r70 = r47 * r7;
    r70 = r70 * r60;
    r15 = fma(r32, r70, r15);
    WriteIdx2<1024, double, double, double2>(out_focal_jac,
                                             2 * out_focal_jac_num_alloc,
                                             global_thread_idx,
                                             r18,
                                             r15);
    r70 = r42 * r33;
    r70 = fma(r14, r70, r35 * r72);
    r26 = r42 * r18;
    r26 = fma(r14, r26, r15 * r72);
    WriteSum2<double, double>((double*)inout_shared, r70, r26);
  };
  FlushSumShared<2, double>(out_focal_njtr,
                            0 * out_focal_njtr_num_alloc,
                            focal_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = fma(r33, r33, r35 * r35);
    r70 = fma(r15, r15, r18 * r18);
    WriteSum2<double, double>((double*)inout_shared, r26, r70);
  };
  FlushSumShared<2, double>(out_focal_precond_diag,
                            0 * out_focal_precond_diag_num_alloc,
                            focal_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r15 = fma(r35, r15, r33 * r18);
    WriteSum1<double, double>((double*)inout_shared, r15);
  };
  FlushSumShared<1, double>(out_focal_precond_tril,
                            0 * out_focal_precond_tril_num_alloc,
                            focal_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r17 = fma(r36, r30, r17 * r40);
    r15 = r36 * r6;
    r15 = r15 * r39;
    r35 = r29 * r25;
    r35 = fma(r32, r35, r38 * r15);
    r15 = fma(r0, r35, r1 * r17);
    r70 = fma(r60, r15, r61);
    r26 = r13 * r4;
    r35 = fma(r46, r35, r47 * r17);
    r15 = fma(r15, r66, r35 * r26);
    r26 = r15 * r69;
    r26 = r5 <= r51 ? r15 : r26;
    r17 = r15 * r45;
    r26 = r48 < r52 ? r17 : r26;
    r17 = r15 * r56;
    r26 = r48 < r55 ? r17 : r26;
    r26 = r48 < r50 ? r15 : r26;
    r17 = r50 * r26;
    r15 = fma(r15, r57, r58 * r17);
    r15 = r50 * r15;
    r15 = r15 * r59;
    r15 = r5 <= r31 ? r44 : r15;
    r70 = fma(r42, r15, r70);
    r35 = fma(r60, r35, r62);
    r35 = fma(r4, r15, r35);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             0 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r70,
                                             r35);
    r28 = fma(r28, r40, r19 * r30);
    r15 = r19 * r6;
    r15 = r15 * r39;
    r17 = r29 * r16;
    r17 = fma(r32, r17, r38 * r15);
    r15 = fma(r0, r17, r1 * r28);
    r20 = fma(r60, r15, r61);
    r73 = r13 * r4;
    r17 = fma(r46, r17, r47 * r28);
    r15 = fma(r15, r66, r17 * r73);
    r73 = r15 * r69;
    r73 = r5 <= r51 ? r15 : r73;
    r28 = r15 * r45;
    r73 = r48 < r52 ? r28 : r73;
    r28 = r15 * r56;
    r73 = r48 < r55 ? r28 : r73;
    r73 = r48 < r50 ? r15 : r73;
    r28 = r50 * r73;
    r15 = fma(r15, r57, r58 * r28);
    r15 = r50 * r15;
    r15 = r15 * r59;
    r15 = r5 <= r31 ? r44 : r15;
    r20 = fma(r42, r15, r20);
    r17 = fma(r60, r17, r62);
    r17 = fma(r4, r15, r17);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             2 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r20,
                                             r17);
    r15 = r13 * r4;
    r28 = r27 * r6;
    r28 = r28 * r39;
    r39 = r29 * r34;
    r39 = fma(r32, r39, r38 * r28);
    r40 = fma(r23, r40, r27 * r30);
    r23 = fma(r47, r40, r46 * r39);
    r40 = fma(r1, r40, r0 * r39);
    r66 = fma(r40, r66, r23 * r15);
    r69 = r66 * r69;
    r69 = r5 <= r51 ? r66 : r69;
    r45 = r66 * r45;
    r69 = r48 < r52 ? r45 : r69;
    r56 = r66 * r56;
    r69 = r48 < r55 ? r56 : r69;
    r69 = r48 < r50 ? r66 : r69;
    r48 = r50 * r69;
    r57 = fma(r66, r57, r58 * r48);
    r57 = r50 * r57;
    r57 = r57 * r59;
    r57 = r5 <= r31 ? r44 : r57;
    r61 = fma(r42, r57, r61);
    r61 = fma(r60, r40, r61);
    r23 = fma(r60, r23, r62);
    r23 = fma(r4, r57, r23);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             4 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r61,
                                             r23);
    r57 = r42 * r70;
    r57 = fma(r35, r72, r14 * r57);
    r62 = r42 * r20;
    r62 = fma(r14, r62, r17 * r72);
    WriteSum2<double, double>((double*)inout_shared, r57, r62);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r62 = r42 * r61;
    r72 = fma(r23, r72, r14 * r62);
    WriteSum1<double, double>((double*)inout_shared, r72);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = fma(r35, r35, r70 * r70);
    r62 = fma(r17, r17, r20 * r20);
    WriteSum2<double, double>((double*)inout_shared, r72, r62);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r62 = fma(r23, r23, r61 * r61);
    WriteSum1<double, double>((double*)inout_shared, r62);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r62 = fma(r70, r20, r35 * r17);
    r35 = fma(r35, r23, r70 * r61);
    WriteSum2<double, double>((double*)inout_shared, r62, r35);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = fma(r17, r23, r20 * r61);
    WriteSum1<double, double>((double*)inout_shared, r23);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void PinholeSplitFixedPrincipalPointResJacFirst(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* focal,
    unsigned int focal_num_alloc,
    SharedIndex* focal_indices,
    double* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* weight_loss,
    unsigned int weight_loss_num_alloc,
    double* principal_point,
    unsigned int principal_point_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_rTr,
    double* out_pose_jac,
    unsigned int out_pose_jac_num_alloc,
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    double* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    double* out_focal_jac,
    unsigned int out_focal_jac_num_alloc,
    double* const out_focal_njtr,
    unsigned int out_focal_njtr_num_alloc,
    double* const out_focal_precond_diag,
    unsigned int out_focal_precond_diag_num_alloc,
    double* const out_focal_precond_tril,
    unsigned int out_focal_precond_tril_num_alloc,
    double* out_point_jac,
    unsigned int out_point_jac_num_alloc,
    double* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    double* const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    double* const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitFixedPrincipalPointResJacFirstKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
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
      principal_point,
      principal_point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_rTr,
      out_pose_jac,
      out_pose_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_pose_precond_diag,
      out_pose_precond_diag_num_alloc,
      out_pose_precond_tril,
      out_pose_precond_tril_num_alloc,
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