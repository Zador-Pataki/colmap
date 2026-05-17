#include "kernel_pinhole_split_fixed_focal_fixed_principal_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedFocalFixedPrincipalPointResJacFirstKernel(
        double* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
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
      r61, r62, r63, r64, r65, r66, r67, r68, r69;

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
    ReadIdx2<1024, double, double, double2>(
        focal, 0 * focal_num_alloc, global_thread_idx, r29, r30);
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
    r3 = r29 * r3;
    r4 = fma(r32, r3, r4);
    r38 = fma(r0, r4, r1 * r5);
    ReadIdx1<1024, double, double, double>(
        weight_loss, 6 * weight_loss_num_alloc, global_thread_idx, r42);
    r43 = 0.00000000000000000e+00;
    r42 = fmax(r42, r43);
    r44 = sqrt(r42);
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 2 * weight_loss_num_alloc, global_thread_idx, r45, r46);
    r4 = fma(r45, r4, r46 * r5);
    r5 = fma(r4, r4, r38 * r38);
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 4 * weight_loss_num_alloc, global_thread_idx, r47, r48);
    r49 = 5.00000000000000000e-01;
    r48 = fmax(r48, r31);
    r50 = r48 * r48;
    r51 = r6 * r48;
    r52 = r13 * r48;
    r53 = fmax(r31, r5);
    r54 = sqrt(r53);
    r52 = fma(r54, r52, r48 * r51);
    r52 = r5 <= r50 ? r5 : r52;
    r51 = 2.50000000000000000e+00;
    r54 = r48 * r48;
    r55 = 1.0 / r50;
    r55 = fma(r5, r55, r26);
    r56 = log(r55);
    r54 = r54 * r56;
    r52 = r47 < r51 ? r54 : r52;
    r54 = 1.50000000000000000e+00;
    r56 = r13 * r48;
    r57 = sqrt(r55);
    r57 = r6 + r57;
    r56 = r56 * r48;
    r56 = r56 * r57;
    r52 = r47 < r54 ? r56 : r52;
    r52 = r47 < r49 ? r5 : r52;
    r56 = fmax(r43, r52);
    r57 = 1.0 / r53;
    r57 = r42 * r57;
    r58 = r56 * r57;
    r59 = sqrt(r58);
    r59 = r5 <= r31 ? r44 : r59;
    r44 = r38 * r59;
    r60 = r4 * r59;
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r44, r60);
    r60 = r38 * r38;
    r60 = r60 * r59;
    r44 = r4 * r4;
    r44 = r44 * r59;
    r44 = fma(r59, r44, r59 * r60);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r44);
  if (global_thread_idx < problem_size) {
    r56 = r42 * r56;
    r42 = -5.00000000000000000e-01;
    r44 = -1.00000000000000008e-15;
    r44 = r44 + r5;
    r44 = copysign(1.0, r44);
    r44 = r26 + r44;
    r60 = r53 * r53;
    r60 = 1.0 / r60;
    r56 = r56 * r42;
    r56 = r56 * r44;
    r56 = r56 * r60;
    r60 = r13 * r4;
    r42 = r15 * r10;
    r42 = r42 * r20;
    r22 = r42 + r22;
    r20 = r6 * r37;
    r61 = r24 + r20;
    r11 = r11 * r11;
    r62 = r14 * r14;
    r62 = r62 * r6;
    r63 = r11 + r62;
    r64 = r61 + r63;
    r64 = fma(r9, r64, r18 * r22);
    r22 = r6 * r64;
    r39 = r39 * r39;
    r39 = 1.0 / r39;
    r22 = r22 * r39;
    r15 = r15 * r21;
    r12 = r12 + r15;
    r12 = fma(r9, r34, r18 * r12);
    r65 = r29 * r12;
    r65 = fma(r32, r65, r3 * r22);
    r14 = r14 * r14;
    r22 = r6 * r11;
    r66 = r14 + r22;
    r61 = r61 + r66;
    r61 = fma(r18, r61, r9 * r23);
    r7 = r30 * r7;
    r7 = r7 * r6;
    r7 = r7 * r39;
    r61 = fma(r64, r7, r61 * r40);
    r30 = fma(r46, r61, r45 * r65);
    r61 = fma(r1, r61, r0 * r65);
    r65 = r13 * r38;
    r60 = fma(r61, r65, r30 * r60);
    r67 = r49 * r48;
    r53 = rsqrt(r53);
    r67 = r67 * r44;
    r67 = r67 * r53;
    r53 = r60 * r67;
    r53 = r5 <= r50 ? r60 : r53;
    r44 = 1.0 / r55;
    r68 = r60 * r44;
    r53 = r47 < r51 ? r68 : r53;
    r55 = rsqrt(r55);
    r68 = r60 * r55;
    r53 = r47 < r54 ? r68 : r53;
    r53 = r47 < r49 ? r60 : r53;
    r68 = r49 * r53;
    r52 = copysign(1.0, r52);
    r52 = r26 + r52;
    r57 = r52 * r57;
    r68 = fma(r57, r68, r60 * r56);
    r68 = r49 * r68;
    r58 = rsqrt(r58);
    r68 = r68 * r58;
    r68 = r5 <= r31 ? r43 : r68;
    r60 = r6 * r38;
    r52 = 2.50000000000000000e-01;
    r26 = r5 <= r50 ? r43 : r43;
    r26 = r47 < r51 ? r43 : r26;
    r26 = r47 < r54 ? r43 : r26;
    r26 = r47 < r49 ? r43 : r26;
    r26 = r52 * r26;
    r26 = r26 * r58;
    r26 = r26 * r57;
    r26 = r5 <= r31 ? r43 : r26;
    r60 = r60 * r26;
    r52 = fma(r38, r68, r60);
    r52 = fma(r59, r61, r52);
    r61 = r6 * r4;
    r61 = r61 * r26;
    r68 = fma(r4, r68, r61);
    r68 = fma(r59, r30, r68);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r52, r68);
    r30 = r13 * r4;
    r26 = r6 * r24;
    r69 = r37 + r26;
    r66 = r66 + r69;
    r66 = fma(r8, r66, r18 * r36);
    r42 = r33 + r42;
    r42 = fma(r8, r42, r18 * r17);
    r42 = fma(r42, r40, r66 * r7);
    r21 = r10 * r21;
    r35 = r35 + r21;
    r11 = r14 + r11;
    r11 = r11 + r20;
    r11 = r11 + r26;
    r11 = fma(r18, r11, r8 * r35);
    r18 = r29 * r11;
    r35 = r6 * r66;
    r35 = r35 * r39;
    r35 = fma(r3, r35, r32 * r18);
    r18 = fma(r45, r35, r46 * r42);
    r35 = fma(r0, r35, r1 * r42);
    r30 = fma(r35, r65, r18 * r30);
    r42 = r30 * r67;
    r42 = r5 <= r50 ? r30 : r42;
    r26 = r30 * r44;
    r42 = r47 < r51 ? r26 : r42;
    r26 = r30 * r55;
    r42 = r47 < r54 ? r26 : r42;
    r42 = r47 < r49 ? r30 : r42;
    r26 = r49 * r42;
    r26 = fma(r57, r26, r30 * r56);
    r26 = r49 * r26;
    r26 = r26 * r58;
    r26 = r5 <= r31 ? r43 : r26;
    r30 = fma(r38, r26, r60);
    r30 = fma(r59, r35, r30);
    r26 = fma(r4, r26, r61);
    r26 = fma(r59, r18, r26);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r30, r26);
    r18 = r13 * r4;
    r21 = r41 + r21;
    r21 = fma(r9, r21, r8 * r19);
    r15 = r2 + r15;
    r69 = r63 + r69;
    r69 = fma(r8, r69, r9 * r15);
    r69 = fma(r69, r40, r21 * r7);
    r15 = r6 * r21;
    r15 = r15 * r39;
    r37 = r24 + r37;
    r37 = r37 + r62;
    r37 = r37 + r22;
    r37 = fma(r9, r37, r8 * r16);
    r9 = r29 * r37;
    r9 = fma(r32, r9, r3 * r15);
    r15 = fma(r45, r9, r46 * r69);
    r9 = fma(r0, r9, r1 * r69);
    r18 = fma(r9, r65, r15 * r18);
    r69 = r18 * r67;
    r69 = r5 <= r50 ? r18 : r69;
    r8 = r18 * r44;
    r69 = r47 < r51 ? r8 : r69;
    r8 = r18 * r55;
    r69 = r47 < r54 ? r8 : r69;
    r69 = r47 < r49 ? r18 : r69;
    r8 = r49 * r69;
    r18 = fma(r18, r56, r57 * r8);
    r18 = r49 * r18;
    r18 = r18 * r58;
    r18 = r5 <= r31 ? r43 : r18;
    r8 = fma(r38, r18, r60);
    r8 = fma(r59, r9, r8);
    r18 = fma(r4, r18, r61);
    r18 = fma(r59, r15, r18);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r8, r18);
    r15 = r0 * r29;
    r15 = r15 * r32;
    r9 = r45 * r29;
    r9 = r9 * r13;
    r9 = r9 * r4;
    r9 = fma(r32, r9, r65 * r15);
    r15 = r9 * r67;
    r15 = r5 <= r50 ? r9 : r15;
    r22 = r9 * r44;
    r15 = r47 < r51 ? r22 : r15;
    r22 = r9 * r55;
    r15 = r47 < r54 ? r22 : r15;
    r15 = r47 < r49 ? r9 : r15;
    r22 = r49 * r15;
    r9 = fma(r9, r56, r57 * r22);
    r9 = r49 * r9;
    r9 = r9 * r58;
    r9 = r5 <= r31 ? r43 : r9;
    r22 = fma(r38, r9, r60);
    r62 = r0 * r29;
    r62 = r62 * r59;
    r22 = fma(r32, r62, r22);
    r9 = fma(r4, r9, r61);
    r62 = r45 * r29;
    r62 = r62 * r59;
    r9 = fma(r32, r62, r9);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r22, r9);
    r62 = r1 * r40;
    r24 = r46 * r13;
    r24 = r24 * r4;
    r24 = fma(r40, r24, r65 * r62);
    r63 = r24 * r67;
    r63 = r5 <= r50 ? r24 : r63;
    r2 = r24 * r44;
    r63 = r47 < r51 ? r2 : r63;
    r2 = r24 * r55;
    r63 = r47 < r54 ? r2 : r63;
    r63 = r47 < r49 ? r24 : r63;
    r2 = r49 * r63;
    r2 = fma(r57, r2, r24 * r56);
    r2 = r49 * r2;
    r2 = r2 * r58;
    r2 = r5 <= r31 ? r43 : r2;
    r24 = fma(r38, r2, r60);
    r24 = fma(r59, r62, r24);
    r2 = fma(r4, r2, r61);
    r62 = r46 * r59;
    r2 = fma(r40, r62, r2);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r24, r2);
    r62 = r13 * r4;
    r41 = r45 * r6;
    r41 = r41 * r39;
    r41 = fma(r46, r7, r3 * r41);
    r35 = r0 * r6;
    r35 = r35 * r39;
    r35 = fma(r1, r7, r3 * r35);
    r62 = fma(r35, r65, r41 * r62);
    r20 = r62 * r67;
    r20 = r5 <= r50 ? r62 : r20;
    r14 = r62 * r44;
    r20 = r47 < r51 ? r14 : r20;
    r14 = r62 * r55;
    r20 = r47 < r54 ? r14 : r20;
    r20 = r47 < r49 ? r62 : r20;
    r14 = r49 * r20;
    r14 = fma(r57, r14, r62 * r56);
    r14 = r49 * r14;
    r14 = r14 * r58;
    r14 = r5 <= r31 ? r43 : r14;
    r62 = fma(r38, r14, r60);
    r62 = fma(r59, r35, r62);
    r14 = fma(r4, r14, r61);
    r14 = fma(r59, r41, r14);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r62, r14);
    r41 = r38 * r52;
    r35 = r6 * r59;
    r10 = r4 * r35;
    r41 = fma(r68, r10, r35 * r41);
    r33 = r38 * r30;
    r33 = fma(r35, r33, r26 * r10);
    WriteSum2<double, double>((double*)inout_shared, r41, r33);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = r38 * r8;
    r33 = fma(r18, r10, r35 * r33);
    r41 = r38 * r22;
    r41 = fma(r35, r41, r9 * r10);
    WriteSum2<double, double>((double*)inout_shared, r33, r41);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = r38 * r24;
    r41 = fma(r2, r10, r35 * r41);
    r33 = r38 * r62;
    r33 = fma(r14, r10, r35 * r33);
    WriteSum2<double, double>((double*)inout_shared, r41, r33);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = fma(r68, r68, r52 * r52);
    r41 = fma(r26, r26, r30 * r30);
    WriteSum2<double, double>((double*)inout_shared, r33, r41);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = fma(r8, r8, r18 * r18);
    r33 = fma(r9, r9, r22 * r22);
    WriteSum2<double, double>((double*)inout_shared, r41, r33);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = fma(r2, r2, r24 * r24);
    r41 = fma(r14, r14, r62 * r62);
    WriteSum2<double, double>((double*)inout_shared, r33, r41);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = fma(r68, r26, r52 * r30);
    r33 = fma(r68, r18, r52 * r8);
    WriteSum2<double, double>((double*)inout_shared, r41, r33);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = fma(r52, r22, r68 * r9);
    r41 = fma(r52, r24, r68 * r2);
    WriteSum2<double, double>((double*)inout_shared, r33, r41);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r68 = fma(r68, r14, r52 * r62);
    r41 = fma(r26, r18, r30 * r8);
    WriteSum2<double, double>((double*)inout_shared, r68, r41);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = fma(r30, r22, r26 * r9);
    r68 = fma(r26, r2, r30 * r24);
    WriteSum2<double, double>((double*)inout_shared, r41, r68);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = fma(r30, r62, r26 * r14);
    r68 = fma(r18, r9, r8 * r22);
    WriteSum2<double, double>((double*)inout_shared, r26, r68);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r68 = fma(r18, r2, r8 * r24);
    r18 = fma(r18, r14, r8 * r62);
    WriteSum2<double, double>((double*)inout_shared, r68, r18);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r18 = fma(r22, r24, r9 * r2);
    r9 = fma(r22, r62, r9 * r14);
    WriteSum2<double, double>((double*)inout_shared, r18, r9);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r14 = fma(r2, r14, r24 * r62);
    WriteSum1<double, double>((double*)inout_shared, r14);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r17 = fma(r36, r7, r17 * r40);
    r14 = r36 * r6;
    r14 = r14 * r39;
    r2 = r29 * r25;
    r2 = fma(r32, r2, r3 * r14);
    r14 = fma(r0, r2, r1 * r17);
    r9 = fma(r59, r14, r60);
    r18 = r13 * r4;
    r2 = fma(r45, r2, r46 * r17);
    r14 = fma(r14, r65, r2 * r18);
    r18 = r14 * r67;
    r18 = r5 <= r50 ? r14 : r18;
    r17 = r14 * r44;
    r18 = r47 < r51 ? r17 : r18;
    r17 = r14 * r55;
    r18 = r47 < r54 ? r17 : r18;
    r18 = r47 < r49 ? r14 : r18;
    r17 = r49 * r18;
    r14 = fma(r14, r56, r57 * r17);
    r14 = r49 * r14;
    r14 = r14 * r58;
    r14 = r5 <= r31 ? r43 : r14;
    r9 = fma(r38, r14, r9);
    r2 = fma(r59, r2, r61);
    r2 = fma(r4, r14, r2);
    WriteIdx2<1024, double, double, double2>(
        out_point_jac, 0 * out_point_jac_num_alloc, global_thread_idx, r9, r2);
    r28 = fma(r28, r40, r19 * r7);
    r14 = r19 * r6;
    r14 = r14 * r39;
    r17 = r29 * r16;
    r17 = fma(r32, r17, r3 * r14);
    r14 = fma(r0, r17, r1 * r28);
    r68 = fma(r59, r14, r60);
    r26 = r13 * r4;
    r17 = fma(r45, r17, r46 * r28);
    r14 = fma(r14, r65, r17 * r26);
    r26 = r14 * r67;
    r26 = r5 <= r50 ? r14 : r26;
    r28 = r14 * r44;
    r26 = r47 < r51 ? r28 : r26;
    r28 = r14 * r55;
    r26 = r47 < r54 ? r28 : r26;
    r26 = r47 < r49 ? r14 : r26;
    r28 = r49 * r26;
    r14 = fma(r14, r56, r57 * r28);
    r14 = r49 * r14;
    r14 = r14 * r58;
    r14 = r5 <= r31 ? r43 : r14;
    r68 = fma(r38, r14, r68);
    r17 = fma(r59, r17, r61);
    r17 = fma(r4, r14, r17);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             2 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r68,
                                             r17);
    r14 = r13 * r4;
    r28 = r27 * r6;
    r28 = r28 * r39;
    r39 = r29 * r34;
    r39 = fma(r32, r39, r3 * r28);
    r40 = fma(r23, r40, r27 * r7);
    r23 = fma(r46, r40, r45 * r39);
    r40 = fma(r1, r40, r0 * r39);
    r65 = fma(r40, r65, r23 * r14);
    r67 = r65 * r67;
    r67 = r5 <= r50 ? r65 : r67;
    r44 = r65 * r44;
    r67 = r47 < r51 ? r44 : r67;
    r55 = r65 * r55;
    r67 = r47 < r54 ? r55 : r67;
    r67 = r47 < r49 ? r65 : r67;
    r47 = r49 * r67;
    r56 = fma(r65, r56, r57 * r47);
    r56 = r49 * r56;
    r56 = r56 * r58;
    r56 = r5 <= r31 ? r43 : r56;
    r60 = fma(r38, r56, r60);
    r60 = fma(r59, r40, r60);
    r23 = fma(r59, r23, r61);
    r23 = fma(r4, r56, r23);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             4 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r60,
                                             r23);
    r56 = r38 * r9;
    r56 = fma(r2, r10, r35 * r56);
    r61 = r38 * r68;
    r61 = fma(r35, r61, r17 * r10);
    WriteSum2<double, double>((double*)inout_shared, r56, r61);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r61 = r38 * r60;
    r10 = fma(r23, r10, r35 * r61);
    WriteSum1<double, double>((double*)inout_shared, r10);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r10 = fma(r2, r2, r9 * r9);
    r61 = fma(r17, r17, r68 * r68);
    WriteSum2<double, double>((double*)inout_shared, r10, r61);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r61 = fma(r23, r23, r60 * r60);
    WriteSum1<double, double>((double*)inout_shared, r61);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r61 = fma(r9, r68, r2 * r17);
    r2 = fma(r2, r23, r9 * r60);
    WriteSum2<double, double>((double*)inout_shared, r61, r2);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = fma(r17, r23, r68 * r60);
    WriteSum1<double, double>((double*)inout_shared, r23);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void PinholeSplitFixedFocalFixedPrincipalPointResJacFirst(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
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
  PinholeSplitFixedFocalFixedPrincipalPointResJacFirstKernel<<<n_blocks,
                                                               1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
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