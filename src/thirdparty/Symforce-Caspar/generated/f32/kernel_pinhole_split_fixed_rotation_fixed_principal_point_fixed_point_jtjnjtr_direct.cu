#include "kernel_pinhole_split_fixed_rotation_fixed_principal_point_fixed_point_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedRotationFixedPrincipalPointFixedPointJtjnjtrDirectKernel(
        float* translation_njtr,
        unsigned int translation_njtr_num_alloc,
        SharedIndex* translation_njtr_indices,
        float* translation_jac,
        unsigned int translation_jac_num_alloc,
        float* focal_njtr,
        unsigned int focal_njtr_num_alloc,
        SharedIndex* focal_njtr_indices,
        float* focal_jac,
        unsigned int focal_jac_num_alloc,
        float* const out_translation_njtr,
        unsigned int out_translation_njtr_num_alloc,
        float* const out_focal_njtr,
        unsigned int out_focal_njtr_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex translation_njtr_indices_loc[1024];
  translation_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? translation_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex focal_njtr_indices_loc[1024];
  focal_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(translation_jac,
                                         0 * translation_jac_num_alloc,
                                         global_thread_idx,
                                         r0,
                                         r1,
                                         r2,
                                         r3);
  };
  LoadShared<2, float, float>(focal_njtr,
                              0 * focal_njtr_num_alloc,
                              focal_njtr_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared,
                       focal_njtr_indices_loc[threadIdx.x].target,
                       r4,
                       r5);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(
        focal_jac, 0 * focal_jac_num_alloc, global_thread_idx, r6, r7, r8, r9);
    r10 = fmaf(r5, r9, r4 * r7);
    r4 = fmaf(r4, r6, r5 * r8);
    r5 = fmaf(r0, r4, r1 * r10);
    r11 = fmaf(r2, r4, r3 * r10);
    ReadIdx2<1024, float, float, float2>(translation_jac,
                                         4 * translation_jac_num_alloc,
                                         global_thread_idx,
                                         r12,
                                         r13);
    r4 = fmaf(r12, r4, r13 * r10);
    WriteSum3<float, float>((float*)inout_shared, r5, r11, r4);
  };
  FlushSumShared<3, float>(out_translation_njtr,
                           0 * out_translation_njtr_num_alloc,
                           translation_njtr_indices_loc,
                           (float*)inout_shared);
  LoadShared<3, float, float>(translation_njtr,
                              0 * translation_njtr_num_alloc,
                              translation_njtr_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       translation_njtr_indices_loc[threadIdx.x].target,
                       r4,
                       r11,
                       r5);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r3 = fmaf(r11, r3, r4 * r1);
    r3 = fmaf(r5, r13, r3);
    r2 = fmaf(r11, r2, r4 * r0);
    r2 = fmaf(r5, r12, r2);
    r6 = fmaf(r6, r2, r7 * r3);
    r2 = fmaf(r8, r2, r9 * r3);
    WriteSum2<float, float>((float*)inout_shared, r6, r2);
  };
  FlushSumShared<2, float>(out_focal_njtr,
                           0 * out_focal_njtr_num_alloc,
                           focal_njtr_indices_loc,
                           (float*)inout_shared);
}

void PinholeSplitFixedRotationFixedPrincipalPointFixedPointJtjnjtrDirect(
    float* translation_njtr,
    unsigned int translation_njtr_num_alloc,
    SharedIndex* translation_njtr_indices,
    float* translation_jac,
    unsigned int translation_jac_num_alloc,
    float* focal_njtr,
    unsigned int focal_njtr_num_alloc,
    SharedIndex* focal_njtr_indices,
    float* focal_jac,
    unsigned int focal_jac_num_alloc,
    float* const out_translation_njtr,
    unsigned int out_translation_njtr_num_alloc,
    float* const out_focal_njtr,
    unsigned int out_focal_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitFixedRotationFixedPrincipalPointFixedPointJtjnjtrDirectKernel<<<
      n_blocks,
      1024>>>(translation_njtr,
              translation_njtr_num_alloc,
              translation_njtr_indices,
              translation_jac,
              translation_jac_num_alloc,
              focal_njtr,
              focal_njtr_num_alloc,
              focal_njtr_indices,
              focal_jac,
              focal_jac_num_alloc,
              out_translation_njtr,
              out_translation_njtr_num_alloc,
              out_focal_njtr,
              out_focal_njtr_num_alloc,
              problem_size);
}

}  // namespace caspar