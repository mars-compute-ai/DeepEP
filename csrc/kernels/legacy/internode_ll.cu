#include "compiled.cuh"
#include "ibgda_device.cuh"
#include "launch.cuh"
#include "api.cuh"

namespace deep_ep::legacy {

namespace internode_ll {

template <bool use_warp_sync = false>
__forceinline__ __device__ bool is_rank_masked(int* mask_buffer_ptr, int rank) {
    if (mask_buffer_ptr == nullptr) {
        return false;
    }
    if constexpr (use_warp_sync) {
        return __shfl_sync(0xffffffff, ld_acquire_global(mask_buffer_ptr + rank), 0) != 0;
    } else {
        return ld_acquire_global(mask_buffer_ptr + rank) != 0;
    }
}

template <int kNumThreads>
__forceinline__ __device__ void barrier(int thread_id, int rank, int num_ranks, int* mask_buffer_ptr, int* sync_buffer_ptr) {
    EP_DEVICE_ASSERT(kNumThreads >= num_ranks);

    // Quiet all QPs
    auto qps_per_rank = ibgda_get_state()->num_rc_per_pe * ibgda_get_state()->num_devices_initialized;

    for (int i = thread_id; i < qps_per_rank * (num_ranks - 1); i += kNumThreads) {
        auto dst_rank = (rank + 1 + i / qps_per_rank) % num_ranks;
        auto qp_id = i % qps_per_rank;
        nvshmemi_ibgda_quiet(dst_rank, qp_id);
    }

    // Update local counter
    if (thread_id == 0)
        atomicAdd(sync_buffer_ptr + rank, -1);
    __syncthreads();

    int cnt = sync_buffer_ptr[rank];
    // Update remote counter and wait for local counter to be updated
    if (thread_id < num_ranks && thread_id != rank) {
        const auto dst_rank = thread_id;
        const auto dst_ptr = reinterpret_cast<uint64_t>(sync_buffer_ptr + rank);
        const auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);

        if (not is_rank_masked(mask_buffer_ptr, dst_rank)) {
            if (dst_p2p_ptr == 0) {
                nvshmemi_ibgda_rma_p(reinterpret_cast<int*>(dst_ptr), cnt, dst_rank, 0);
            } else {
                st_release_sys_global(reinterpret_cast<int*>(dst_p2p_ptr), cnt);
            }

            auto start_time = clock64();
            uint64_t wait_recv_cost = 0;
            while (ld_acquire_sys_global(sync_buffer_ptr + dst_rank) != cnt            // remote is not ready
                   && (wait_recv_cost = clock64() - start_time) <= LEGACY_NUM_TIMEOUT_CYCLES  // not timeout
            )
                ;
            // Mask rank if timeout
            if (wait_recv_cost > LEGACY_NUM_TIMEOUT_CYCLES) {
                printf("Warning: DeepEP timeout for barrier, rank %d, dst_rank %d\n", rank, dst_rank);
                if (mask_buffer_ptr == nullptr)
                    trap();
                atomicExch(mask_buffer_ptr + dst_rank, 1);
            }
        }
    }
    __syncthreads();
}

template <int kNumThreads>
__launch_bounds__(kNumThreads, 1) __global__ void clean_low_latency_buffer(int* clean_0,
                                                                           int num_clean_int_0,
                                                                           int* clean_1,
                                                                           int num_clean_int_1,
                                                                           int rank,
                                                                           int num_ranks,
                                                                           int* mask_buffer_ptr,
                                                                           int* sync_buffer_ptr) {
    auto thread_id = static_cast<int>(threadIdx.x);

    // Barrier before cleaning (in case of unfinished chunked EP)
    if (sync_buffer_ptr == nullptr)
        nvshmemx_barrier_all_block();
    else
        barrier<kNumThreads>(thread_id, rank, num_ranks, mask_buffer_ptr, sync_buffer_ptr);

    // Clean
    #pragma unroll
    for (int i = thread_id; i < num_clean_int_0; i += kNumThreads)
        clean_0[i] = 0;
    #pragma unroll
    for (int i = thread_id; i < num_clean_int_1; i += kNumThreads)
        clean_1[i] = 0;

    // Barrier after cleaning (make sure the low-latency mode works fine)
    if (sync_buffer_ptr == nullptr)
        nvshmemx_barrier_all_block();
    else
        barrier<kNumThreads>(thread_id, rank, num_ranks, mask_buffer_ptr, sync_buffer_ptr);
}

void clean_low_latency_buffer(int* clean_0,
                              int num_clean_int_0,
                              int* clean_1,
                              int num_clean_int_1,
                              int rank,
                              int num_ranks,
                              int* mask_buffer_ptr,
                              int* sync_buffer_ptr,
                              cudaStream_t stream) {
    constexpr int kNumThreads = 256;

    SETUP_LAUNCH_CONFIG(1, kNumThreads, stream);

    LAUNCH_KERNEL(&cfg,
                  clean_low_latency_buffer<kNumThreads>,
                  clean_0,
                  num_clean_int_0,
                  clean_1,
                  num_clean_int_1,
                  rank,
                  num_ranks,
                  mask_buffer_ptr,
                  sync_buffer_ptr);
}

struct PackedVec {
  __nv_bfloat162 elts[4];
};

using Type = __nv_bfloat16;

__device__ __forceinline__ float exp2f_rcp(uint8_t exp) {
  constexpr uint32_t FP32_EXPONENT_BIAS = 127;
  return (exp == 0) ? 1 : exp2f(FP32_EXPONENT_BIAS - static_cast<float>(exp));
}

// Fast reciprocal.
inline __device__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

// Convert 1 float value into one e2m1 value (represented as one uint8_t).
__device__ inline uint8_t float_to_e2m1(float f) {
    // Get sign
    uint8_t sign = (f < 0);
    float abs_f = fabsf(f);
    float abs_f_log2 = log2f(abs_f);
    // map float to 2-bit exponent
    uint8_t exp, mant;
    if (abs_f_log2 < 0) {
        exp = 0;
        if (abs_f_log2 < -1) {
            mant = 0;
        }
        else {
            mant = 1;
        }
    }
    else{
        exp = static_cast<int>(floorf(abs_f_log2 + 1));
        exp = fminf(exp, 3.0f);
        mant = (abs_f_log2 + 1 - exp > 0.5f) ? 1 : 0;
    }
    // Take one bit for mantissa
    return (sign << 3) | (exp << 1) | mant;
}

// Convert 4 float2 values into 8 e2m1 values (represented as one uint32_t).
inline __device__ uint32_t fp32_vec_to_e2m1(float2 (&array)[4]) {
    // PTX instructions used here requires sm100a.
  #if CUDA_VERSION >= 12080
  #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && __CUDA_ARCH_HAS_FEATURE__(SM100_ALL)
    uint32_t val;
    asm volatile(
        "{\n"
        ".reg .b8 byte0;\n"
        ".reg .b8 byte1;\n"
        ".reg .b8 byte2;\n"
        ".reg .b8 byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
        "}"
        : "=r"(val)
        : "f"(array[0].x),
          "f"(array[0].y),
          "f"(array[1].x),
          "f"(array[1].y),
          "f"(array[2].x),
          "f"(array[2].y),
          "f"(array[3].x),
          "f"(array[3].y));
    return val;
  #else
    uint32_t val = 0;
    float2* data = reinterpret_cast<float2*>(&array[0]);
    for (int i = 0; i < 4; ++i) {
        val |= (float_to_e2m1(data[i].x) & 0xFF) << (8 * i);
        val |= (float_to_e2m1(data[i].y) & 0xFF) << (8 * i + 4);
    }
    return val;
  #endif
  #endif
  }

constexpr int CVT_ELTS_PER_THREAD = 8;
// Quantizes the provided PackedVec into the uint32_t output
template <int SF_VEC_SIZE, bool UE8M0_SF>
__device__ uint32_t cvt_warp_fp16_to_fp4(PackedVec& vec, float SFScaleVal, uint8_t* SFout) {
  // Get absolute maximum values among the local 8 values.
  auto localMax = __habs2(vec.elts[0]);

// Local maximum value.
#pragma unroll
  for (int i = 1; i < CVT_ELTS_PER_THREAD / 2; i++) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));
  }

  constexpr int CVT_NUM_THREADS_PER_SF = SF_VEC_SIZE / CVT_ELTS_PER_THREAD;
  EP_STATIC_ASSERT(CVT_NUM_THREADS_PER_SF == 2 or CVT_NUM_THREADS_PER_SF == 4, "Invalid number of threads per SF");
  // Get the absolute maximum among all 16 values (two threads for 16, four threads for 32).
  localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  if constexpr (CVT_NUM_THREADS_PER_SF == 4) {
    localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 2), localMax);
  }
  // Get the final absolute maximum values.
  float vecMax = float(__hmax(localMax.x, localMax.y));

  // 8 bits representation of the SF.
  uint8_t fp8SFVal;
  float outputScale;
  // Write the SF to global memory (STG.8).
  if constexpr (UE8M0_SF) {
    __nv_fp8_e8m0 tmp;
    // Scale the max value to the range of E2m1.
    vecMax *= reciprocal_approximate_ftz(6.0f);
    tmp.__x = __nv_cvt_float_to_e8m0(vecMax, __NV_SATFINITE, cudaRoundPosInf);
    fp8SFVal = tmp.__x;
    outputScale = exp2f_rcp(fp8SFVal);
  } else {
    // Get the SF (max value of the vector / max value of e2m1).
    // maximum value of e2m1 = 6.0.
    // TODO: use half as compute data type.
    auto SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
    // Here SFValue is always positive, so E4M3 is the same as UE4M3.
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
    fp8SFVal = tmp.__x;
    SFValue = static_cast<float>(tmp);
    // Get the output scale.
    // Recipe: final_scale = reciprocal(fp32(fp8(SFValue * SFScaleVal)) * reciprocal(SFScaleVal))
    outputScale = SFValue != 0
                      ? reciprocal_approximate_ftz(SFValue * reciprocal_approximate_ftz(SFScaleVal))
                      : 0.0f;
  }

  if (SFout) {
    // Write the SF to global memory (STG.8).
    *SFout = fp8SFVal;
  }

  // Convert the input to float.
  float2 fp2Vals[CVT_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_ELTS_PER_THREAD / 2; i++) {
    fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  // Convert to e2m1 values.
  uint32_t e2m1Vec = fp32_vec_to_e2m1(fp2Vals);

  // Write the e2m1 values to global memory.
  return e2m1Vec;
}

template <bool kUseFP8, bool kUseUE8M0, bool kUseNVFP4, int kHidden>
__global__ __launch_bounds__(1024, 1) void dispatch(void* packed_recv_x,
                                                    void* packed_recv_x_scales,
                                                    int* packed_recv_src_info,
                                                    int64_t* packed_recv_layout_range,
                                                    int* packed_recv_count,
                                                    int* mask_buffer_ptr,
                                                    int* cumulative_local_expert_recv_stats,
                                                    int64_t* dispatch_wait_recv_cost_stats,
                                                    const float* x_global_scale,
                                                    void* rdma_recv_x,
                                                    int* rdma_recv_count,
                                                    void* rdma_x,
                                                    const void* x,
                                                    const topk_idx_t* topk_idx,
                                                    int* atomic_counter_per_expert,
                                                    int* atomic_finish_counter_per_expert,
                                                    int* next_clean,
                                                    int num_next_clean_int,
                                                    int num_tokens,
                                                    int num_max_dispatch_tokens_per_rank,
                                                    int num_topk,
                                                    int num_experts,
                                                    int rank,
                                                    int num_ranks,
                                                    int num_warp_groups,
                                                    int num_warps_per_group,
                                                    bool round_scale,
                                                    int phases) {
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto warp_id = thread_id / 32, lane_id = get_lane_id();
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto num_warps = num_warp_groups * num_warps_per_group;
    const auto num_local_experts = num_experts / num_ranks;
    const auto warp_group_id = warp_id / num_warps_per_group;
    const auto sub_warp_id = warp_id % num_warps_per_group;
    const auto responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;

    // May extract UE8M0 from the scales
    using scale_t = std::conditional_t<kUseUE8M0 || kUseNVFP4, uint8_t, float>;
    using packed_t = std::conditional_t<kUseUE8M0 || kUseNVFP4, uint32_t, float>;
    EP_STATIC_ASSERT(sizeof(packed_t) % sizeof(scale_t) == 0, "Invalid vector length");
    EP_STATIC_ASSERT(!(kUseFP8 && kUseNVFP4), "FP8 and NVFP4 cannot be used together");

    // FP8 staffs
    constexpr int kNumPerChannels = kUseNVFP4 ? 16 : 128;
    const int num_scales = kHidden / kNumPerChannels;
    constexpr size_t hidden_bytes =
    kUseNVFP4
        ? kHidden * sizeof(__nv_fp8_storage_t) / 2
        : kHidden * (kUseFP8 ? sizeof(__nv_fp8_storage_t) : sizeof(nv_bfloat16));
    const size_t hidden_int4 = hidden_bytes / sizeof(int4);

    // Message package: index at source (int), 3 reserved int fields, hidden data, FP8 scales
    // NOTES: currently we have 3 reserved int fields for future use
    using vec_t = std::conditional_t<
        kUseNVFP4,
        int32_t,
        std::conditional_t<kUseFP8, int2, int4>>;
    using rdma_x_scale_t = std::conditional_t<kUseNVFP4, uint8_t, float>;
    const size_t num_bytes_per_msg = sizeof(int4) + ((kUseFP8 || kUseNVFP4) ? (hidden_bytes + num_scales * sizeof(rdma_x_scale_t)) : hidden_bytes);
    const size_t num_int4_per_msg = num_bytes_per_msg / sizeof(int4);
    EP_DEVICE_ASSERT(num_bytes_per_msg % sizeof(int4) == 0);

    // Expert counts
    constexpr int kNumMaxWarpGroups = 32;
    __shared__ int shared_num_tokens_sent_per_expert[kNumMaxWarpGroups];

    // Sending phase
    if ((phases & LEGACY_LOW_LATENCY_SEND_PHASE) == 0)
        goto LOW_LATENCY_DISPATCH_RECV;

    // There are 2 kinds of warps in this part:
    // 1. The first-kind warps for FP8 cast and sending top-k tokens
    // 2. The last warp for reading `topk_idx` and count for per-expert information
    if (warp_id < num_warps - 1) {
        constexpr int kNumElemsPerRead = sizeof(int4) / sizeof(nv_bfloat16);
        EP_STATIC_ASSERT(kHidden % (32 * kNumElemsPerRead) == 0, "Invalid hidden");
        EP_STATIC_ASSERT(kNumElemsPerRead * 32 % kNumPerChannels == 0, "Invalid vectorization");
        const auto num_threads = (num_warps - 1) * 32;
        const size_t hidden_bf16_int4 = kHidden / kNumElemsPerRead;

        for (int token_idx = sm_id; token_idx < num_tokens; token_idx += num_sms) {
            const auto x_int4 = static_cast<const int4*>(x) + token_idx * hidden_bf16_int4;
            const auto rdma_x_src_idx = reinterpret_cast<int*>(static_cast<uint8_t*>(rdma_x) + token_idx * num_bytes_per_msg);
            const auto rdma_x_vec = reinterpret_cast<vec_t*>(reinterpret_cast<uint8_t*>(rdma_x_src_idx) + sizeof(int4));
            const auto rdma_x_scales = reinterpret_cast<rdma_x_scale_t*>(reinterpret_cast<uint8_t*>(rdma_x_vec) + hidden_bytes);

            // Overlap top-k index read and source token index writes
            auto dst_expert_idx = warp_id < num_topk ? static_cast<int>(__ldg(topk_idx + token_idx * num_topk + warp_id)) : -1;
            thread_id == 0 ? (*rdma_x_src_idx = token_idx) : 0;
            float SFScaleVal = 1.0f;
            if constexpr (kUseNVFP4) {
                // Get scaling value;
                EP_DEVICE_ASSERT(x_global_scale != nullptr);
                SFScaleVal = *(static_cast<const float*>(x_global_scale));
            }

            // FP8 or NVFP4 cast
            EP_STATIC_ASSERT(hidden_bf16_int4 % 32 == 0, "Must use the full warp to reduce");
            #pragma unroll
            for (int i = thread_id; i < hidden_bf16_int4; i += num_threads) {
                // Read
                auto int4_value = __ldg(x_int4 + i);

                if constexpr (kUseFP8) {
                    // Calculate local amax
                    auto bf16_values = reinterpret_cast<nv_bfloat16*>(&int4_value);
                    float fp32_values[kNumElemsPerRead];
                    float amax = kFP8Margin, scale, scale_inv;
                    #pragma unroll
                    for (int j = 0; j < kNumElemsPerRead; ++j) {
                        fp32_values[j] = static_cast<float>(bf16_values[j]);
                        amax = fmaxf(amax, fabsf(fp32_values[j]));
                    }

                    // Reduce amax and scale
                    EP_STATIC_ASSERT(kNumElemsPerRead * 32 / kNumPerChannels == 2, "Invalid vectorization");
                    amax = warp_reduce_max<16>(amax);
                    calculate_fp8_scales(amax, scale, scale_inv, round_scale);
                    if (lane_id == 0 or lane_id == 16)
                        rdma_x_scales[i * kNumElemsPerRead / 128] = scale_inv;

                    // Cast into send buffer
                    vec_t int2_value;
                    auto fp8x2_values = reinterpret_cast<__nv_fp8x2_storage_t*>(&int2_value);
                    #pragma unroll
                    for (int j = 0; j < kNumElemsPerRead; j += 2) {
                        float2 fp32x2 = {fp32_values[j] * scale, fp32_values[j + 1] * scale};
                        fp8x2_values[j / 2] = __nv_cvt_float2_to_fp8x2(fp32x2, __NV_SATFINITE, __NV_E4M3);
                    }
                    rdma_x_vec[i] = int2_value;
                } else if constexpr (kUseNVFP4) {
                    // Convert to NVFP4
                    uint8_t sf_val;
                    PackedVec vec = *reinterpret_cast<PackedVec*>(&int4_value);
                    uint32_t result = cvt_warp_fp16_to_fp4<kNumPerChannels, kUseUE8M0>(vec, SFScaleVal, &sf_val);

                    // Write scale to send buffer
                    if (lane_id % 2 == 0){
                        EP_DEVICE_ASSERT((i * kNumElemsPerRead) % kNumPerChannels == 0);
                        int rdma_x_scale_idx = i * kNumElemsPerRead / kNumPerChannels;
                        rdma_x_scales[rdma_x_scale_idx] = sf_val;
                        }
                    // Cast into send buffer
                    rdma_x_vec[i] = *reinterpret_cast<vec_t*>(&result);
                } else {
                    // Reinterpret-cast is for C++14 compatibility
                    rdma_x_vec[i] = *reinterpret_cast<vec_t*>(&int4_value);
                }
            }
            asm volatile("bar.sync 1, %0;" ::"r"(num_threads));

            // Issue IBGDA sends
            if (dst_expert_idx >= 0) {
                int slot_idx = lane_id == 0 ? atomicAdd(atomic_counter_per_expert + dst_expert_idx, 1) : 0;
                slot_idx = __shfl_sync(0xffffffff, slot_idx, 0);
                const auto dst_rank = dst_expert_idx / num_local_experts;
                const auto dst_expert_local_idx = dst_expert_idx % num_local_experts;
                const auto src_ptr = reinterpret_cast<uint64_t>(rdma_x_src_idx);
                const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x) +
                    dst_expert_local_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
                    rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg + slot_idx * num_bytes_per_msg;
                const auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
                if (not is_rank_masked<true>(mask_buffer_ptr, dst_rank)) {
                    if (dst_p2p_ptr == 0) {
                        nvshmemi_ibgda_put_nbi_warp(dst_ptr, src_ptr, num_bytes_per_msg, dst_rank, dst_expert_local_idx, lane_id, slot_idx);
                    } else {
                        // NOTES: only 2 load iterations for 7K hidden with 8 unrolls
                        const auto* src_int4_ptr = reinterpret_cast<const int4*>(src_ptr);
                        const auto* dst_int4_ptr = reinterpret_cast<int4*>(dst_p2p_ptr);
                        UNROLLED_WARP_COPY(8, lane_id, num_int4_per_msg, dst_int4_ptr, src_int4_ptr, ld_nc_global, st_na_global);
                    }
                }

                // Increase counter after finishing
                __syncwarp();
                lane_id == 0 ? atomic_add_release_global(atomic_finish_counter_per_expert + dst_expert_idx, 1) : 0;
            }
        }
    } else if (warp_id == num_warps - 1) {
        EP_DEVICE_ASSERT(num_sms > 1);
        if (sm_id == 0) {
            // The first SM is also responsible for checking QPs
            EP_DEVICE_ASSERT(ibgda_get_state()->num_rc_per_pe >= num_local_experts);

            // The first SM is also responsible for cleaning the next buffer
            #pragma unroll
            for (int i = lane_id; i < num_next_clean_int; i += 32)
                next_clean[i] = 0;

            // Notify before executing `int_p`
            __syncwarp();
            #pragma unroll
            for (int i = lane_id; i < num_experts; i += 32)
                atomic_add_release_global(atomic_finish_counter_per_expert + i, LEGACY_FINISHED_SUM_TAG);
        }

        // This SM should be responsible for some destination experts, read `topk_idx` for them
        int expert_count[kNumMaxWarpGroups] = {0};
        const auto expert_begin_idx = sm_id * num_warp_groups;
        const auto expert_end_idx = min(expert_begin_idx + num_warp_groups, num_experts);

        // Per lane count
        #pragma unroll 8
        for (int i = lane_id; i < num_tokens * num_topk; i += 32) {
            auto idx = static_cast<int>(__ldg(topk_idx + i));
            if (idx >= expert_begin_idx and idx < expert_end_idx)
                expert_count[idx - expert_begin_idx]++;
        }

        // Warp reduce
        #pragma unroll
        for (int i = expert_begin_idx; i < expert_end_idx; ++i) {
            auto sum = warp_reduce_sum(expert_count[i - expert_begin_idx]);
            if (lane_id == 0) {
                shared_num_tokens_sent_per_expert[i - expert_begin_idx] = sum;
                atomic_add_release_global(atomic_finish_counter_per_expert + i, LEGACY_FINISHED_SUM_TAG - sum);
            }
        }
    }
    __syncthreads();

    // Issue count sends
    if (responsible_expert_idx < num_experts and sub_warp_id == 0 and lane_id == 0) {
        const auto dst_rank = responsible_expert_idx / num_local_experts;
        const auto dst_expert_local_idx = responsible_expert_idx % num_local_experts;
        const auto num_tokens_sent = shared_num_tokens_sent_per_expert[responsible_expert_idx - sm_id * num_warp_groups];

        // Wait local sends issued and send expert counts
        while (ld_acquire_global(atomic_finish_counter_per_expert + responsible_expert_idx) != LEGACY_FINISHED_SUM_TAG * 2)
            ;
        auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_count + dst_expert_local_idx * num_ranks + rank);
        auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
        if (not is_rank_masked(mask_buffer_ptr, dst_rank)) {
            if (dst_p2p_ptr == 0) {
                nvshmemi_ibgda_amo_nonfetch_add(reinterpret_cast<int*>(dst_ptr), -num_tokens_sent - 1, dst_rank, dst_expert_local_idx);
            } else {
                st_release_sys_global(reinterpret_cast<int*>(dst_p2p_ptr), -num_tokens_sent - 1);
            }
        }

        // Clean workspace for next use
        atomic_counter_per_expert[responsible_expert_idx] = 0;
        atomic_finish_counter_per_expert[responsible_expert_idx] = 0;

        // Clean `packed_recv_count`
        if (dst_rank == 0)
            packed_recv_count[dst_expert_local_idx] = 0;
    }
    __syncwarp();

// Receiving phase
LOW_LATENCY_DISPATCH_RECV:
    if ((phases & LEGACY_LOW_LATENCY_RECV_PHASE) == 0)
        return;

    // For send-and-recv kernels, we need a grid sync for making `packed_recv_count` visible
    if (phases & LEGACY_LOW_LATENCY_SEND_PHASE)
        cg::this_grid().sync();

    // Receiving and packing
    if (responsible_expert_idx < num_experts) {
        const auto src_rank = responsible_expert_idx / num_local_experts;
        const auto local_expert_idx = responsible_expert_idx % num_local_experts;
        const auto rdma_recv_x_uint8 = static_cast<uint8_t*>(rdma_recv_x) +
            local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
            src_rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg;
        const auto recv_x_int4 =
            static_cast<int4*>(packed_recv_x) + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * hidden_int4;
        const auto recv_src_info = packed_recv_src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
        const auto recv_range = packed_recv_layout_range + local_expert_idx * num_ranks;
        const auto num_aligned_tokens = align_up<int>(num_ranks * num_max_dispatch_tokens_per_rank, 128);
        const auto num_aligned_scales = align_up<int>(num_scales, sizeof(float) / sizeof(scale_t));
        const auto recv_x_scales = static_cast<scale_t*>(packed_recv_x_scales) + local_expert_idx * num_aligned_tokens * num_aligned_scales;

        // Shared between sub-warps in warp groups
        __shared__ int shared_num_recv_tokens[kNumMaxWarpGroups], shared_recv_token_begin_idx[kNumMaxWarpGroups];

        // Wait tokens to arrive
        // NOTES: using sub-warp 1 to overlap with sub-warp 0
        int num_recv_tokens = 0, recv_token_begin_idx;
        EP_DEVICE_ASSERT(num_warps_per_group > 1 and num_warp_groups < 15);
        if (sub_warp_id == 1 and lane_id == 0) {
            auto start_time = clock64();
            uint64_t wait_recv_cost = 0;
            if (not is_rank_masked(mask_buffer_ptr, src_rank)) {
                while ((num_recv_tokens = ld_acquire_sys_global(rdma_recv_count + local_expert_idx * num_ranks + src_rank)) ==
                           0                                                               // data not arrived
                       && (wait_recv_cost = clock64() - start_time) <= LEGACY_NUM_TIMEOUT_CYCLES  // not timeout
                )
                    ;
            }
            // Do not receive tokens if rank timeout or masked
            if (num_recv_tokens == 0)
                num_recv_tokens = -1;
            // Mask rank if timeout
            if (wait_recv_cost > LEGACY_NUM_TIMEOUT_CYCLES) {
                printf("Warning: DeepEP timeout for dispatch receive, rank %d, local_expert_idx %d, src_rank %d\n",
                       rank,
                       local_expert_idx,
                       src_rank);
                if (mask_buffer_ptr == nullptr)
                    trap();
                atomicExch(mask_buffer_ptr + src_rank, 1);
            }

            num_recv_tokens = -num_recv_tokens - 1;
            recv_token_begin_idx = atomicAdd(packed_recv_count + local_expert_idx, num_recv_tokens);
            shared_num_recv_tokens[warp_group_id] = num_recv_tokens;
            shared_recv_token_begin_idx[warp_group_id] = recv_token_begin_idx;
            recv_range[src_rank] = pack2<int, int64_t>(num_recv_tokens, recv_token_begin_idx);

            // Add stats for diagnosis
            if (cumulative_local_expert_recv_stats != nullptr)
                atomicAdd(cumulative_local_expert_recv_stats + local_expert_idx, num_recv_tokens);
            if (dispatch_wait_recv_cost_stats != nullptr)
                atomicAdd(reinterpret_cast<unsigned long long*>(dispatch_wait_recv_cost_stats + src_rank), wait_recv_cost);
        }
        asm volatile("bar.sync %0, %1;" ::"r"(warp_group_id + 2), "r"(num_warps_per_group * 32));
        num_recv_tokens = shared_num_recv_tokens[warp_group_id];
        recv_token_begin_idx = shared_recv_token_begin_idx[warp_group_id];

        // Copy tokens
        if constexpr (kUseFP8)
            EP_DEVICE_ASSERT(num_scales <= 64);
        for (int i = sub_warp_id; i < num_recv_tokens; i += num_warps_per_group) {
            // Copy source info
            const auto src_src_idx = reinterpret_cast<int*>(rdma_recv_x_uint8 + i * num_bytes_per_msg);
            if (lane_id == 0)
                recv_src_info[recv_token_begin_idx + i] = ld_nc_global(src_src_idx);
            __syncwarp();

            // Copy data
            // NOTES: only 2 load iterations for 7K hidden with 7 unrolls
            const auto src_data = reinterpret_cast<int4*>(reinterpret_cast<uint8_t*>(src_src_idx) + sizeof(int4));
            const auto dst_data = recv_x_int4 + (recv_token_begin_idx + i) * hidden_int4;
            UNROLLED_WARP_COPY(7, lane_id, hidden_int4, dst_data, src_data, ld_nc_global, st_na_global);

            // Copy scales
            if constexpr (kUseFP8) {
                // Equivalent CuTe layout:
                //   (num_tokens, (num_packed, num_elems_per_pack)):(num_elems_per_pack, (num_tokens * num_elems_per_pack, 1))
                const auto src_scales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(src_data) + hidden_bytes);
                const auto num_elems_per_pack = static_cast<int>(sizeof(packed_t) / sizeof(scale_t));
                const auto token_idx = recv_token_begin_idx + i;
                const auto token_stride = num_elems_per_pack;
                const auto pack_stride = num_ranks * num_max_dispatch_tokens_per_rank * num_elems_per_pack;
                if (lane_id < num_scales) {
                    const auto pack_idx = lane_id / num_elems_per_pack;
                    const auto elem_idx = lane_id % num_elems_per_pack;
                    auto scale = extract_required_scale_format<kUseUE8M0>(ld_nc_global(src_scales + lane_id));
                    recv_x_scales[token_idx * token_stride + pack_idx * pack_stride + elem_idx] = scale;
                }
                if (lane_id + 32 < num_scales) {
                    const auto pack_idx = (lane_id + 32) / num_elems_per_pack;
                    const auto elem_idx = (lane_id + 32) % num_elems_per_pack;
                    auto scale = extract_required_scale_format<kUseUE8M0>(ld_nc_global(src_scales + lane_id + 32));
                    recv_x_scales[token_idx * token_stride + pack_idx * pack_stride + elem_idx] = scale;
                }
            } else if constexpr (kUseNVFP4) {
                 // The physical layout is (l, rm, rk, 32, 4, 4)
                const auto src_scales = reinterpret_cast<uint8_t*>(reinterpret_cast<uint8_t*>(src_data) + hidden_bytes);
                const auto num_elems_per_pack = static_cast<int>(sizeof(packed_t) / sizeof(scale_t));
                const auto token_idx = recv_token_begin_idx + i;

                const auto rk = align_up<int>(kHidden / kNumPerChannels, 4) / 4;
                const auto dim0_stride = rk * 128 * num_elems_per_pack;
                const auto dim1_stride = 128 * num_elems_per_pack;
                const auto dim2_stride = 4 * num_elems_per_pack;
                const auto dim3_stride = num_elems_per_pack;

                const auto dim0_offset = token_idx / 128;
                const auto dim2_offset = (token_idx % 128) % 32;
                const auto dim3_offset = (token_idx % 128) / 32;

                #pragma unroll
                for (int j = lane_id; j < num_scales; j += 32) {
                    const auto dim1_offset = j / num_elems_per_pack;
                    const auto dim4_offset = j % num_elems_per_pack;
                    auto scale = ld_nc_global(src_scales + j);
                    const auto offset = dim0_offset * dim0_stride + dim1_offset * dim1_stride + dim2_offset * dim2_stride + dim3_offset * dim3_stride + dim4_offset;
                    recv_x_scales[offset] = scale;
                }
            }
        }
    }
}

void dispatch(void* packed_recv_x,
              void* packed_recv_x_scales,
              int* packed_recv_src_info,
              int64_t* packed_recv_layout_range,
              int* packed_recv_count,
              int* mask_buffer_ptr,
              int* cumulative_local_expert_recv_stats,
              int64_t* dispatch_wait_recv_cost_stats,
              const float* x_global_scale,
              void* rdma_recv_x,
              int* rdma_recv_count,
              void* rdma_x,
              const void* x,
              const topk_idx_t* topk_idx,
              int* next_clean,
              int num_next_clean_int,
              int num_tokens,
              int hidden,
              int num_max_dispatch_tokens_per_rank,
              int num_topk,
              int num_experts,
              int rank,
              int num_ranks,
              bool use_fp8,
              bool round_scale,
              bool use_ue8m0,
              bool use_nvfp4,
              void* workspace,
              int num_device_sms,
              cudaStream_t stream,
              int phases) {
    constexpr int kNumMaxTopK = 16;  // Kimi-K3 routes top-16
    const int num_warp_groups = ceil_div(num_experts, num_device_sms);
    const int num_warps_per_group = 32 / num_warp_groups;
    EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 0);
    EP_HOST_ASSERT(kNumMaxTopK + 1 <= num_warp_groups * num_warps_per_group);

    const auto num_warps = num_warp_groups * num_warps_per_group;
    const auto num_sms = ceil_div(num_experts, num_warp_groups);
    EP_HOST_ASSERT(num_topk <= kNumMaxTopK);

    // Workspace checks
    auto atomic_counter_per_expert = static_cast<int*>(workspace);
    auto atomic_finish_counter_per_expert = atomic_counter_per_expert + num_experts;
    EP_HOST_ASSERT(num_experts * sizeof(int) * 2 <= LEGACY_NUM_WORKSPACE_BYTES);

    // FP8 checks
    if (use_fp8 and use_ue8m0)
        EP_HOST_ASSERT(round_scale and "FP8 UE8M0 SF requires `round_scale=True`");

#define DISPATCH_LAUNCH_CASE(hidden)                         \
    {                                                        \
        auto dispatch_func = dispatch<false, false, false, hidden>; \
        if (use_fp8 and not use_ue8m0)                       \
            dispatch_func = dispatch<true, false, false, hidden>; \
        if (use_fp8 and use_ue8m0)                           \
            dispatch_func = dispatch<true, true, false, hidden>; \
        if (use_nvfp4 and not use_ue8m0)                     \
            dispatch_func = dispatch<false, false, true, hidden>; \
        if (use_nvfp4 and use_ue8m0)                         \
            dispatch_func = dispatch<false, true, true, hidden>; \
        LAUNCH_KERNEL(&cfg,                                  \
                      dispatch_func,                         \
                      packed_recv_x,                         \
                      packed_recv_x_scales,                  \
                      packed_recv_src_info,                  \
                      packed_recv_layout_range,              \
                      packed_recv_count,                     \
                      mask_buffer_ptr,                       \
                      cumulative_local_expert_recv_stats,    \
                      dispatch_wait_recv_cost_stats,         \
                      x_global_scale,                        \
                      rdma_recv_x,                           \
                      rdma_recv_count,                       \
                      rdma_x,                                \
                      x,                                     \
                      topk_idx,                              \
                      atomic_counter_per_expert,             \
                      atomic_finish_counter_per_expert,      \
                      next_clean,                            \
                      num_next_clean_int,                    \
                      num_tokens,                            \
                      num_max_dispatch_tokens_per_rank,      \
                      num_topk,                              \
                      num_experts,                           \
                      rank,                                  \
                      num_ranks,                             \
                      num_warp_groups,                       \
                      num_warps_per_group,                   \
                      round_scale,                           \
                      phases);                               \
    }                                                        \
    break

    SETUP_LAUNCH_CONFIG(num_sms, num_warps * 32, stream);
    SWITCH_HIDDEN(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

template <int kNumSendUnrolls>
__forceinline__ __device__ int logfmt_encode(void* buffer, nv_bfloat162* shared_amaxmin, const int& lane_id) {
    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
    constexpr float kLogThreshold = 0;
    constexpr float kMinClip = 32;  // `== log_2(2 ^ (2 ^ 5))`
    constexpr int kNumBits = 10;
    constexpr int kNumValues = 1 << (kNumBits - 1);

    int4 int4_values[kNumSendUnrolls];
    const auto uint32_values = reinterpret_cast<uint32_t*>(int4_values);
    const auto bf162_values = reinterpret_cast<nv_bfloat162*>(int4_values);

    // Calculate lane offset
    const auto ld_buffer = reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(buffer) + lane_id * (kNumSendUnrolls * sizeof(int4)));
    const auto st_buffer =
        reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(buffer) + lane_id * (kNumSendUnrolls * sizeof(int4) * 10 / 16));

    // Local log amax
    auto bf162_amax = __nv_bfloat162(CUDART_ZERO_BF16, CUDART_ZERO_BF16);
    auto bf162_amin = __nv_bfloat162(CUDART_INF_BF16, CUDART_INF_BF16);
    uint32_t local_signs = 0;
    #pragma unroll
    for (int k = 0; k < kNumSendUnrolls * kNumElemsPerInt4 / 2; ++k) {
        // TODO: eliminate bank conflicts
        uint32_values[k] = ld_buffer[k];
        local_signs |= ((uint32_values[k] >> 15) & 1) << (k * 2);
        local_signs |= ((uint32_values[k] >> 31) & 1) << (k * 2 + 1);
        uint32_values[k] &= 0x7fff7fff;

        bf162_amax = __hmax2(bf162_amax, bf162_values[k]);
        bf162_amin = __hmin2(bf162_amin, bf162_values[k]);
    }

    // Reduce per 128 channels
    // TODO: figure out how hardware do 2-byte min/max
    auto amax = std::max(static_cast<float>(bf162_amax.x), static_cast<float>(bf162_amax.y));
    auto amin = std::min(static_cast<float>(bf162_amin.x), static_cast<float>(bf162_amin.y));
    constexpr static int kNumLanesToReduce = 128 * sizeof(nv_bfloat16) / (kNumSendUnrolls * sizeof(int4));
    amax = warp_reduce_max<kNumLanesToReduce>(amax);
    amin = warp_reduce_min<kNumLanesToReduce>(amin);

    // Write min/max into the shared memory
    if (shared_amaxmin != nullptr)
        *shared_amaxmin = __nv_bfloat162(amax, amin);
    __syncwarp();

    // Calculate log amin/amax float
    const auto log_amax = log2f_approx(amax);
    const auto log_amin = fmaxf(log2f_approx(amin), log_amax - kMinClip);
    const bool enable_cast = warp_reduce_and<kNumLanesToReduce, true>(log_amax < kLogThreshold and log_amin < log_amax);

    // Case into LogFMT-10 if satisfied
    if (enable_cast) {
        const auto step = (log_amax - log_amin) / static_cast<float>(kNumValues - 2);
        const auto step_inv = 1.0f / step;
        const auto rounding = 2.0f - log2f_approx((1.0f + exp2f_approx(step)) * 0.5f) * step_inv;
        const auto fused_rounding = rounding - log_amin * step_inv;

        // Pack every 256 bits into 160 bits
        EP_STATIC_ASSERT(kNumSendUnrolls == 2 or kNumSendUnrolls == 4, "kNumSendUnrolls == 2 or 4 only");
        uint32_t encoded[kNumElemsPerInt4 * 2];
        #pragma unroll 1
        for (int i = 0; i < kNumSendUnrolls / 2; ++i) {
            #pragma unroll
            for (int k = 0; k < kNumElemsPerInt4; ++k) {
                const auto [x, y] = __bfloat1622float2(bf162_values[i * kNumElemsPerInt4 + k]);
                encoded[k * 2 + 0] = __float2uint_rd(fmaxf(log2f_approx(x) * step_inv + fused_rounding, 0));
                encoded[k * 2 + 1] = __float2uint_rd(fmaxf(log2f_approx(y) * step_inv + fused_rounding, 0));
            }
            st_buffer[i * 5 + 0] = (encoded[0] >> 0) | (encoded[1] << 9) | (encoded[2] << 18) | (encoded[3] << 27);
            st_buffer[i * 5 + 1] = (encoded[3] >> 5) | (encoded[4] << 4) | (encoded[5] << 13) | (encoded[6] << 22) | (encoded[7] << 31);
            st_buffer[i * 5 + 2] = (encoded[7] >> 1) | (encoded[8] << 8) | (encoded[9] << 17) | (encoded[10] << 26);
            st_buffer[i * 5 + 3] =
                (encoded[10] >> 6) | (encoded[11] << 3) | (encoded[12] << 12) | (encoded[13] << 21) | (encoded[14] << 30);
            st_buffer[i * 5 + 4] = (encoded[14] >> 2) | (encoded[15] << 7) | ((i == 0) ? (local_signs << 16) : (local_signs & 0xffff0000u));
        }
        tma_store_fence();
        __syncwarp();
    }

    // Return TMA copy bytes
    return enable_cast ? (32 * (kNumSendUnrolls * sizeof(int4) * 8 * 10 / 16 / 8)) : (32 * (kNumSendUnrolls * sizeof(int4)));
}

template <int kNumLanes, int kNumSendUnrolls, int kNumRecvUnrolls>
__forceinline__ __device__ void logfmt_check_amaxmin(
    uint8_t* meta_buffer, float2* shared_log_amax, float2* shared_log_amin, int* shared_cast_info, const int lane_id) {
    constexpr float kLogThreshold = 0;
    constexpr float kMinClip = 32;  // `== log_2(2 ^ (2 ^ 5))`

    bool enable_cast = true;
    if (lane_id < kNumLanes) {
        // Calculate log amin/amax float
        auto amaxmin2 = reinterpret_cast<uint64_t*>(meta_buffer)[lane_id];
        const auto bf162_amaxmin = reinterpret_cast<__nv_bfloat162*>(&amaxmin2);
        float log_amax[2], log_amin[2];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            auto amax = static_cast<float>(bf162_amaxmin[i].x);
            auto amin = static_cast<float>(bf162_amaxmin[i].y);
            log_amax[i] = log2f_approx(amax);
            log_amin[i] = amin == 0 ? log_amax[i] - kMinClip : fmaxf(log2f_approx(amin), log_amax[i] - kMinClip);
            enable_cast = enable_cast and log_amax[i] < kLogThreshold and log_amin[i] < log_amax[i];
        }
        shared_log_amax[lane_id] = make_float2(log_amax[0], log_amax[1]);
        shared_log_amin[lane_id] = make_float2(log_amin[0], log_amin[1]);
    }

    const auto casted = warp_reduce_and<kNumSendUnrolls>(enable_cast) ? 1u << (lane_id / kNumRecvUnrolls) : 0u;
    const auto num_casted_prefix = __popc(warp_reduce_or<kNumRecvUnrolls, true>(casted) & ((1u << (lane_id / kNumRecvUnrolls)) - 1));

    if (lane_id < kNumLanes and lane_id % kNumRecvUnrolls == 0)
        shared_cast_info[lane_id / kNumRecvUnrolls] = (num_casted_prefix << 1) | (casted ? 1u : 0u);
    __syncwarp();
}

template <int kNumRecvUnrolls>
__forceinline__ __device__ void decode_and_accumulate(
    uint32_t* ld_buffer, float* accum, const float& log_amax, const float& log_amin, const bool& enable_cast, const float& weight) {
    if (enable_cast) {
        constexpr int kNumBits = 10;
        constexpr int kNumValues = 1 << (kNumBits - 1);

        const auto step = (log_amax - log_amin) / static_cast<float>(kNumValues - 2);
        auto decode = [=](const uint32_t& encoded, const uint32_t& sign) {
            const auto decoded = encoded == 0 ? .0f : exp2f_approx((encoded - 1) * step + log_amin);
            return sign ? -decoded : decoded;
        };

        EP_STATIC_ASSERT(kNumRecvUnrolls == 2 or kNumRecvUnrolls == 4, "kNumRecvUnrolls == 2 or 4 only");
        #pragma unroll
        for (int i = 0; i < kNumRecvUnrolls / 2; ++i) {
            uint32_t concat[6];
            concat[0] = ld_buffer[i * 5];
            #pragma unroll
            for (int k = 1; k < 5; ++k)
                concat[k] = (ld_buffer[i * 5 + k - 1] >> (32 - k * 5)) | (ld_buffer[i * 5 + k] << (k * 5));
            concat[5] = ld_buffer[i * 5 + 4] >> 7;

            const uint32_t local_signs = ld_buffer[i * 5 + 4] >> 16;
            #pragma unroll
            for (int k = 0; k < 5; ++k) {
                accum[i * 16 + k * 3 + 0] += decode((concat[k] >> 0) & 0x1ff, (local_signs >> (k * 3 + 0)) & 1) * weight;
                accum[i * 16 + k * 3 + 1] += decode((concat[k] >> 9) & 0x1ff, (local_signs >> (k * 3 + 1)) & 1) * weight;
                accum[i * 16 + k * 3 + 2] += decode((concat[k] >> 18) & 0x1ff, (local_signs >> (k * 3 + 2)) & 1) * weight;
            }
            accum[i * 16 + 15] += decode(concat[5] & 0x1ff, (local_signs >> 15) & 1) * weight;
        }
    } else {
        #pragma unroll
        for (int k = 0; k < kNumRecvUnrolls * 4; ++k) {
            auto bf16_pack = *reinterpret_cast<__nv_bfloat162*>(ld_buffer + k);
            accum[k * 2 + 0] += static_cast<float>(bf16_pack.x) * weight;
            accum[k * 2 + 1] += static_cast<float>(bf16_pack.y) * weight;
        }
    }
}

template <bool kUseLogFMT, int kHidden, int kNumMaxTopk, int kNumMaxUnrolls>
__global__
__launch_bounds__(1024, 1)
void
combine_overlappable(void* combined_x,
        void* rdma_recv_x, int* rdma_recv_flag, void* rdma_send_x,
        const void* x, const topk_idx_t* topk_idx, const float* topk_weights,
        const int* src_info, const int64_t* layout_range,
        int64_t* combine_wait_recv_cost_stats,
        int* next_clean, int num_next_clean_int,
        int* atomic_clean_flag,
        int num_combined_tokens, int hidden, int num_topk,
        int num_max_dispatch_tokens_per_rank,
        int num_experts, int rank, int num_ranks,
        int num_warp_groups, int num_warps_per_group,
        int phases, bool zero_copy,
        uint32_t* src_signals, uint32_t src_signal_expect_value) {
    const auto sm_id = __shfl_sync(0xffffffff, static_cast<int>(blockIdx.x), 0);
    const auto num_sms = __shfl_sync(0xffffffff, static_cast<int>(gridDim.x), 0);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto num_threads = __shfl_sync(0xffffffff, static_cast<int>(blockDim.x), 0);
    const auto warp_id = __shfl_sync(0xffffffff, thread_id / 32, 0), lane_id = get_lane_id();
    const auto num_local_experts = num_experts / num_ranks;
    const auto warp_group_id = warp_id / num_warps_per_group;
    const auto sub_warp_id = warp_id % num_warps_per_group;
    const auto responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;

    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // Data type staffs
    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
    constexpr int64_t hidden_bf16_int4 = kHidden / kNumElemsPerInt4;

    // Use different unroll factors for send and recv phases
    constexpr int kNumSendUnrolls = kHidden % (32 * 4 * sizeof(int4) / sizeof(nv_bfloat16)) == 0 ? 4 : 2;
    constexpr int kNumRecvUnrolls = 2;
    constexpr int hidden_bf16_int4_pad = align_up(static_cast<int>(hidden_bf16_int4), 32 * kNumSendUnrolls);
    EP_STATIC_ASSERT(kHidden % (32 * 2 * sizeof(int4) / sizeof(nv_bfloat16)) == 0, "Invalid hidden");
    EP_STATIC_ASSERT(kNumSendUnrolls <= kNumMaxUnrolls and kNumRecvUnrolls <= kNumMaxUnrolls, "Invalid unrolls");
    EP_STATIC_ASSERT(hidden_bf16_int4 % kNumSendUnrolls == 0, "Invalid hidden");
    EP_STATIC_ASSERT(kNumSendUnrolls >= kNumRecvUnrolls, "Invalid unroll factors");

    // Message package
    EP_STATIC_ASSERT(kHidden % 128 == 0, "Invalid hidden");
    constexpr int kNumDivisions = kHidden / 128;
    constexpr int kNumMetaBytes = kNumDivisions * sizeof(nv_bfloat162);
    constexpr size_t num_bytes_per_slot = kHidden * sizeof(nv_bfloat16) + kNumMetaBytes;
    EP_STATIC_ASSERT(num_bytes_per_slot % sizeof(int4) == 0, "Invalid vectorization");

    // Sending phase
    if ((phases & LEGACY_LOW_LATENCY_SEND_PHASE) == 0)
        goto LOW_LATENCY_COMBINE_RECV;

    // Clean up next buffer
    if (sm_id == 0 and warp_group_id == 0 and sub_warp_id == 0) {
        #pragma unroll
        for (int i = lane_id; i < num_next_clean_int; i += 32)
            next_clean[i] = 0;

        // Notify before executing `int_p`
        __syncwarp();
        if (lane_id == 0)
            atomic_add_release_global(atomic_clean_flag, num_experts);
    }

    // Issue IBGDA sends
    if (responsible_expert_idx < num_experts) {
        // ------------------------------------------ START tma-related -------------------------------------------------
        // TMA stuffs
        constexpr int kNumTMABufferBytes = sizeof(int4) * 32 * kNumSendUnrolls;
        constexpr int kNumStages = 3;
        constexpr int kNumPrefetch = 1;
        EP_STATIC_ASSERT(kNumStages == 3 and kNumPrefetch == 1, "Invalid stages");

        auto smem_ptr = smem_buffer + warp_id * (kNumStages * (kNumTMABufferBytes + 16) + kNumMetaBytes);
        uint32_t tma_phase = 0;
        auto tma_buffers   = PatternVisitor([=](const int& i) { return reinterpret_cast<int4*>(smem_ptr + i * (kNumTMABufferBytes + 16)); });
        auto full_barriers = PatternVisitor([=](const int& i) { return reinterpret_cast<uint64_t*>(smem_ptr + i * (kNumTMABufferBytes + 16) + kNumTMABufferBytes); });
        auto meta_buffers  = kUseLogFMT ? reinterpret_cast<nv_bfloat162*>(smem_ptr + kNumStages * (kNumTMABufferBytes + 16)) : nullptr;
        EP_STATIC_ASSERT(kNumSendUnrolls * kNumStages <= 12, "TMA buffer size exceed limit");

        // Initialize m-barriers
        if (lane_id < kNumStages) {
            mbarrier_init(full_barriers[lane_id], 1);
            fence_view_async_shared();
            fence_barrier_init();
        }
        __syncwarp();

        constexpr int kNumIters = hidden_bf16_int4_pad / (32 * kNumSendUnrolls);
        auto tma_load_and_arrive = [&](const int& stage_idx, const int4* gmem_ptr, const int& num_bytes) {
            tma_load_1d(tma_buffers[stage_idx], gmem_ptr, full_barriers[stage_idx], num_bytes);
            mbarrier_arrive_and_expect_tx(full_barriers[stage_idx], num_bytes);
        };
        auto get_num_tma_bytes = [&](const int& offset_int4) {
            return min(kNumTMABufferBytes, static_cast<int>((hidden_bf16_int4 - offset_int4) * sizeof(int4)));
        };
        // -------------------------------------------- END tma-related -----------------------------------------------

        const auto dst_rank = responsible_expert_idx / num_local_experts;

        // NOTE
        // before: "one warp group --- all tokens for one (dsk_rank, local_expert_idx)"
        // after: "multiple warp groups --- cooperate on tokens for one (dsk_rank, local_expert_idx)"
        for (int local_expert_idx = 0; local_expert_idx < num_local_experts; ++local_expert_idx) {
            // NOTE changed
            // const auto local_expert_idx = responsible_expert_idx % num_local_experts;
            const auto token_cooperate_part_idx = responsible_expert_idx % num_local_experts;
            const auto num_token_cooperate_parts = num_local_experts;

            const auto global_expert_idx = rank * num_local_experts + local_expert_idx;
            const auto layout = __ldg(layout_range + local_expert_idx * num_ranks + dst_rank);
            const auto local_x = static_cast<const int4*>(x) +
                    local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * hidden_bf16_int4;
            const auto local_src_info = src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
            const auto rdma_send_x_vec = static_cast<uint8_t*>(rdma_send_x) +
                    local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_slot;

            // Unpack layout
            int offset, num_tokens_to_send;
            unpack2(layout, num_tokens_to_send, offset);

            // NOTE added
            if (src_signals != nullptr) {
                // TODO shall we let 1st expert be separately computed and then do *not* wait for it
                if (threadIdx.x == 0) {
                    wait_signal(src_signals + local_expert_idx, src_signal_expect_value);
                }

                __syncthreads();
            }

            // Issue IBGDA send
            // NOTE changed
            // for (int token_idx = offset + sub_warp_id; token_idx < offset + num_tokens_to_send; token_idx += num_warps_per_group) {
            const int num_tokens_to_send_per_cooperate_part = ceil_div(num_tokens_to_send, num_token_cooperate_parts);
            const int token_idx_part_end = offset + min(num_tokens_to_send, num_tokens_to_send_per_cooperate_part * (token_cooperate_part_idx + 1));
            for (
                int token_idx = offset + num_tokens_to_send_per_cooperate_part * token_cooperate_part_idx + sub_warp_id;
                token_idx < token_idx_part_end;
                token_idx += num_warps_per_group
            ) {
                const auto x_int4 = local_x + token_idx * hidden_bf16_int4;
                const auto rdma_send_type_row = reinterpret_cast<int*>(rdma_send_x_vec + token_idx * num_bytes_per_slot);
                const auto rdma_send_x_vec_row = reinterpret_cast<uint8_t*>(rdma_send_type_row);

                // Copy directly to local rank, or copy to buffer and issue RDMA
                const auto src_idx = __shfl_sync(0xffffffff, __ldg(local_src_info + token_idx), 0);
                const auto buf_ptr = reinterpret_cast<int64_t>(rdma_send_x_vec_row);
                const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x) + (global_expert_idx * num_max_dispatch_tokens_per_rank + src_idx) * num_bytes_per_slot;
                const auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
                int num_send_bytes = hidden * sizeof(nv_bfloat16);

                if (not zero_copy or dst_p2p_ptr != 0) {
                    // Read from `cpy_src_int4_ptr` and copy into `cpy_dst_int4_ptr`
                    const auto cpy_src_int4_ptr = zero_copy ? reinterpret_cast<int4*>(buf_ptr) : x_int4;
                    const auto cpy_dst_int4_ptr = dst_p2p_ptr == 0 ? reinterpret_cast<int4*>(buf_ptr) : reinterpret_cast<int4*>(dst_p2p_ptr);

                    // Prefetch
                    if (elect_one_sync())
                        tma_load_and_arrive(0, cpy_src_int4_ptr, get_num_tma_bytes(0));
                    __syncwarp();

                    int tma_offset_bytes = kNumMetaBytes;
                    #pragma unroll
                    for (int i = lane_id * kNumSendUnrolls, iter_idx = 0; i < hidden_bf16_int4_pad; i += 32 * kNumSendUnrolls, ++ iter_idx) {
                        // Load the next iteration
                        const int& stage_idx = iter_idx % kNumStages;
                        const int& next_stage_idx = (iter_idx + 1) % kNumStages;
                        if (iter_idx + 1 < kNumIters and elect_one_sync()) {
                            tma_store_wait<kNumStages - kNumPrefetch - 1>();
                            const auto& offset_int4 = i + 32 * kNumSendUnrolls;
                            tma_load_and_arrive(next_stage_idx, cpy_src_int4_ptr + offset_int4, get_num_tma_bytes(offset_int4));
                        }
                        __syncwarp();

                        // Wait the current TMA arrival
                        EP_STATIC_ASSERT(kNumStages < 32, "Too many stages");
                        mbarrier_wait<true>(full_barriers[stage_idx], tma_phase, stage_idx);
                        if constexpr (kUseLogFMT) {
                            // Cast if possible
                            constexpr int kNumInt4PerDivision = 128 / kNumElemsPerInt4;
                            int num_tma_bytes = logfmt_encode<kNumSendUnrolls>(
                                tma_buffers[stage_idx],
                                // NOTES: only the leader lane will write the result
                                (i % kNumInt4PerDivision == 0) ? meta_buffers + i / kNumInt4PerDivision : nullptr,
                                lane_id);
                            if (elect_one_sync())
                                tma_store_1d(tma_buffers[stage_idx], reinterpret_cast<uint8_t*>(cpy_dst_int4_ptr) + tma_offset_bytes, num_tma_bytes);
                            tma_offset_bytes += num_tma_bytes;
                        } else {
                            // BF16 original values
                            if (elect_one_sync())
                                tma_store_1d(tma_buffers[stage_idx], cpy_dst_int4_ptr + i, get_num_tma_bytes(i));
                        }
                        __syncwarp();
                    }

                    // Store metadata (min/max values) for LogFMT
                    if constexpr (kUseLogFMT) {
                        num_send_bytes = tma_offset_bytes;
                        if (elect_one_sync())
                            tma_store_1d(meta_buffers, cpy_dst_int4_ptr, kNumMetaBytes);
                    }

                    // Flush all stores
                    tma_store_wait<0>();
                    __syncwarp();
                }

                // Issue RDMA
                // NOTES: for zero-copy mode, we assume the data is already in the send buffer
                if (dst_p2p_ptr == 0)
                    nvshmemi_ibgda_put_nbi_warp(dst_ptr, buf_ptr, num_send_bytes, dst_rank, local_expert_idx, lane_id, token_idx - offset);
            }
        }

        // Put the finishing flag
        EP_DEVICE_ASSERT(num_warps_per_group > 1 and num_warp_groups < 16);
        asm volatile("bar.sync %0, %1;" :: "r"(warp_group_id + 1), "r"(num_warps_per_group * 32));
        if (sub_warp_id == 1 and lane_id == 0) {
            // copied from global to this part
            const auto local_expert_idx_for_signal = responsible_expert_idx % num_local_experts;
            const auto global_expert_idx_for_signal = rank * num_local_experts + local_expert_idx_for_signal;
            // =============================================

            while (ld_acquire_global(atomic_clean_flag) == 0);
            auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_flag + global_expert_idx_for_signal);
            auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
            if (dst_p2p_ptr == 0) {
                // will not visit this branch
                // nvshmemi_ibgda_amo_nonfetch_add(reinterpret_cast<int*>(dst_ptr), 1, dst_rank, local_expert_idx);
                EP_DEVICE_ASSERT(0);
            } else {
                st_release_sys_global(reinterpret_cast<int*>(dst_p2p_ptr), 1);
            }
            atomic_add_release_global(atomic_clean_flag, -1);
        }
        __syncwarp();

        // Destroy m-barriers
        if (lane_id < kNumStages) {
            mbarrier_inval(full_barriers[lane_id]);
            fence_view_async_shared();
            fence_barrier_init();
        }
        __syncwarp();
    } else {
        // NOTE add
        for (int local_expert_idx = 0; local_expert_idx < num_local_experts; ++local_expert_idx) {
            if (src_signals != nullptr) {
              // TODO original code uses NamedBarrier, better than this?
              __syncthreads();
            }
        }
    }

    // Receiving phase
    LOW_LATENCY_COMBINE_RECV:
    if ((phases & LEGACY_LOW_LATENCY_RECV_PHASE) == 0)
        return;

    // Wait all ranks to arrive
    if (responsible_expert_idx < num_experts) {
        EP_DEVICE_ASSERT(num_warps_per_group > 1);
        if (sub_warp_id == 0 and lane_id == 0) {
            auto start_time = clock64();
            while (ld_acquire_sys_global(rdma_recv_flag + responsible_expert_idx) == 0);
            auto wait_recv_cost = clock64() - start_time;
            if (combine_wait_recv_cost_stats != nullptr) {
                const auto& src_rank = responsible_expert_idx / num_local_experts;
                atomicAdd(reinterpret_cast<unsigned long long*>(combine_wait_recv_cost_stats + src_rank), wait_recv_cost);
            }
        }
    }
    cg::this_grid().sync();

    // Reassign warp groups
    constexpr int kMaxNumGroups = 2;
    const int num_decode_warps = hidden_bf16_int4_pad / (kNumRecvUnrolls * 32);
    const int num_groups = min(kMaxNumGroups, (num_threads / 32) / (num_decode_warps + 1));
    const int decode_warp_idx = __shfl_sync(0xffffffff, warp_id % (num_decode_warps + 1), 0);
    const int group_idx = __shfl_sync(0xffffffff, warp_id / (num_decode_warps + 1), 0);
    EP_STATIC_ASSERT(kHidden % (32 * kNumElemsPerInt4) == 0, "Invalid vectorization");
    EP_DEVICE_ASSERT(num_topk <= 32);
    EP_DEVICE_ASSERT(num_groups > 0);

    if (group_idx < num_groups) {
        constexpr int kNumStages = 3;
        constexpr int kNumTMABufferBytes = 16 * 2 + kHidden * 2;
        constexpr int kNumBF16PerWarpBytes = 32 * kNumRecvUnrolls * kNumElemsPerInt4 * 2;
        constexpr int kNumLogFMTPerWarpBytes = kNumBF16PerWarpBytes / 16 * 10;
        constexpr int kNumDivisionBytes = kNumDivisions * sizeof(uint32_t);
        constexpr int kNumBytesPerGroup = kNumStages * kNumTMABufferBytes + kHidden * 2 + kNumStages * kNumDivisionBytes * 3;

        // Reallocate shared memory
        const auto smem_group_buffer = smem_buffer + kNumBytesPerGroup * group_idx;
        auto full_barriers  = PatternVisitor([=](const int& i) { return reinterpret_cast<uint64_t*>(smem_group_buffer + i * kNumTMABufferBytes); });
        auto empty_barriers = PatternVisitor([=](const int& i) { return reinterpret_cast<uint64_t*>(smem_group_buffer + i * kNumTMABufferBytes + 8); });
        auto tma_ld_buffers = PatternVisitor([=](const int& i) { return reinterpret_cast<uint8_t* >(smem_group_buffer + i * kNumTMABufferBytes + 16); });
        auto tma_st_buffers = PatternVisitor([=](const int& i) { return reinterpret_cast<uint32_t*>(smem_group_buffer + kNumStages * kNumTMABufferBytes + i * kNumBF16PerWarpBytes); });

        // Redundant when logfmt is disabled
        const auto smem_group_ptr = smem_group_buffer + kNumStages * kNumTMABufferBytes + kHidden * 2;
        auto log_amax_buffers  = PatternVisitor([=](const int& i) { return reinterpret_cast<float*>(smem_group_ptr + i * kNumDivisionBytes); });
        auto log_amin_buffers  = PatternVisitor([=](const int& i) { return reinterpret_cast<float*>(smem_group_ptr + kNumStages * kNumDivisionBytes + i * kNumDivisionBytes); });
        auto cast_info_buffers = PatternVisitor([=](const int& i) { return reinterpret_cast<int*>  (smem_group_ptr + kNumStages * kNumDivisionBytes * 2 + i * kNumDivisionBytes); });

        uint32_t tma_phase = 0;
        EP_STATIC_ASSERT(kNumStages < 32, "Too many stages");
        if (decode_warp_idx == num_decode_warps)
            tma_phase = (1 << kNumStages) - 1;

        // Initialize m-barriers
        if (decode_warp_idx == num_decode_warps and lane_id < kNumStages) {
            mbarrier_init(full_barriers[lane_id], 1);
            mbarrier_init(empty_barriers[lane_id], num_decode_warps);
        }
        asm volatile("bar.sync %0, %1;" :: "r"(group_idx + 1), "r"((num_decode_warps + 1) * 32));

        int stage_idx = 0, topk_idx_by_lane = 0;
        EP_STATIC_ASSERT(kNumMaxTopk <= 32, "Invalid number of topks");
        if (decode_warp_idx == num_decode_warps) {
            // TMA load warp
            for (int token_idx = sm_id + num_sms * group_idx; token_idx < num_combined_tokens; token_idx += num_sms * num_groups) {
                if (lane_id < num_topk)
                    topk_idx_by_lane = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + lane_id));
                for (int i = 0; i < num_topk; ++ i) {
                    int topk_idx_reg = __shfl_sync(0xffffffff, topk_idx_by_lane, i);
                    if (topk_idx_reg < 0)
                        continue;

                    mbarrier_wait<true>(empty_barriers[stage_idx], tma_phase, stage_idx);
                    auto buffer = static_cast<uint8_t*>(rdma_recv_x) + (topk_idx_reg * num_max_dispatch_tokens_per_rank + token_idx) * num_bytes_per_slot;
                    if constexpr (kUseLogFMT) {
                        logfmt_check_amaxmin<kNumDivisions / 2, kNumSendUnrolls, kNumRecvUnrolls>(
                            buffer, reinterpret_cast<float2*>(log_amax_buffers[stage_idx]),
                            reinterpret_cast<float2*>(log_amin_buffers[stage_idx]), cast_info_buffers[stage_idx], lane_id);
                    }
                    if (elect_one_sync()) {
                        int num_casted = 0;
                        if constexpr (kUseLogFMT) {
                            const auto& info = cast_info_buffers[stage_idx][num_decode_warps - 1];
                            num_casted = (info >> 1) + (info & 1);
                        }
                        int num_tma_bytes = num_casted * kNumLogFMTPerWarpBytes + (num_decode_warps - num_casted) * kNumBF16PerWarpBytes;
                        tma_load_1d(tma_ld_buffers[stage_idx], buffer + (kUseLogFMT ? kNumMetaBytes : 0), full_barriers[stage_idx], num_tma_bytes);
                        mbarrier_arrive_and_expect_tx(full_barriers[stage_idx], num_tma_bytes);
                    }
                    __syncwarp();
                    stage_idx = (stage_idx + 1) % kNumStages;
                }
            }
        } else {
            // Reduction warps
            float topk_weights_by_lane;
            for (int token_idx = sm_id + num_sms * group_idx; token_idx < num_combined_tokens; token_idx += num_sms * num_groups) {
                if (lane_id < num_topk) {
                    topk_idx_by_lane = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + lane_id));
                    topk_weights_by_lane = __ldg(topk_weights + token_idx * num_topk + lane_id);
                }
                __syncwarp();

                float combined_values[kNumElemsPerInt4 * kNumRecvUnrolls] = {0.0f};
                for (int i = 0; i < num_topk; ++ i) {
                    if (__shfl_sync(0xffffffff, topk_idx_by_lane, i) < 0)
                        continue;
                    const auto& topk_weight = __shfl_sync(0xffffffff, topk_weights_by_lane, i);

                    mbarrier_wait<true>(full_barriers[stage_idx], tma_phase, stage_idx);
                    if constexpr (kUseLogFMT) {
                        const auto& info = cast_info_buffers[stage_idx][decode_warp_idx];
                        bool enable_cast = info & 1;
                        int num_casted_prefix = info >> 1;
                        int tma_offset = kNumLogFMTPerWarpBytes * num_casted_prefix + kNumBF16PerWarpBytes * (decode_warp_idx - num_casted_prefix);
                        int division_idx = decode_warp_idx * (kNumRecvUnrolls * 2) + lane_id * kNumRecvUnrolls / 16;
                        decode_and_accumulate<kNumRecvUnrolls>(
                            reinterpret_cast<uint32_t*>(tma_ld_buffers[stage_idx] + tma_offset + (enable_cast ? kNumLogFMTPerWarpBytes : kNumBF16PerWarpBytes) / 32 * lane_id),
                            combined_values, log_amax_buffers[stage_idx][division_idx], log_amin_buffers[stage_idx][division_idx], enable_cast, topk_weight
                        );
                    } else {
                        int tma_offset = kNumBF16PerWarpBytes * decode_warp_idx;
                        decode_and_accumulate<kNumRecvUnrolls>(
                            reinterpret_cast<uint32_t*>(tma_ld_buffers[stage_idx] + tma_offset + kNumBF16PerWarpBytes / 32 * lane_id),
                            combined_values, 0, 0, false, topk_weight
                        );
                    }

                    if (elect_one_sync())
                        mbarrier_arrive(empty_barriers[stage_idx]);
                    stage_idx = (stage_idx + 1) % kNumStages;
                }
                tma_store_wait<0>();

                #pragma unroll
                for (int k = 0; k < kNumRecvUnrolls * 4; ++ k) {
                    auto combined_pack = __nv_bfloat162(combined_values[k * 2], combined_values[k * 2 + 1]);
                    tma_st_buffers[decode_warp_idx][kNumRecvUnrolls * 4 * lane_id + k] = *reinterpret_cast<uint32_t*>(&combined_pack);
                }
                tma_store_fence();
                if (elect_one_sync()) {
                    tma_store_1d(tma_st_buffers[decode_warp_idx],
                                 static_cast<int4*>(combined_x) + token_idx * hidden_bf16_int4 + decode_warp_idx * kNumRecvUnrolls * 32,
                                 kNumBF16PerWarpBytes);
                }
                __syncwarp();
            }
        }

        // Flush all stores
        tma_store_wait<0>();
    }
}

void combine_overlappable(void* combined_x,
             void* rdma_recv_x, int* rdma_recv_flag, void* rdma_send_x,
             const void* x, const topk_idx_t* topk_idx, const float* topk_weights,
             const int* src_info, const int64_t* layout_range,
             int* mask_buffer_ptr,
             int64_t* combine_wait_recv_cost_stats,
             int* next_clean, int num_next_clean_int,
             int num_combined_tokens, int hidden, int num_max_dispatch_tokens_per_rank,
             int num_topk, int num_experts, int rank, int num_ranks,
             bool use_logfmt,
             void* workspace, int num_device_sms,
             cudaStream_t stream, int phases, bool zero_copy,
             uint32_t* src_signals, uint32_t src_signal_expect_value) {
    // Shrink mode was added after the fine-grained overlap kernel. Do not silently
    // bypass its rank mask until the overlap path supports the same semantics.
    EP_HOST_ASSERT(mask_buffer_ptr == nullptr);

    // NOTE reduce combine_send num sm
    if ((phases & LEGACY_LOW_LATENCY_RECV_PHASE) == 0) {
        // TODO let it be configurable
        num_device_sms = 32;
    }

    constexpr int kNumMaxTopk = 9;
    const int num_warp_groups = ceil_div(num_experts, num_device_sms);
    const int num_warps_per_group = 32 / num_warp_groups;
    const int num_recv_per_sm = ceil_div(num_combined_tokens, num_device_sms);
    EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 0 and ((num_combined_tokens == 0) or (num_recv_per_sm > 0)));

    const auto num_warps = num_warp_groups * num_warps_per_group;
    const auto num_sms =
        max(ceil_div(num_experts, num_warp_groups), num_recv_per_sm == 0 ? 1 : ceil_div(num_combined_tokens, num_recv_per_sm));

    // Check workspace
    auto atomic_clean_flag = static_cast<int*>(workspace);
    EP_HOST_ASSERT(sizeof(int) <= LEGACY_NUM_WORKSPACE_BYTES);
    EP_HOST_ASSERT(num_topk <= kNumMaxTopk);

    // Online cast cannot use zero-copy
    EP_HOST_ASSERT(not (zero_copy and use_logfmt));

    constexpr int kNumStages = 3;
    constexpr int kNumMaxUnrolls = 4;
    constexpr int kMaxNumGroups = 2;

    // Send buffer size
    const int num_meta_bytes = hidden / 128 * 4;
    const int num_send_tma_bytes = 32 * sizeof(int4) * kNumMaxUnrolls + 16;
    const int smem_send_size = num_warps * (kNumStages * num_send_tma_bytes + num_meta_bytes);

    // Receive buffer size
    const int num_recv_tma_bytes = 16 + hidden * 2;
    const int smem_recv_size = kMaxNumGroups * (kNumStages * num_recv_tma_bytes + hidden * 2 + kNumStages * num_meta_bytes * 3);

    // Total requirement
    const int smem_size = max(smem_send_size, smem_recv_size);

#define COMBINE_LAUNCH_CASE(hidden) { \
auto combine_func = use_logfmt ? \
    combine_overlappable<true, hidden, kNumMaxTopk, kNumMaxUnrolls> : \
    combine_overlappable<false, hidden, kNumMaxTopk, kNumMaxUnrolls>; \
SET_SHARED_MEMORY_FOR_TMA(combine_func); \
LAUNCH_KERNEL(&cfg, combine_func, \
              combined_x, \
              rdma_recv_x, rdma_recv_flag, rdma_send_x, \
              x, topk_idx, topk_weights, src_info, layout_range, \
              combine_wait_recv_cost_stats, \
              next_clean, num_next_clean_int, \
              atomic_clean_flag, \
              num_combined_tokens, hidden, num_topk, \
              num_max_dispatch_tokens_per_rank, \
              num_experts, rank, num_ranks, \
              num_warp_groups, num_warps_per_group, \
              phases, zero_copy, \
              src_signals, src_signal_expect_value); } break

    SETUP_LAUNCH_CONFIG(num_sms, num_warps * 32, stream);
    SWITCH_HIDDEN(COMBINE_LAUNCH_CASE);
#undef COMBINE_LAUNCH_CASE
}

template <bool kUseLogFMT, int kHidden, int kNumMaxTopk, int kNumMaxUnrolls>
__global__ __launch_bounds__(1024, 1) void
combine(void* combined_x,
        void* rdma_recv_x, int* rdma_recv_flag, void* rdma_send_x,
        const void* x, const topk_idx_t* topk_idx, const float* topk_weights,
        const int* src_info, const int64_t* layout_range,
        int* mask_buffer_ptr,
        int64_t* combine_wait_recv_cost_stats,
        int* next_clean, int num_next_clean_int,
        int* atomic_clean_flag,
        int num_combined_tokens, int hidden, int num_topk,
        int num_max_dispatch_tokens_per_rank,
        int num_experts, int rank, int num_ranks,
        int num_warp_groups, int num_warps_per_group,
        int phases, bool zero_copy) {
    const auto sm_id = __shfl_sync(0xffffffff, static_cast<int>(blockIdx.x), 0);
    const auto num_sms = __shfl_sync(0xffffffff, static_cast<int>(gridDim.x), 0);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto num_threads = __shfl_sync(0xffffffff, static_cast<int>(blockDim.x), 0);
    const auto warp_id = __shfl_sync(0xffffffff, thread_id / 32, 0), lane_id = get_lane_id();
    const auto num_local_experts = num_experts / num_ranks;
    const auto warp_group_id = warp_id / num_warps_per_group;
    const auto sub_warp_id = warp_id % num_warps_per_group;
    const auto responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;

    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // Data type staffs
    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
    constexpr int64_t hidden_bf16_int4 = kHidden / kNumElemsPerInt4;

    // Use different unroll factors for send and recv phases
    constexpr int kNumSendUnrolls = kHidden % (32 * 4 * sizeof(int4) / sizeof(nv_bfloat16)) == 0 ? 4 : 2;
    constexpr int kNumRecvUnrolls = 2;
    constexpr int hidden_bf16_int4_pad = align_up(static_cast<int>(hidden_bf16_int4), 32 * kNumSendUnrolls);
    EP_STATIC_ASSERT(kHidden % (32 * 2 * sizeof(int4) / sizeof(nv_bfloat16)) == 0, "Invalid hidden");
    EP_STATIC_ASSERT(kNumSendUnrolls <= kNumMaxUnrolls and kNumRecvUnrolls <= kNumMaxUnrolls, "Invalid unrolls");
    EP_STATIC_ASSERT(hidden_bf16_int4 % kNumSendUnrolls == 0, "Invalid hidden");
    EP_STATIC_ASSERT(kNumSendUnrolls >= kNumRecvUnrolls, "Invalid unroll factors");

    // Message package
    EP_STATIC_ASSERT(kHidden % 128 == 0, "Invalid hidden");
    constexpr int kNumDivisions = kHidden / 128;
    constexpr int kNumMetaBytes = kNumDivisions * sizeof(nv_bfloat162);
    constexpr size_t num_bytes_per_slot = kHidden * sizeof(nv_bfloat16) + kNumMetaBytes;
    EP_STATIC_ASSERT(num_bytes_per_slot % sizeof(int4) == 0, "Invalid vectorization");

    // Sending phase
    if ((phases & LEGACY_LOW_LATENCY_SEND_PHASE) == 0)
        goto LOW_LATENCY_COMBINE_RECV;

    // Clean up next buffer
    if (sm_id == 0 and warp_group_id == 0 and sub_warp_id == 0) {
        #pragma unroll
        for (int i = lane_id; i < num_next_clean_int; i += 32)
            next_clean[i] = 0;

        // Notify before executing `int_p`
        __syncwarp();
        if (lane_id == 0)
            atomic_add_release_global(atomic_clean_flag, num_experts);
    }

    // Issue IBGDA sends
    if (responsible_expert_idx < num_experts) {
        const auto dst_rank = responsible_expert_idx / num_local_experts;
        const auto local_expert_idx = responsible_expert_idx % num_local_experts;
        const auto global_expert_idx = rank * num_local_experts + local_expert_idx;
        const auto layout = __ldg(layout_range + local_expert_idx * num_ranks + dst_rank);
        const auto local_x =
            static_cast<const int4*>(x) + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * hidden_bf16_int4;
        const auto local_src_info = src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
        const auto rdma_send_x_vec =
            static_cast<uint8_t*>(rdma_send_x) + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_slot;

        // Unpack layout
        int offset, num_tokens_to_send;
        unpack2(layout, num_tokens_to_send, offset);

        // TMA stuffs
        constexpr int kNumTMABufferBytes = sizeof(int4) * 32 * kNumSendUnrolls;
        constexpr int kNumStages = 3;
        constexpr int kNumPrefetch = 1;
        EP_STATIC_ASSERT(kNumStages == 3 and kNumPrefetch == 1, "Invalid stages");

        auto smem_ptr = smem_buffer + warp_id * (kNumStages * (kNumTMABufferBytes + 16) + kNumMetaBytes);
        uint32_t tma_phase = 0;
        auto tma_buffers = PatternVisitor([=](const int& i) { return reinterpret_cast<int4*>(smem_ptr + i * (kNumTMABufferBytes + 16)); });
        auto full_barriers = PatternVisitor(
            [=](const int& i) { return reinterpret_cast<uint64_t*>(smem_ptr + i * (kNumTMABufferBytes + 16) + kNumTMABufferBytes); });
        auto meta_buffers = kUseLogFMT ? reinterpret_cast<nv_bfloat162*>(smem_ptr + kNumStages * (kNumTMABufferBytes + 16)) : nullptr;
        EP_STATIC_ASSERT(kNumSendUnrolls * kNumStages <= 12, "TMA buffer size exceed limit");

        // Initialize m-barriers
        if (lane_id < kNumStages) {
            mbarrier_init(full_barriers[lane_id], 1);
            fence_barrier_init();
        }
        __syncwarp();

        constexpr int kNumIters = hidden_bf16_int4_pad / (32 * kNumSendUnrolls);
        auto tma_load_and_arrive = [&](const int& stage_idx, const int4* gmem_ptr, const int& num_bytes) {
            tma_load_1d(tma_buffers[stage_idx], gmem_ptr, full_barriers[stage_idx], num_bytes);
            mbarrier_arrive_and_expect_tx(full_barriers[stage_idx], num_bytes);
        };
        auto get_num_tma_bytes = [&](const int& offset_int4) {
            return min(kNumTMABufferBytes, static_cast<int>((hidden_bf16_int4 - offset_int4) * sizeof(int4)));
        };

        // Issue IBGDA send
        if (not is_rank_masked<true>(mask_buffer_ptr, dst_rank)) {
            for (int token_idx = offset + sub_warp_id; token_idx < offset + num_tokens_to_send; token_idx += num_warps_per_group) {
                const auto x_int4 = local_x + token_idx * hidden_bf16_int4;
                const auto rdma_send_type_row = reinterpret_cast<int*>(rdma_send_x_vec + token_idx * num_bytes_per_slot);
                const auto rdma_send_x_vec_row = reinterpret_cast<uint8_t*>(rdma_send_type_row);

                // Copy directly to local rank, or copy to buffer and issue RDMA
                const auto src_idx = __shfl_sync(0xffffffff, __ldg(local_src_info + token_idx), 0);
                const auto buf_ptr = reinterpret_cast<int64_t>(rdma_send_x_vec_row);
                const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x) +
                    (global_expert_idx * num_max_dispatch_tokens_per_rank + src_idx) * num_bytes_per_slot;
                const auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
                int num_send_bytes = hidden * sizeof(nv_bfloat16);

                if (not zero_copy or dst_p2p_ptr != 0) {
                    // Read from `cpy_src_int4_ptr` and copy into `cpy_dst_int4_ptr`
                    const auto cpy_src_int4_ptr = zero_copy ? reinterpret_cast<int4*>(buf_ptr) : x_int4;
                    const auto cpy_dst_int4_ptr =
                        dst_p2p_ptr == 0 ? reinterpret_cast<int4*>(buf_ptr) : reinterpret_cast<int4*>(dst_p2p_ptr);

                    // Prefetch
                    if (elect_one_sync())
                        tma_load_and_arrive(0, cpy_src_int4_ptr, get_num_tma_bytes(0));
                    __syncwarp();

                    int tma_offset_bytes = kNumMetaBytes;
                    #pragma unroll
                    for (int i = lane_id * kNumSendUnrolls, iter_idx = 0; i < hidden_bf16_int4_pad; i += 32 * kNumSendUnrolls, ++iter_idx) {
                        // Load the next iteration
                        const int stage_idx = iter_idx % kNumStages;
                        const int next_stage_idx = (iter_idx + 1) % kNumStages;
                        if (iter_idx + 1 < kNumIters and elect_one_sync()) {
                            tma_store_wait<kNumStages - kNumPrefetch - 1>();
                            const auto offset_int4 = i + 32 * kNumSendUnrolls;
                            tma_load_and_arrive(next_stage_idx, cpy_src_int4_ptr + offset_int4, get_num_tma_bytes(offset_int4));
                        }
                        __syncwarp();

                        // Wait the current TMA arrival
                        EP_STATIC_ASSERT(kNumStages < 32, "Too many stages");
                        mbarrier_wait<true>(full_barriers[stage_idx], tma_phase, stage_idx);
                        if constexpr (kUseLogFMT) {
                            // Cast if possible
                            constexpr int kNumInt4PerDivision = 128 / kNumElemsPerInt4;
                            int num_tma_bytes = logfmt_encode<kNumSendUnrolls>(
                                tma_buffers[stage_idx],
                                // NOTES: only the leader lane will write the result
                                (i % kNumInt4PerDivision == 0) ? meta_buffers + i / kNumInt4PerDivision : nullptr,
                                lane_id);
                            if (elect_one_sync())
                                tma_store_1d(
                                    tma_buffers[stage_idx], reinterpret_cast<uint8_t*>(cpy_dst_int4_ptr) + tma_offset_bytes, num_tma_bytes);
                            tma_offset_bytes += num_tma_bytes;
                        } else {
                            // BF16 original values
                            if (elect_one_sync())
                                tma_store_1d(tma_buffers[stage_idx], cpy_dst_int4_ptr + i, get_num_tma_bytes(i));
                        }
                        __syncwarp();
                    }

                    // Store metadata (min/max values) for LogFMT
                    if constexpr (kUseLogFMT) {
                        num_send_bytes = tma_offset_bytes;
                        if (elect_one_sync())
                            tma_store_1d(meta_buffers, cpy_dst_int4_ptr, kNumMetaBytes);
                    }

                    // Flush all stores
                    tma_store_wait<0>();
                    __syncwarp();
                }

                // Issue RDMA
                // NOTES: for zero-copy mode, we assume the data is already in the send buffer
                if (dst_p2p_ptr == 0)
                    nvshmemi_ibgda_put_nbi_warp(dst_ptr, buf_ptr, num_send_bytes, dst_rank, local_expert_idx, lane_id, token_idx - offset);
            }
        }

        // Put the finishing flag
        EP_DEVICE_ASSERT(num_warps_per_group > 1 and num_warp_groups < 16);
        asm volatile("bar.sync %0, %1;" ::"r"(warp_group_id + 1), "r"(num_warps_per_group * 32));
        if (sub_warp_id == 1 and lane_id == 0) {
            while (ld_acquire_global(atomic_clean_flag) == 0)
                ;
            auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_flag + global_expert_idx);
            auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
            if (not is_rank_masked(mask_buffer_ptr, dst_rank)) {
                if (dst_p2p_ptr == 0) {
                    nvshmemi_ibgda_amo_nonfetch_add(reinterpret_cast<int*>(dst_ptr), 1, dst_rank, local_expert_idx);
                } else {
                    st_release_sys_global(reinterpret_cast<int*>(dst_p2p_ptr), 1);
                }
            }
            atomic_add_release_global(atomic_clean_flag, -1);
        }
        __syncwarp();

        // Destroy m-barriers
        if (lane_id < kNumStages) {
            mbarrier_inval(full_barriers[lane_id]);
            fence_barrier_init();
        }
        __syncwarp();
    }

// Receiving phase
LOW_LATENCY_COMBINE_RECV:
    if ((phases & LEGACY_LOW_LATENCY_RECV_PHASE) == 0)
        return;

    // Wait all ranks to arrive
    if (responsible_expert_idx < num_experts) {
        EP_DEVICE_ASSERT(num_warps_per_group > 1);
        if (sub_warp_id == 0 and lane_id == 0) {
            const auto src_rank = responsible_expert_idx / num_local_experts;
            auto start_time = clock64();
            uint64_t wait_recv_cost = 0;
            if (not is_rank_masked(mask_buffer_ptr, src_rank)) {
                while (ld_acquire_sys_global(rdma_recv_flag + responsible_expert_idx) == 0  // recv not ready
                       && (wait_recv_cost = clock64() - start_time) <= LEGACY_NUM_TIMEOUT_CYCLES   // not timeout
                )
                    ;
            }
            // Mask rank if timeout
            if (wait_recv_cost > LEGACY_NUM_TIMEOUT_CYCLES) {
                printf("Warning: DeepEP timeout for combine receive, rank %d, local_expert_idx %d, src_rank %d\n",
                       rank,
                       responsible_expert_idx % num_local_experts,
                       src_rank);
                if (mask_buffer_ptr == nullptr)
                    trap();
                atomicExch(mask_buffer_ptr + src_rank, 1);
            }

            if (combine_wait_recv_cost_stats != nullptr) {
                atomicAdd(reinterpret_cast<unsigned long long*>(combine_wait_recv_cost_stats + src_rank), wait_recv_cost);
            }
        }
    }
    cg::this_grid().sync();

    // Reassign warp groups
    constexpr int kMaxNumGroups = 2;
    const int num_decode_warps = hidden_bf16_int4_pad / (kNumRecvUnrolls * 32);
    const int num_groups = min(kMaxNumGroups, (num_threads / 32) / (num_decode_warps + 1));
    const int decode_warp_idx = __shfl_sync(0xffffffff, warp_id % (num_decode_warps + 1), 0);
    const int group_idx = __shfl_sync(0xffffffff, warp_id / (num_decode_warps + 1), 0);
    EP_STATIC_ASSERT(kHidden % (32 * kNumElemsPerInt4) == 0, "Invalid vectorization");
    EP_DEVICE_ASSERT(num_topk <= 32);
    EP_DEVICE_ASSERT(num_groups > 0);

    if (group_idx < num_groups) {
        constexpr int kNumStages = 3;
        constexpr int kNumTMABufferBytes = 16 * 2 + kHidden * 2;
        constexpr int kNumBF16PerWarpBytes = 32 * kNumRecvUnrolls * kNumElemsPerInt4 * 2;
        constexpr int kNumLogFMTPerWarpBytes = kNumBF16PerWarpBytes / 16 * 10;
        constexpr int kNumDivisionBytes = kNumDivisions * sizeof(uint32_t);
        constexpr int kNumBytesPerGroup = kNumStages * kNumTMABufferBytes + kHidden * 2 + kNumStages * kNumDivisionBytes * 3;

        // Reallocate shared memory
        const auto smem_group_buffer = smem_buffer + kNumBytesPerGroup * group_idx;
        auto full_barriers =
            PatternVisitor([=](const int& i) { return reinterpret_cast<uint64_t*>(smem_group_buffer + i * kNumTMABufferBytes); });
        auto empty_barriers =
            PatternVisitor([=](const int& i) { return reinterpret_cast<uint64_t*>(smem_group_buffer + i * kNumTMABufferBytes + 8); });
        auto tma_ld_buffers =
            PatternVisitor([=](const int& i) { return reinterpret_cast<uint8_t*>(smem_group_buffer + i * kNumTMABufferBytes + 16); });
        auto tma_st_buffers = PatternVisitor([=](const int& i) {
            return reinterpret_cast<uint32_t*>(smem_group_buffer + kNumStages * kNumTMABufferBytes + i * kNumBF16PerWarpBytes);
        });

        // Redundant when logfmt is disabled
        const auto smem_group_ptr = smem_group_buffer + kNumStages * kNumTMABufferBytes + kHidden * 2;
        auto log_amax_buffers =
            PatternVisitor([=](const int& i) { return reinterpret_cast<float*>(smem_group_ptr + i * kNumDivisionBytes); });
        auto log_amin_buffers = PatternVisitor([=](const int& i) {
            return reinterpret_cast<float*>(smem_group_ptr + kNumStages * kNumDivisionBytes + i * kNumDivisionBytes);
        });
        auto cast_info_buffers = PatternVisitor([=](const int& i) {
            return reinterpret_cast<int*>(smem_group_ptr + kNumStages * kNumDivisionBytes * 2 + i * kNumDivisionBytes);
        });

        uint32_t tma_phase = 0;
        EP_STATIC_ASSERT(kNumStages < 32, "Too many stages");
        if (decode_warp_idx == num_decode_warps)
            tma_phase = (1 << kNumStages) - 1;

        // Initialize m-barriers
        if (decode_warp_idx == num_decode_warps and lane_id < kNumStages) {
            mbarrier_init(full_barriers[lane_id], 1);
            mbarrier_init(empty_barriers[lane_id], num_decode_warps);
        }
        asm volatile("bar.sync %0, %1;" ::"r"(group_idx + 1), "r"((num_decode_warps + 1) * 32));

        int stage_idx = 0, topk_idx_by_lane = 0;
        EP_STATIC_ASSERT(kNumMaxTopk <= 32, "Invalid number of topks");
        if (decode_warp_idx == num_decode_warps) {
            // TMA load warp
            for (int token_idx = sm_id + num_sms * group_idx; token_idx < num_combined_tokens; token_idx += num_sms * num_groups) {
                if (lane_id < num_topk)
                    topk_idx_by_lane = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + lane_id));
                for (int i = 0; i < num_topk; ++i) {
                    int topk_idx_reg = __shfl_sync(0xffffffff, topk_idx_by_lane, i);
                    if (topk_idx_reg < 0)
                        continue;
                    if (is_rank_masked(mask_buffer_ptr, topk_idx_reg / num_local_experts))
                        continue;

                    mbarrier_wait<true>(empty_barriers[stage_idx], tma_phase, stage_idx);
                    auto buffer = static_cast<uint8_t*>(rdma_recv_x) +
                        (topk_idx_reg * num_max_dispatch_tokens_per_rank + token_idx) * num_bytes_per_slot;
                    if constexpr (kUseLogFMT) {
                        logfmt_check_amaxmin<kNumDivisions / 2, kNumSendUnrolls, kNumRecvUnrolls>(
                            buffer,
                            reinterpret_cast<float2*>(log_amax_buffers[stage_idx]),
                            reinterpret_cast<float2*>(log_amin_buffers[stage_idx]),
                            cast_info_buffers[stage_idx],
                            lane_id);
                    }
                    if (elect_one_sync()) {
                        int num_casted = 0;
                        if constexpr (kUseLogFMT) {
                            const auto info = cast_info_buffers[stage_idx][num_decode_warps - 1];
                            num_casted = (info >> 1) + (info & 1);
                        }
                        int num_tma_bytes = num_casted * kNumLogFMTPerWarpBytes + (num_decode_warps - num_casted) * kNumBF16PerWarpBytes;
                        tma_load_1d(
                            tma_ld_buffers[stage_idx], buffer + (kUseLogFMT ? kNumMetaBytes : 0), full_barriers[stage_idx], num_tma_bytes);
                        mbarrier_arrive_and_expect_tx(full_barriers[stage_idx], num_tma_bytes);
                    }
                    __syncwarp();
                    stage_idx = (stage_idx + 1) % kNumStages;
                }
            }
        } else {
            // Reduction warps
            float topk_weights_by_lane;
            for (int token_idx = sm_id + num_sms * group_idx; token_idx < num_combined_tokens; token_idx += num_sms * num_groups) {
                if (lane_id < num_topk) {
                    topk_idx_by_lane = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + lane_id));
                    topk_weights_by_lane = __ldg(topk_weights + token_idx * num_topk + lane_id);
                }
                __syncwarp();

                float combined_values[kNumElemsPerInt4 * kNumRecvUnrolls] = {0.0f};
                for (int i = 0; i < num_topk; ++i) {
                    int topk_idx_reg = __shfl_sync(0xffffffff, topk_idx_by_lane, i);
                    if (topk_idx_reg < 0)
                        continue;
                    if (is_rank_masked(mask_buffer_ptr, topk_idx_reg / num_local_experts))
                        continue;
                    const auto topk_weight = __shfl_sync(0xffffffff, topk_weights_by_lane, i);

                    mbarrier_wait<true>(full_barriers[stage_idx], tma_phase, stage_idx);
                    if constexpr (kUseLogFMT) {
                        const auto info = cast_info_buffers[stage_idx][decode_warp_idx];
                        bool enable_cast = info & 1;
                        int num_casted_prefix = info >> 1;
                        int tma_offset =
                            kNumLogFMTPerWarpBytes * num_casted_prefix + kNumBF16PerWarpBytes * (decode_warp_idx - num_casted_prefix);
                        int division_idx = decode_warp_idx * (kNumRecvUnrolls * 2) + lane_id * kNumRecvUnrolls / 16;
                        decode_and_accumulate<kNumRecvUnrolls>(
                            reinterpret_cast<uint32_t*>(tma_ld_buffers[stage_idx] + tma_offset +
                                                        (enable_cast ? kNumLogFMTPerWarpBytes : kNumBF16PerWarpBytes) / 32 * lane_id),
                            combined_values,
                            log_amax_buffers[stage_idx][division_idx],
                            log_amin_buffers[stage_idx][division_idx],
                            enable_cast,
                            topk_weight);
                    } else {
                        int tma_offset = kNumBF16PerWarpBytes * decode_warp_idx;
                        decode_and_accumulate<kNumRecvUnrolls>(
                            reinterpret_cast<uint32_t*>(tma_ld_buffers[stage_idx] + tma_offset + kNumBF16PerWarpBytes / 32 * lane_id),
                            combined_values,
                            0,
                            0,
                            false,
                            topk_weight);
                    }
                    fence_view_async_shared();
                    if (elect_one_sync())
                        mbarrier_arrive(empty_barriers[stage_idx]);
                    stage_idx = (stage_idx + 1) % kNumStages;
                }
                tma_store_wait<0>();

                #pragma unroll
                for (int k = 0; k < kNumRecvUnrolls * 4; ++k) {
                    auto combined_pack = __nv_bfloat162(combined_values[k * 2], combined_values[k * 2 + 1]);
                    tma_st_buffers[decode_warp_idx][kNumRecvUnrolls * 4 * lane_id + k] = *reinterpret_cast<uint32_t*>(&combined_pack);
                }
                tma_store_fence();
                if (elect_one_sync()) {
                    tma_store_1d(tma_st_buffers[decode_warp_idx],
                                 static_cast<int4*>(combined_x) + token_idx * hidden_bf16_int4 + decode_warp_idx * kNumRecvUnrolls * 32,
                                 kNumBF16PerWarpBytes);
                }
                __syncwarp();
            }
        }
    }
}

void combine(void* combined_x,
             void* rdma_recv_x,
             int* rdma_recv_flag,
             void* rdma_send_x,
             const void* x,
             const topk_idx_t* topk_idx,
             const float* topk_weights,
             const int* src_info,
             const int64_t* layout_range,
             int* mask_buffer_ptr,
             int64_t* combine_wait_recv_cost_stats,
             int* next_clean,
             int num_next_clean_int,
             int num_combined_tokens,
             int hidden,
             int num_max_dispatch_tokens_per_rank,
             int num_topk,
             int num_experts,
             int rank,
             int num_ranks,
             bool use_logfmt,
             void* workspace, int num_device_sms,
             cudaStream_t stream, int phases, bool zero_copy,
             bool overlap, uint32_t* src_signals, uint32_t src_signal_expect_value) {
    if (overlap) {
        return combine_overlappable(
            combined_x,
            rdma_recv_x, rdma_recv_flag, rdma_send_x,
            x, topk_idx, topk_weights,
            src_info, layout_range,
            mask_buffer_ptr,
            combine_wait_recv_cost_stats,
            next_clean, num_next_clean_int,
            num_combined_tokens, hidden, num_max_dispatch_tokens_per_rank,
            num_topk, num_experts, rank, num_ranks,
            use_logfmt,
            workspace, num_device_sms,
            stream, phases, zero_copy,
            src_signals, src_signal_expect_value
        );
    }

    constexpr int kNumMaxTopk = 16;  // Kimi-K3 routes top-16
    const int num_warp_groups = ceil_div(num_experts, num_device_sms);
    const int num_warps_per_group = 32 / num_warp_groups;
    const int num_recv_per_sm = ceil_div(num_combined_tokens, num_device_sms);
    EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 0 and num_recv_per_sm >= 0);

    const auto num_warps = num_warp_groups * num_warps_per_group;
    const auto num_sms =
        max(ceil_div(num_experts, num_warp_groups), num_recv_per_sm == 0 ? 1 : ceil_div(num_combined_tokens, num_recv_per_sm));

    // Check workspace
    auto atomic_clean_flag = static_cast<int*>(workspace);
    EP_HOST_ASSERT(sizeof(int) <= LEGACY_NUM_WORKSPACE_BYTES);
    EP_HOST_ASSERT(num_topk <= kNumMaxTopk);

    // Online cast cannot use zero-copy
    EP_HOST_ASSERT(not(zero_copy and use_logfmt));

    constexpr int kNumStages = 3;
    constexpr int kNumMaxUnrolls = 4;
    constexpr int kMaxNumGroups = 2;

    // Send buffer size
    const int num_meta_bytes = hidden / 128 * 4;
    const int num_send_tma_bytes = 32 * sizeof(int4) * kNumMaxUnrolls + 16;
    const int smem_send_size = num_warps * (kNumStages * num_send_tma_bytes + num_meta_bytes);

    // Receive buffer size
    const int num_recv_tma_bytes = 16 + hidden * 2;
    const int smem_recv_size = kMaxNumGroups * (kNumStages * num_recv_tma_bytes + hidden * 2 + kNumStages * num_meta_bytes * 3);

    // Total requirement
    const int smem_size = max(smem_send_size, smem_recv_size);

#define COMBINE_LAUNCH_CASE(hidden)                                                                                                \
    {                                                                                                                              \
        auto combine_func =                                                                                                        \
            use_logfmt ? combine<true, hidden, kNumMaxTopk, kNumMaxUnrolls> : combine<false, hidden, kNumMaxTopk, kNumMaxUnrolls>; \
        SET_SHARED_MEMORY_FOR_TMA(combine_func);                                                                                   \
        LAUNCH_KERNEL(&cfg,                                                                                                        \
                      combine_func,                                                                                                \
                      combined_x,                                                                                                  \
                      rdma_recv_x,                                                                                                 \
                      rdma_recv_flag,                                                                                              \
                      rdma_send_x,                                                                                                 \
                      x,                                                                                                           \
                      topk_idx,                                                                                                    \
                      topk_weights,                                                                                                \
                      src_info,                                                                                                    \
                      layout_range,                                                                                                \
                      mask_buffer_ptr,                                                                                             \
                      combine_wait_recv_cost_stats,                                                                                \
                      next_clean,                                                                                                  \
                      num_next_clean_int,                                                                                          \
                      atomic_clean_flag,                                                                                           \
                      num_combined_tokens,                                                                                         \
                      hidden,                                                                                                      \
                      num_topk,                                                                                                    \
                      num_max_dispatch_tokens_per_rank,                                                                            \
                      num_experts,                                                                                                 \
                      rank,                                                                                                        \
                      num_ranks,                                                                                                   \
                      num_warp_groups,                                                                                             \
                      num_warps_per_group,                                                                                         \
                      phases,                                                                                                      \
                      zero_copy);                                                                                                  \
    }                                                                                                                              \
    break

    SETUP_LAUNCH_CONFIG(num_sms, num_warps * 32, stream);
    SWITCH_HIDDEN(COMBINE_LAUNCH_CASE);
#undef COMBINE_LAUNCH_CASE
}

template <int kNumThreads>
__launch_bounds__(kNumThreads, 1) __global__ void query_mask_buffer(int* mask_buffer_ptr, int num_ranks, int* mask_tensor) {
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto num_threads = num_sms * kNumThreads;
    const auto thread_id = sm_id * kNumThreads + static_cast<int>(threadIdx.x);
    for (int rank_id = thread_id; rank_id < num_ranks; rank_id += num_threads)
        mask_tensor[rank_id] = mask_buffer_ptr[rank_id];
}

void query_mask_buffer(int* mask_buffer_ptr, int num_ranks, int* mask_tensor, cudaStream_t stream) {
    constexpr int num_sms = 1;
    constexpr int kNumThreads = 1024;
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    LAUNCH_KERNEL(&cfg, query_mask_buffer<kNumThreads>, mask_buffer_ptr, num_ranks, mask_tensor);
}

template <int kNumThreads>
__launch_bounds__(kNumThreads, 1) __global__ void update_mask_buffer(int* mask_buffer_ptr, int rank_to_mask, bool mask) {
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    if (sm_id == 0 and thread_id == 0)
        atomicExch(mask_buffer_ptr + rank_to_mask, mask ? 1 : 0);
}

void update_mask_buffer(int* mask_buffer_ptr, int rank, bool mask, cudaStream_t stream) {
    constexpr int num_sms = 1;
    constexpr int kNumThreads = 32;
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    LAUNCH_KERNEL(&cfg, update_mask_buffer<kNumThreads>, mask_buffer_ptr, rank, mask);
}

template <int kNumThreads>
__launch_bounds__(kNumThreads, 1) __global__ void clean_mask_buffer(int* mask_buffer_ptr, int num_ranks) {
    auto thread_id = static_cast<int>(threadIdx.x);
    #pragma unroll
    for (int i = thread_id; i < num_ranks; i += kNumThreads)
        mask_buffer_ptr[i] = 0;
}

void clean_mask_buffer(int* mask_buffer_ptr, int num_ranks, cudaStream_t stream) {
    constexpr int num_sms = 1;
    constexpr int kNumThreads = 32;
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    LAUNCH_KERNEL(&cfg, clean_mask_buffer<kNumThreads>, mask_buffer_ptr, num_ranks);
}

// Fence-free combine over NVLink P2P: every 16-byte unit carries 8 bytes of BF16 data and the call's flag twice,
// so a receiver accepts each unit on its own and no sender issues a system-scope fence.
// With cta_clean, every CTA clears the next buffer's signals itself and fences before its own sends,
// instead of waiting for CTA 0's clean through the shared clean flag.
constexpr int kPayloadFlagsNumSeqSlots = 256;

// Further options of the fast-send payload-flags combine, as bits of its `ext_opts`
// Group check: each peer's base is checked once per warp group, so an item's x loads do not wait on its src_info
constexpr int PAYLOAD_FLAGS_COMBINE_GROUP_CHECK = 2;
constexpr size_t kPayloadFlagsSeqOffsetBytes = 16 * 1024 * 1024;

__device__ __forceinline__ int payload_flags_value(const int& seq) {
    return static_cast<int>(static_cast<uint32_t>(seq) * 2654435761u) | 1;
}

__device__ __forceinline__ int payload_flags_topk(const topk_idx_t* topk_idx, const int& i, const bool& is_int32) {
    return is_int32 ? __ldg(reinterpret_cast<const int*>(topk_idx) + i) : static_cast<int>(__ldg(topk_idx + i));
}

__device__ __forceinline__ void st_volatile_global_v4(const int4* ptr, const int4& value) {
    asm volatile("st.volatile.global.v4.s32 [%0], {%1, %2, %3, %4};"
                 ::"l"(ptr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w) : "memory");
}

__device__ __forceinline__ int4 ld_volatile_global_v4(const int4* ptr) {
    int4 ret;
    asm volatile("ld.volatile.global.v4.s32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l"(ptr) : "memory");
    return ret;
}

template <int kHidden>
__global__ __launch_bounds__(1024, 1) void
combine_payload_flags(void* combined_x,
                      void* rdma_recv_x, void* rdma_send_x,
                      const void* x, const topk_idx_t* topk_idx, const float* topk_weights,
                      const int* src_info, const int64_t* layout_range,
                      int* next_clean, int num_next_clean_int,
                      int* atomic_clean_flag, int* seq_counters,
                      int num_combined_tokens, int num_topk,
                      int num_max_dispatch_tokens_per_rank,
                      int num_experts, int rank, int num_ranks,
                      int num_warp_groups, int num_warps_per_group,
                      int phases, bool cta_clean, bool tile_recv, bool fast_send, int ext_opts) {
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto warp_id = thread_id / 32, lane_id = get_lane_id();
    const auto num_local_experts = num_experts / num_ranks;
    const auto warp_group_id = warp_id / num_warps_per_group;
    const auto sub_warp_id = warp_id % num_warps_per_group;
    const auto responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;

    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
    constexpr int kHiddenInt4 = kHidden / kNumElemsPerInt4;
    constexpr int kNumElemsPerUnit = sizeof(int2) / sizeof(nv_bfloat16);
    constexpr int kNumUnits = kHidden / kNumElemsPerUnit;
    constexpr int kNumHalfUnits = kNumUnits / 2;
    constexpr int kNumChunks = 4;
    constexpr int kChunkUnits = (kNumUnits + kNumChunks - 1) / kNumChunks;
    constexpr size_t kNumBytesPerSlot = kHidden * sizeof(nv_bfloat16) + kHidden / 128 * sizeof(nv_bfloat162);
    EP_STATIC_ASSERT(kNumHalfUnits * sizeof(int4) <= kNumBytesPerSlot, "Half message exceeds a slot");

    // Fast send: the next buffer's signals and this group's layout are read in the same round trip as the sequence,
    // and the signals are cleared (with their fence) only when a shipped kernel left one set
    int is_next_dirty = 0;
    int64_t prefetched_layout = 0;
    if (fast_send and (phases & LEGACY_LOW_LATENCY_SEND_PHASE)) {
        for (int i = thread_id; i < num_next_clean_int; i += blockDim.x)
            is_next_dirty |= ld_volatile_global(next_clean + i) != 0;
        if (responsible_expert_idx < num_experts)
            prefetched_layout = __ldg(layout_range + (responsible_expert_idx % num_local_experts) * num_ranks +
                                      responsible_expert_idx / num_local_experts);
    }

    __shared__ int shared_seq;
    __shared__ int shared_recv_seq;
    if (thread_id == 0) {
        if (phases & LEGACY_LOW_LATENCY_SEND_PHASE)
            shared_seq = seq_counters[sm_id] + 1;
        if (phases & LEGACY_LOW_LATENCY_RECV_PHASE)
            shared_recv_seq = seq_counters[kPayloadFlagsNumSeqSlots + sm_id] + 1;
    }
    if (fast_send)
        is_next_dirty = __syncthreads_or(is_next_dirty);
    else
        __syncthreads();
    if (thread_id == 0) {
        if (phases & LEGACY_LOW_LATENCY_SEND_PHASE)
            seq_counters[sm_id] = shared_seq;
        if (phases & LEGACY_LOW_LATENCY_RECV_PHASE)
            seq_counters[kPayloadFlagsNumSeqSlots + sm_id] = shared_recv_seq;
    }
    if (sm_id == 0) {
        for (int slot = num_sms + thread_id; slot < kPayloadFlagsNumSeqSlots; slot += blockDim.x) {
            if (phases & LEGACY_LOW_LATENCY_SEND_PHASE)
                seq_counters[slot] += 1;
            if (phases & LEGACY_LOW_LATENCY_RECV_PHASE)
                seq_counters[kPayloadFlagsNumSeqSlots + slot] += 1;
        }
    }

    if ((phases & LEGACY_LOW_LATENCY_SEND_PHASE) == 0)
        goto PAYLOAD_FLAGS_RECV;

    // Clean the next buffer's signals before any peer can see this call's data
    if (fast_send) {
        if (is_next_dirty) {
            for (int i = thread_id; i < num_next_clean_int; i += blockDim.x) {
                next_clean[i] = 0;
                __threadfence();
            }
            __syncthreads();
        }
    } else if (cta_clean) {
        for (int i = thread_id; i < num_next_clean_int; i += blockDim.x) {
            next_clean[i] = 0;
            __threadfence();
        }
        __syncthreads();
    } else if (sm_id == 0 and warp_id == 0) {
        #pragma unroll
        for (int i = lane_id; i < num_next_clean_int; i += 32)
            next_clean[i] = 0;
        __syncwarp();
        if (lane_id == 0)
            atomic_add_release_global(atomic_clean_flag, num_experts);
    }

    // Send each token as flagged units straight into the peer's buffer
    if (responsible_expert_idx < num_experts) {
        const auto dst_rank = responsible_expert_idx / num_local_experts;
        const auto local_expert_idx = responsible_expert_idx % num_local_experts;
        const auto global_expert_idx = rank * num_local_experts + local_expert_idx;
        const auto layout = fast_send ? prefetched_layout : __ldg(layout_range + local_expert_idx * num_ranks + dst_rank);
        const auto local_x =
            static_cast<const int4*>(x) + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * kHiddenInt4;
        const auto dst_lo_base = nvshmemi_get_p2p_ptr(reinterpret_cast<uint64_t>(rdma_recv_x), rank, dst_rank);
        const auto dst_hi_base = nvshmemi_get_p2p_ptr(reinterpret_cast<uint64_t>(rdma_send_x), rank, dst_rank);
        const auto local_src_info = src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
        const int flag = payload_flags_value(shared_seq);
        int offset, num_tokens_to_send;
        unpack2(layout, num_tokens_to_send, offset);

        if (lane_id == 0 and not cta_clean) {
            while (ld_acquire_global(atomic_clean_flag) == 0)
                ;
        }
        __syncwarp();
        if (ext_opts & PAYLOAD_FLAGS_COMBINE_GROUP_CHECK) {
            // Group check: an item's src_info and x loads are issued together, its peer address formed after them
            constexpr int kUnitsPerLane = kChunkUnits / 32;
            EP_STATIC_ASSERT(kChunkUnits % 32 == 0 and kChunkUnits * kNumChunks == kNumUnits, "Chunks must split into lanes");
            EP_DEVICE_ASSERT(dst_lo_base != 0 and dst_hi_base != 0);
            for (int item = sub_warp_id; item < num_tokens_to_send * kNumChunks; item += num_warps_per_group) {
                const int token_idx = offset + item / kNumChunks;
                const int unit_begin = (item % kNumChunks) * kChunkUnits + lane_id;
                const auto x_int2 = reinterpret_cast<const int2*>(local_x + token_idx * kHiddenInt4);
                const auto src_idx = __ldg(local_src_info + token_idx);
                int2 list_value[kUnitsPerLane];
                #pragma unroll
                for (int j = 0; j < kUnitsPerLane; ++j)
                    list_value[j] = __ldg(x_int2 + unit_begin + j * 32);
                const auto slot_offset = (static_cast<uint64_t>(global_expert_idx) * num_max_dispatch_tokens_per_rank + src_idx) * kNumBytesPerSlot;
                #pragma unroll
                for (int j = 0; j < kUnitsPerLane; ++j) {
                    const int unit = unit_begin + j * 32;
                    const auto dst = reinterpret_cast<int4*>(unit < kNumHalfUnits ? dst_lo_base + slot_offset : dst_hi_base + slot_offset) +
                        (unit % kNumHalfUnits);
                    st_volatile_global_v4(dst, make_int4(list_value[j].x, flag, list_value[j].y, flag));
                }
            }
        }

        for (int item = (ext_opts & PAYLOAD_FLAGS_COMBINE_GROUP_CHECK) ? num_tokens_to_send * kNumChunks : sub_warp_id; item < num_tokens_to_send * kNumChunks; item += num_warps_per_group) {
            const int token_idx = offset + item / kNumChunks;
            const int chunk_idx = item % kNumChunks;
            const auto src_idx = __ldg(local_src_info + token_idx);
            const auto slot_offset = (static_cast<uint64_t>(global_expert_idx) * num_max_dispatch_tokens_per_rank + src_idx) * kNumBytesPerSlot;
            const auto dst_lo = fast_send ? dst_lo_base + slot_offset
                                          : nvshmemi_get_p2p_ptr(reinterpret_cast<uint64_t>(rdma_recv_x) + slot_offset, rank, dst_rank);
            const auto dst_hi = fast_send ? dst_hi_base + slot_offset
                                          : nvshmemi_get_p2p_ptr(reinterpret_cast<uint64_t>(rdma_send_x) + slot_offset, rank, dst_rank);
            EP_DEVICE_ASSERT(dst_lo != 0 and dst_hi != 0);
            const auto x_int2 = reinterpret_cast<const int2*>(local_x + token_idx * kHiddenInt4);
            const int chunk_end = min(kNumUnits, (chunk_idx + 1) * kChunkUnits);
            #pragma unroll 4
            for (int unit = chunk_idx * kChunkUnits + lane_id; unit < chunk_end; unit += 32) {
                const auto value = __ldg(x_int2 + unit);
                const auto dst = reinterpret_cast<int4*>(unit < kNumHalfUnits ? dst_lo : dst_hi) + (unit % kNumHalfUnits);
                st_volatile_global_v4(dst, make_int4(value.x, flag, value.y, flag));
            }
        }

        // Return the clean flag once every warp of the group has sent
        asm volatile("bar.sync %0, %1;" ::"r"(warp_group_id + 1), "r"(num_warps_per_group * 32));
        if (sub_warp_id == 0 and lane_id == 0 and not cta_clean)
            atomic_add_release_global(atomic_clean_flag, -1);
    }

PAYLOAD_FLAGS_RECV:
    if ((phases & LEGACY_LOW_LATENCY_RECV_PHASE) == 0)
        return;

    // Tile receive: a CTA per (token, 128 units), a thread per (top-k slot, unit) polling one unit, sum over the slots
    if (tile_recv) {
        constexpr int kTileUnits = 128;
        constexpr int kTileTopk = 8;
        constexpr int kNumTiles = kNumUnits / kTileUnits;
        EP_STATIC_ASSERT(kNumUnits % kTileUnits == 0, "Hidden must split into 128-unit tiles");
        __shared__ int2 shared_tile_data[kTileTopk][kTileUnits];
        __shared__ float shared_tile_weight[kTileTopk];
        const int flag = payload_flags_value(shared_recv_seq);
        const int topk_slot = thread_id / kTileUnits;
        const int unit_in_tile = thread_id % kTileUnits;
        for (int tile = sm_id; tile < num_combined_tokens * kNumTiles; tile += num_sms) {
            const int token_idx = tile / kNumTiles;
            const int unit = (tile % kNumTiles) * kTileUnits + unit_in_tile;
            int2 data = make_int2(0, 0);
            float weight = 0.0f;
            const int expert_idx = payload_flags_topk(topk_idx, token_idx * num_topk + topk_slot, ext_opts & kPayloadFlagsCombineInt32Ids);
            if (expert_idx >= 0) {
                weight = __ldg(topk_weights + token_idx * num_topk + topk_slot);
                const auto recv_base = static_cast<const uint8_t*>(unit < kNumHalfUnits ? rdma_recv_x : rdma_send_x);
                const auto ptr = reinterpret_cast<const int4*>(
                    recv_base + (static_cast<uint64_t>(expert_idx) * num_max_dispatch_tokens_per_rank + token_idx) * kNumBytesPerSlot) +
                    (unit % kNumHalfUnits);
                auto value = ld_volatile_global_v4(ptr);
                auto start_time = clock64();
                while (value.y != flag or value.w != flag) {
                    if (clock64() - start_time > LEGACY_NUM_TIMEOUT_CYCLES) {
                        printf("DeepEP payload-flags tile combine timeout, rank %d, token %d\n", rank, token_idx);
                        trap();
                    }
                    value = ld_volatile_global_v4(ptr);
                }
                data = make_int2(value.x, value.z);
            }
            shared_tile_data[topk_slot][unit_in_tile] = data;
            if (unit_in_tile == 0)
                shared_tile_weight[topk_slot] = weight;
            __syncthreads();
            if (topk_slot == 0) {
                float combined_values[kNumElemsPerUnit] = {0.0f};
                #pragma unroll
                for (int k = 0; k < kTileTopk; ++k) {
                    const int2 slot_data = shared_tile_data[k][unit_in_tile];
                    const float slot_weight = shared_tile_weight[k];
                    const int list_data[2] = {slot_data.x, slot_data.y};
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        const auto pair = *reinterpret_cast<const nv_bfloat162*>(&list_data[j]);
                        combined_values[j * 2] += static_cast<float>(pair.x) * slot_weight;
                        combined_values[j * 2 + 1] += static_cast<float>(pair.y) * slot_weight;
                    }
                }
                int2 out;
                auto out_bf16 = reinterpret_cast<nv_bfloat162*>(&out);
                out_bf16[0] = __floats2bfloat162_rn(combined_values[0], combined_values[1]);
                out_bf16[1] = __floats2bfloat162_rn(combined_values[2], combined_values[3]);
                reinterpret_cast<int2*>(combined_x)[static_cast<int64_t>(token_idx) * kNumUnits + unit] = out;
            }
            __syncthreads();
        }
        return;
    }

    // Each thread reduces one unit (4 BF16) of a token's hidden over its top-k slots, polling each unit's flags
    {
        constexpr int kThreadsPerToken = kNumUnits < 1024 ? kNumUnits : 1024;
        constexpr int kTokensPerCTA = 1024 / kThreadsPerToken;
        constexpr int kTopkBatch = 8;
        const int flag = payload_flags_value(shared_recv_seq);
        const int token_in_cta = thread_id / kThreadsPerToken;
        const int thread_in_token = thread_id % kThreadsPerToken;
        if (token_in_cta < kTokensPerCTA) {
            for (int token_idx = sm_id * kTokensPerCTA + token_in_cta; token_idx < num_combined_tokens; token_idx += num_sms * kTokensPerCTA) {
                for (int unit = thread_in_token; unit < kNumUnits; unit += kThreadsPerToken) {
                    const auto recv_base = static_cast<const uint8_t*>(unit < kNumHalfUnits ? rdma_recv_x : rdma_send_x);
                    float combined_values[kNumElemsPerUnit] = {0.0f};
                    for (int k_base = 0; k_base < num_topk; k_base += kTopkBatch) {
                        int4 list_unit[kTopkBatch];
                        const int4* list_ptr[kTopkBatch];
                        float list_weight[kTopkBatch];
                        #pragma unroll
                        for (int k = 0; k < kTopkBatch; ++k) {
                            list_ptr[k] = nullptr;
                            list_weight[k] = 0.0f;
                            const int topk_slot = k_base + k;
                            if (topk_slot < num_topk) {
                                const int expert_idx = payload_flags_topk(topk_idx, token_idx * num_topk + topk_slot, ext_opts & kPayloadFlagsCombineInt32Ids);
                                if (expert_idx >= 0) {
                                    list_weight[k] = __ldg(topk_weights + token_idx * num_topk + topk_slot);
                                    list_ptr[k] = reinterpret_cast<const int4*>(
                                        recv_base + (static_cast<uint64_t>(expert_idx) * num_max_dispatch_tokens_per_rank + token_idx) * kNumBytesPerSlot) +
                                        (unit % kNumHalfUnits);
                                    list_unit[k] = ld_volatile_global_v4(list_ptr[k]);
                                }
                            }
                        }
                        #pragma unroll
                        for (int k = 0; k < kTopkBatch; ++k) {
                            if (list_ptr[k] == nullptr)
                                continue;
                            auto start_time = clock64();
                            while (list_unit[k].y != flag or list_unit[k].w != flag) {
                                if (clock64() - start_time > LEGACY_NUM_TIMEOUT_CYCLES) {
                                    printf("DeepEP payload-flags combine timeout, rank %d, token %d\n", rank, token_idx);
                                    trap();
                                }
                                list_unit[k] = ld_volatile_global_v4(list_ptr[k]);
                            }
                            const int list_data[2] = {list_unit[k].x, list_unit[k].z};
                            #pragma unroll
                            for (int j = 0; j < 2; ++j) {
                                const auto pair = *reinterpret_cast<const nv_bfloat162*>(&list_data[j]);
                                combined_values[j * 2] += static_cast<float>(pair.x) * list_weight[k];
                                combined_values[j * 2 + 1] += static_cast<float>(pair.y) * list_weight[k];
                            }
                        }
                    }
                    int2 out;
                    auto out_bf16 = reinterpret_cast<nv_bfloat162*>(&out);
                    out_bf16[0] = __floats2bfloat162_rn(combined_values[0], combined_values[1]);
                    out_bf16[1] = __floats2bfloat162_rn(combined_values[2], combined_values[3]);
                    reinterpret_cast<int2*>(combined_x)[static_cast<int64_t>(token_idx) * kNumUnits + unit] = out;
                }
            }
        }
    }
}

void combine_payload_flags(void* combined_x,
                           void* rdma_recv_x,
                           void* rdma_send_x,
                           const void* x,
                           const topk_idx_t* topk_idx,
                           const float* topk_weights,
                           const int* src_info,
                           const int64_t* layout_range,
                           int* next_clean,
                           int num_next_clean_int,
                           int num_combined_tokens,
                           int hidden,
                           int num_max_dispatch_tokens_per_rank,
                           int num_topk,
                           int num_experts,
                           int rank,
                           int num_ranks,
                           void* workspace,
                           int buffer_idx,
                           int num_device_sms,
                           cudaStream_t stream,
                           int phases,
                           bool cta_clean,
                           bool tile_recv,
                           bool fast_send,
                           int ext_opts) {
    const int num_warp_groups = ceil_div(num_experts, num_device_sms);
    const int num_warps_per_group = 32 / num_warp_groups;
    const int num_sms = ceil_div(num_experts, num_warp_groups);
    EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 1 and num_sms <= kPayloadFlagsNumSeqSlots);
    auto atomic_clean_flag = static_cast<int*>(workspace);
    auto seq_counters = reinterpret_cast<int*>(static_cast<uint8_t*>(workspace) + kPayloadFlagsSeqOffsetBytes) +
        buffer_idx * 2 * kPayloadFlagsNumSeqSlots;
    EP_HOST_ASSERT(not tile_recv or (num_topk == 8 and num_warp_groups * num_warps_per_group * 32 == 1024));
    EP_HOST_ASSERT(not fast_send or cta_clean);
    EP_HOST_ASSERT(fast_send or ext_opts == 0);
    EP_HOST_ASSERT(kPayloadFlagsSeqOffsetBytes + 4 * kPayloadFlagsNumSeqSlots * sizeof(int) <= LEGACY_NUM_WORKSPACE_BYTES);

    cudaLaunchConfig_t cfg = {static_cast<unsigned int>(num_sms), static_cast<unsigned int>(num_warp_groups * num_warps_per_group * 32), 0, stream, nullptr, 0};
#define PAYLOAD_FLAGS_COMBINE_LAUNCH_CASE(hidden)                                                     \
    {                                                                                                 \
        LAUNCH_KERNEL(&cfg, combine_payload_flags<hidden>,                                            \
                      combined_x, rdma_recv_x, rdma_send_x, x, topk_idx, topk_weights,                \
                      src_info, layout_range, next_clean, num_next_clean_int,                         \
                      atomic_clean_flag, seq_counters, num_combined_tokens, num_topk,                 \
                      num_max_dispatch_tokens_per_rank, num_experts, rank, num_ranks,                 \
                      num_warp_groups, num_warps_per_group, phases, cta_clean, tile_recv, fast_send,  \
                      ext_opts);                                                                      \
    }                                                                                                 \
    break
    SWITCH_HIDDEN(PAYLOAD_FLAGS_COMBINE_LAUNCH_CASE);
#undef PAYLOAD_FLAGS_COMBINE_LAUNCH_CASE
}

// Fence-free dispatch over NVLink P2P: the FP8 cast writes each 8-byte chunk of a token with the call's flag twice
// straight into every top-k peer slot, and each (expert, source rank) count travels as a flagged unit too, so no sender
// issues a system-scope release and the receiver accepts every unit on its own
constexpr size_t kPayloadFlagsDispatchSeqOffsetBytes = kPayloadFlagsSeqOffsetBytes + 4 * kPayloadFlagsNumSeqSlots * sizeof(int);
constexpr int kPayloadFlagsDispatchNumMaxTopk = 16;
constexpr int kPayloadFlagsDispatchRecvBatch = 6;
constexpr int kPayloadFlagsDispatchUnitRecvBatch = 4;

template <int kHidden>
struct PayloadFlagsDispatchLayout {
    static constexpr int kNumDataUnits = kHidden / (sizeof(int4) / sizeof(nv_bfloat16));
    static constexpr int kNumScaleUnits = kNumDataUnits / 32;
    static constexpr int kSrcUnit = kNumDataUnits + kNumScaleUnits;
    static constexpr int kNumUnits = kSrcUnit + 1;
    static constexpr int kNumLoUnits = (kNumUnits + 1) / 2;
    static constexpr size_t kNumSlotBytes = kNumLoUnits * sizeof(int4);
    static constexpr int kNumCopies = 1024 / kNumDataUnits;
};

template <bool kUseUE8M0, int kHidden>
__global__ __launch_bounds__(1024, 1) void
dispatch_payload_flags(void* packed_recv_x, void* packed_recv_x_scales, int* packed_recv_src_info,
                       int64_t* packed_recv_layout_range, int* packed_recv_count,
                       int* cumulative_local_expert_recv_stats,
                       void* rdma_recv_x, void* rdma_send_x,
                       const void* x, const topk_idx_t* topk_idx,
                       int* next_clean, int num_next_clean_int, int* seq_counters,
                       int num_tokens, int num_max_dispatch_tokens_per_rank, int num_topk,
                       int num_experts, int rank, int num_ranks,
                       int num_warp_groups, int num_warps_per_group,
                       bool round_scale, int phases, bool unit_recv, bool fast_send, int ext_opts) {
    using Layout = PayloadFlagsDispatchLayout<kHidden>;
    using scale_t = std::conditional_t<kUseUE8M0, uint8_t, float>;
    using packed_t = std::conditional_t<kUseUE8M0, uint32_t, float>;
    constexpr int kNumElemsPerRead = sizeof(int4) / sizeof(nv_bfloat16);
    constexpr int kNumPerChannels = 128;
    constexpr int kNumMaxWarpGroups = 32;
    EP_STATIC_ASSERT(Layout::kNumDataUnits % 32 == 0 and Layout::kNumDataUnits <= 1024, "Invalid hidden");
    EP_STATIC_ASSERT(kNumElemsPerRead * 32 / kNumPerChannels == 2, "Invalid vectorization");

    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto warp_id = thread_id / 32, lane_id = get_lane_id();
    const auto num_warps = num_warp_groups * num_warps_per_group;
    const auto num_local_experts = num_experts / num_ranks;
    const auto warp_group_id = warp_id / num_warps_per_group;
    const auto sub_warp_id = warp_id % num_warps_per_group;
    const auto responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;
    const size_t num_hi_region_bytes = static_cast<size_t>(num_experts) * num_max_dispatch_tokens_per_rank * Layout::kNumSlotBytes;

    __shared__ int shared_seq;
    __shared__ int shared_recv_seq;
    __shared__ uint64_t shared_dst_lo[kPayloadFlagsDispatchNumMaxTopk];
    __shared__ uint64_t shared_dst_hi[kPayloadFlagsDispatchNumMaxTopk];
    __shared__ int shared_num_recv_tokens[kNumMaxWarpGroups];
    __shared__ int shared_recv_token_begin_idx[kNumMaxWarpGroups];

    // Fast send: the counting warp reads its share of the next buffer's signals in the sequence's round trip, and
    // clears it (with its fence) only when a shipped kernel left one set
    const int num_clean_per_sm = ceil_div(num_next_clean_int, num_sms);
    const int clean_end = min(num_next_clean_int, (sm_id + 1) * num_clean_per_sm);
    bool is_next_dirty = false;
    if (fast_send and (phases & LEGACY_LOW_LATENCY_SEND_PHASE) and warp_id == num_warps - 1) {
        int is_lane_dirty = 0;
        for (int i = sm_id * num_clean_per_sm + lane_id; i < clean_end; i += 32)
            is_lane_dirty |= ld_volatile_global(next_clean + i) != 0;
        is_next_dirty = __any_sync(0xffffffff, is_lane_dirty);
    }

    if (thread_id == 0) {
        if (phases & LEGACY_LOW_LATENCY_SEND_PHASE)
            shared_seq = seq_counters[sm_id] + 1;
        if (phases & LEGACY_LOW_LATENCY_RECV_PHASE)
            shared_recv_seq = seq_counters[kPayloadFlagsNumSeqSlots + sm_id] + 1;
    }
    __syncthreads();
    if (thread_id == 0) {
        if (phases & LEGACY_LOW_LATENCY_SEND_PHASE)
            seq_counters[sm_id] = shared_seq;
        if (phases & LEGACY_LOW_LATENCY_RECV_PHASE)
            seq_counters[kPayloadFlagsNumSeqSlots + sm_id] = shared_recv_seq;
    }
    if (sm_id == 0) {
        for (int slot = num_sms + thread_id; slot < kPayloadFlagsNumSeqSlots; slot += blockDim.x) {
            if (phases & LEGACY_LOW_LATENCY_SEND_PHASE)
                seq_counters[slot] += 1;
            if (phases & LEGACY_LOW_LATENCY_RECV_PHASE)
                seq_counters[kPayloadFlagsNumSeqSlots + slot] += 1;
        }
    }

    if ((phases & LEGACY_LOW_LATENCY_SEND_PHASE) == 0)
        goto PAYLOAD_FLAGS_DISPATCH_RECV;

    {
        const int flag = payload_flags_value(shared_seq);

        // Clean this SM's share of the next buffer's signals, then count the tokens of its experts and send each count
        // as a flagged unit
        if (warp_id == num_warps - 1) {
            const auto expert_begin_idx = sm_id * num_warp_groups;
            const auto expert_end_idx = min(expert_begin_idx + num_warp_groups, num_experts);
            if (not fast_send or is_next_dirty) {
                for (int i = sm_id * num_clean_per_sm + lane_id; i < clean_end; i += 32)
                    next_clean[i] = 0;
                __syncwarp();
                __threadfence();
            }
            for (int expert_idx = expert_begin_idx; expert_idx < expert_end_idx; ++expert_idx) {
                int count = 0;
                for (int i = lane_id; i < num_tokens * num_topk; i += 32)
                    count += payload_flags_topk(topk_idx, i, ext_opts & kPayloadFlagsDispatchInt32Ids) == expert_idx;
                count = warp_reduce_sum(count);
                if (lane_id == 0) {
                    const auto dst_rank = expert_idx / num_local_experts;
                    const auto dst_expert_local_idx = expert_idx % num_local_experts;
                    const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_send_x) + num_hi_region_bytes +
                        (dst_expert_local_idx * num_ranks + rank) * sizeof(int4);
                    const auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
                    EP_DEVICE_ASSERT(dst_p2p_ptr != 0);
                    st_volatile_global_v4(reinterpret_cast<int4*>(dst_p2p_ptr), make_int4(count, flag, count, flag));
                }
            }
        }

        // Cast each token and write its flagged units into every top-k peer slot
        const int copy_idx = thread_id / Layout::kNumDataUnits;
        const int unit_idx = thread_id % Layout::kNumDataUnits;
        for (int token_idx = sm_id; token_idx < num_tokens; token_idx += num_sms) {
            if (warp_id < num_topk) {
                const int dst_expert_idx = payload_flags_topk(topk_idx, token_idx * num_topk + warp_id, ext_opts & kPayloadFlagsDispatchInt32Ids);
                int slot_idx = 0;
                if (dst_expert_idx >= 0) {
                    for (int i = lane_id; i < token_idx * num_topk; i += 32)
                        slot_idx += payload_flags_topk(topk_idx, i, ext_opts & kPayloadFlagsDispatchInt32Ids) == dst_expert_idx;
                    slot_idx = warp_reduce_sum(slot_idx);
                }
                if (lane_id == 0) {
                    uint64_t dst_lo = 0, dst_hi = 0;
                    if (dst_expert_idx >= 0) {
                        const auto dst_rank = dst_expert_idx / num_local_experts;
                        const auto dst_expert_local_idx = dst_expert_idx % num_local_experts;
                        const auto slot_offset =
                            ((static_cast<size_t>(dst_expert_local_idx) * num_ranks + rank) * num_max_dispatch_tokens_per_rank + slot_idx) *
                            Layout::kNumSlotBytes;
                        dst_lo = nvshmemi_get_p2p_ptr(reinterpret_cast<uint64_t>(rdma_recv_x) + slot_offset, rank, dst_rank);
                        dst_hi = nvshmemi_get_p2p_ptr(reinterpret_cast<uint64_t>(rdma_send_x) + slot_offset, rank, dst_rank);
                        EP_DEVICE_ASSERT(dst_lo != 0 and dst_hi != 0);
                    }
                    shared_dst_lo[warp_id] = dst_lo;
                    shared_dst_hi[warp_id] = dst_hi;
                }
            }

            int2 fp8_value = {0, 0};
            float scale_inv_lo = 0.0f, scale_inv_hi = 0.0f;
            if (copy_idx < Layout::kNumCopies) {
                auto int4_value = __ldg(static_cast<const int4*>(x) + static_cast<size_t>(token_idx) * Layout::kNumDataUnits + unit_idx);
                auto bf16_values = reinterpret_cast<nv_bfloat16*>(&int4_value);
                float fp32_values[kNumElemsPerRead];
                float amax = kFP8Margin, scale, scale_inv;
                #pragma unroll
                for (int j = 0; j < kNumElemsPerRead; ++j) {
                    fp32_values[j] = static_cast<float>(bf16_values[j]);
                    amax = fmaxf(amax, fabsf(fp32_values[j]));
                }
                amax = warp_reduce_max<16>(amax);
                calculate_fp8_scales(amax, scale, scale_inv, round_scale);
                auto fp8x2_values = reinterpret_cast<__nv_fp8x2_storage_t*>(&fp8_value);
                #pragma unroll
                for (int j = 0; j < kNumElemsPerRead; j += 2) {
                    float2 fp32x2 = {fp32_values[j] * scale, fp32_values[j + 1] * scale};
                    fp8x2_values[j / 2] = __nv_cvt_float2_to_fp8x2(fp32x2, __NV_SATFINITE, __NV_E4M3);
                }
                scale_inv_lo = scale_inv;
                scale_inv_hi = __shfl_sync(0xffffffff, scale_inv, 16);
            }
            __syncthreads();

            if (copy_idx < Layout::kNumCopies) {
                for (int k = copy_idx; k < num_topk; k += Layout::kNumCopies) {
                    const auto dst_lo = reinterpret_cast<int4*>(shared_dst_lo[k]);
                    const auto dst_hi = reinterpret_cast<int4*>(shared_dst_hi[k]);
                    if (dst_lo == nullptr)
                        continue;
                    st_volatile_global_v4(unit_idx < Layout::kNumLoUnits ? dst_lo + unit_idx : dst_hi + (unit_idx - Layout::kNumLoUnits),
                                          make_int4(fp8_value.x, flag, fp8_value.y, flag));
                    if (lane_id == 0) {
                        const int scale_unit_idx = Layout::kNumDataUnits + unit_idx / 32;
                        st_volatile_global_v4(scale_unit_idx < Layout::kNumLoUnits ? dst_lo + scale_unit_idx
                                                                                  : dst_hi + (scale_unit_idx - Layout::kNumLoUnits),
                                              make_int4(__float_as_int(scale_inv_lo), flag, __float_as_int(scale_inv_hi), flag));
                    }
                    if (unit_idx == 0)
                        st_volatile_global_v4(dst_hi + (Layout::kSrcUnit - Layout::kNumLoUnits), make_int4(token_idx, flag, 0, flag));
                }
            }
            __syncthreads();
        }
    }

PAYLOAD_FLAGS_DISPATCH_RECV:
    if ((phases & LEGACY_LOW_LATENCY_RECV_PHASE) == 0)
        return;

    // Each warp group packs the tokens one source rank sent to one local expert, at a begin fixed by the counts
    if (responsible_expert_idx < num_experts) {
        const int flag = payload_flags_value(shared_recv_seq);
        const auto src_rank = responsible_expert_idx / num_local_experts;
        const auto local_expert_idx = responsible_expert_idx % num_local_experts;
        const auto num_slots_per_expert = num_ranks * num_max_dispatch_tokens_per_rank;
        if (sub_warp_id == 0) {
            int num_rank_tokens = 0;
            if (lane_id < num_ranks) {
                const auto count_ptr = reinterpret_cast<const int4*>(static_cast<const uint8_t*>(rdma_send_x) + num_hi_region_bytes) +
                    local_expert_idx * num_ranks + lane_id;
                auto start_time = clock64();
                auto value = ld_volatile_global_v4(count_ptr);
                while (value.y != flag or value.w != flag) {
                    if (clock64() - start_time > LEGACY_NUM_TIMEOUT_CYCLES) {
                        printf("DeepEP payload-flags dispatch count timeout, rank %d, local_expert_idx %d, src_rank %d\n",
                               rank, local_expert_idx, lane_id);
                        trap();
                    }
                    value = ld_volatile_global_v4(count_ptr);
                }
                num_rank_tokens = value.x;
            }
            int recv_token_begin_idx = 0, num_total_tokens = 0;
            for (int i = 0; i < num_ranks; ++i) {
                const int num_tokens_of_rank = __shfl_sync(0xffffffff, num_rank_tokens, i);
                recv_token_begin_idx += i < src_rank ? num_tokens_of_rank : 0;
                num_total_tokens += num_tokens_of_rank;
            }
            const int num_recv_tokens = __shfl_sync(0xffffffff, num_rank_tokens, src_rank);
            if (lane_id == 0) {
                shared_num_recv_tokens[warp_group_id] = num_recv_tokens;
                shared_recv_token_begin_idx[warp_group_id] = recv_token_begin_idx;
                packed_recv_layout_range[local_expert_idx * num_ranks + src_rank] = pack2<int, int64_t>(num_recv_tokens, recv_token_begin_idx);
                if (src_rank == 0)
                    packed_recv_count[local_expert_idx] = num_total_tokens;
                if (cumulative_local_expert_recv_stats != nullptr)
                    atomicAdd(cumulative_local_expert_recv_stats + local_expert_idx, num_recv_tokens);
            }
        }
        asm volatile("bar.sync %0, %1;" ::"r"(warp_group_id + 1), "r"(num_warps_per_group * 32));
        const int num_recv_tokens = shared_num_recv_tokens[warp_group_id];
        const int recv_token_begin_idx = shared_recv_token_begin_idx[warp_group_id];

        constexpr int kNumElemsPerPack = sizeof(packed_t) / sizeof(scale_t);
        constexpr int kNumScales = kHidden / kNumPerChannels;
        const auto num_aligned_tokens = align_up<int>(num_slots_per_expert, 128);
        const auto num_aligned_scales = align_up<int>(kNumScales, sizeof(float) / sizeof(scale_t));
        const auto recv_x_scales = static_cast<scale_t*>(packed_recv_x_scales) + local_expert_idx * num_aligned_tokens * num_aligned_scales;
        const auto pack_stride = num_slots_per_expert * kNumElemsPerPack;
        const auto recv_x_int2 = static_cast<int2*>(packed_recv_x) + static_cast<size_t>(local_expert_idx) * num_slots_per_expert * Layout::kNumDataUnits;
        const auto recv_src_info = packed_recv_src_info + local_expert_idx * num_slots_per_expert;

        // Unit receive: the warp group's threads take the (token, unit) items in turn, each polling a batch of units
        // at once and reloading only the pending ones, so a unit's wait never queues behind another's
        if (unit_recv) {
            const int thread_in_group = sub_warp_id * 32 + lane_id;
            const int num_group_threads = num_warps_per_group * 32;
            const int num_items = num_recv_tokens * Layout::kNumUnits;
            const auto store_unit = [&](const int item, const int4& value) {
                const int token_idx = recv_token_begin_idx + item / Layout::kNumUnits;
                const int unit = item % Layout::kNumUnits;
                if (unit < Layout::kNumDataUnits) {
                    recv_x_int2[static_cast<size_t>(token_idx) * Layout::kNumDataUnits + unit] = make_int2(value.x, value.z);
                } else if (unit < Layout::kSrcUnit) {
                    const int list_scale_bits[2] = {value.x, value.z};
                    #pragma unroll
                    for (int half = 0; half < 2; ++half) {
                        const int scale_idx = (unit - Layout::kNumDataUnits) * 2 + half;
                        const int pack_idx = scale_idx / kNumElemsPerPack;
                        const int elem_idx = scale_idx % kNumElemsPerPack;
                        recv_x_scales[token_idx * kNumElemsPerPack + pack_idx * pack_stride + elem_idx] =
                            extract_required_scale_format<kUseUE8M0>(__int_as_float(list_scale_bits[half]));
                    }
                } else {
                    recv_src_info[token_idx] = value.x;
                }
            };
            for (int item_base = 0; item_base < num_items; item_base += num_group_threads * kPayloadFlagsDispatchUnitRecvBatch) {
                int4 list_value[kPayloadFlagsDispatchUnitRecvBatch];
                const int4* list_src[kPayloadFlagsDispatchUnitRecvBatch];
                bool list_pending[kPayloadFlagsDispatchUnitRecvBatch];
                #pragma unroll
                for (int j = 0; j < kPayloadFlagsDispatchUnitRecvBatch; ++j) {
                    const int item = item_base + j * num_group_threads + thread_in_group;
                    list_pending[j] = item < num_items;
                    list_src[j] = nullptr;
                    if (list_pending[j]) {
                        const int unit = item % Layout::kNumUnits;
                        const auto slot_offset =
                            ((static_cast<size_t>(local_expert_idx) * num_ranks + src_rank) * num_max_dispatch_tokens_per_rank +
                             item / Layout::kNumUnits) * Layout::kNumSlotBytes;
                        list_src[j] = unit < Layout::kNumLoUnits
                            ? reinterpret_cast<const int4*>(static_cast<const uint8_t*>(rdma_recv_x) + slot_offset) + unit
                            : reinterpret_cast<const int4*>(static_cast<const uint8_t*>(rdma_send_x) + slot_offset) + (unit - Layout::kNumLoUnits);
                        list_value[j] = ld_volatile_global_v4(list_src[j]);
                    }
                }
                auto start_time = clock64();
                while (true) {
                    bool is_any_pending = false;
                    #pragma unroll
                    for (int j = 0; j < kPayloadFlagsDispatchUnitRecvBatch; ++j) {
                        if (not list_pending[j])
                            continue;
                        if (list_value[j].y == flag and list_value[j].w == flag) {
                            store_unit(item_base + j * num_group_threads + thread_in_group, list_value[j]);
                            list_pending[j] = false;
                        } else {
                            is_any_pending = true;
                        }
                    }
                    if (not is_any_pending)
                        break;
                    if (clock64() - start_time > LEGACY_NUM_TIMEOUT_CYCLES) {
                        printf("DeepEP payload-flags unit dispatch timeout, rank %d, local_expert_idx %d, src_rank %d\n",
                               rank, local_expert_idx, src_rank);
                        trap();
                    }
                    #pragma unroll
                    for (int j = 0; j < kPayloadFlagsDispatchUnitRecvBatch; ++j) {
                        if (list_pending[j])
                            list_value[j] = ld_volatile_global_v4(list_src[j]);
                    }
                }
            }
            return;
        }

        for (int i = sub_warp_id; i < num_recv_tokens; i += num_warps_per_group) {
            const auto slot_offset =
                ((static_cast<size_t>(local_expert_idx) * num_ranks + src_rank) * num_max_dispatch_tokens_per_rank + i) * Layout::kNumSlotBytes;
            const auto src_lo = reinterpret_cast<const int4*>(static_cast<const uint8_t*>(rdma_recv_x) + slot_offset);
            const auto src_hi = reinterpret_cast<const int4*>(static_cast<const uint8_t*>(rdma_send_x) + slot_offset);
            const int token_idx = recv_token_begin_idx + i;
            for (int unit_base = 0; unit_base < Layout::kNumUnits; unit_base += 32 * kPayloadFlagsDispatchRecvBatch) {
                int4 list_value[kPayloadFlagsDispatchRecvBatch];
                #pragma unroll
                for (int j = 0; j < kPayloadFlagsDispatchRecvBatch; ++j) {
                    const int unit = unit_base + j * 32 + lane_id;
                    if (unit < Layout::kNumUnits)
                        list_value[j] = ld_volatile_global_v4(unit < Layout::kNumLoUnits ? src_lo + unit : src_hi + (unit - Layout::kNumLoUnits));
                }
                #pragma unroll
                for (int j = 0; j < kPayloadFlagsDispatchRecvBatch; ++j) {
                    const int unit = unit_base + j * 32 + lane_id;
                    if (unit >= Layout::kNumUnits)
                        continue;
                    auto start_time = clock64();
                    while (list_value[j].y != flag or list_value[j].w != flag) {
                        if (clock64() - start_time > LEGACY_NUM_TIMEOUT_CYCLES) {
                            printf("DeepEP payload-flags dispatch timeout, rank %d, local_expert_idx %d, src_rank %d\n",
                                   rank, local_expert_idx, src_rank);
                            trap();
                        }
                        list_value[j] = ld_volatile_global_v4(unit < Layout::kNumLoUnits ? src_lo + unit : src_hi + (unit - Layout::kNumLoUnits));
                    }
                    if (unit < Layout::kNumDataUnits) {
                        recv_x_int2[static_cast<size_t>(token_idx) * Layout::kNumDataUnits + unit] = make_int2(list_value[j].x, list_value[j].z);
                    } else if (unit < Layout::kSrcUnit) {
                        const int list_scale_bits[2] = {list_value[j].x, list_value[j].z};
                        #pragma unroll
                        for (int half = 0; half < 2; ++half) {
                            const int scale_idx = (unit - Layout::kNumDataUnits) * 2 + half;
                            const int pack_idx = scale_idx / kNumElemsPerPack;
                            const int elem_idx = scale_idx % kNumElemsPerPack;
                            recv_x_scales[token_idx * kNumElemsPerPack + pack_idx * pack_stride + elem_idx] =
                                extract_required_scale_format<kUseUE8M0>(__int_as_float(list_scale_bits[half]));
                        }
                    } else {
                        recv_src_info[token_idx] = list_value[j].x;
                    }
                }
            }
        }
    }
}

void dispatch_payload_flags(void* packed_recv_x,
                            void* packed_recv_x_scales,
                            int* packed_recv_src_info,
                            int64_t* packed_recv_layout_range,
                            int* packed_recv_count,
                            int* cumulative_local_expert_recv_stats,
                            void* rdma_recv_x,
                            void* rdma_send_x,
                            size_t num_region_bytes,
                            const void* x,
                            const topk_idx_t* topk_idx,
                            int* next_clean,
                            int num_next_clean_int,
                            int num_tokens,
                            int hidden,
                            int num_max_dispatch_tokens_per_rank,
                            int num_topk,
                            int num_experts,
                            int rank,
                            int num_ranks,
                            bool round_scale,
                            bool use_ue8m0,
                            void* workspace,
                            int buffer_idx,
                            int num_device_sms,
                            cudaStream_t stream,
                            int phases,
                            bool unit_recv,
                            bool fast_send,
                            int ext_opts) {
    const int num_warp_groups = ceil_div(num_experts, num_device_sms);
    const int num_warps_per_group = 32 / num_warp_groups;
    const int num_sms = ceil_div(num_experts, num_warp_groups);
    EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 1 and num_sms <= kPayloadFlagsNumSeqSlots);
    EP_HOST_ASSERT(num_topk <= kPayloadFlagsDispatchNumMaxTopk and num_topk <= num_warp_groups * num_warps_per_group);
    EP_HOST_ASSERT(hidden % 256 == 0 and hidden <= 8192);
    const size_t num_slot_bytes = ((hidden / 8 + hidden / 256 + 2) / 2) * sizeof(int4);
    EP_HOST_ASSERT(static_cast<size_t>(num_experts) * num_max_dispatch_tokens_per_rank * num_slot_bytes + num_experts * sizeof(int4) <=
                   num_region_bytes);
    auto seq_counters = reinterpret_cast<int*>(static_cast<uint8_t*>(workspace) + kPayloadFlagsDispatchSeqOffsetBytes) +
        buffer_idx * 2 * kPayloadFlagsNumSeqSlots;
    EP_HOST_ASSERT(kPayloadFlagsDispatchSeqOffsetBytes + 4 * kPayloadFlagsNumSeqSlots * sizeof(int) <= LEGACY_NUM_WORKSPACE_BYTES);

#define PAYLOAD_FLAGS_DISPATCH_LAUNCH_CASE(hidden)                                                                 \
    {                                                                                                              \
        EP_HOST_ASSERT(PayloadFlagsDispatchLayout<hidden>::kNumSlotBytes == num_slot_bytes);                       \
        auto dispatch_func = use_ue8m0 ? dispatch_payload_flags<true, hidden> : dispatch_payload_flags<false, hidden>; \
        LAUNCH_KERNEL(&cfg, dispatch_func,                                                                         \
                      packed_recv_x, packed_recv_x_scales, packed_recv_src_info, packed_recv_layout_range,         \
                      packed_recv_count, cumulative_local_expert_recv_stats, rdma_recv_x, rdma_send_x,             \
                      x, topk_idx, next_clean, num_next_clean_int, seq_counters,                                   \
                      num_tokens, num_max_dispatch_tokens_per_rank, num_topk, num_experts, rank, num_ranks,        \
                      num_warp_groups, num_warps_per_group, round_scale, phases, unit_recv, fast_send, ext_opts);  \
    }                                                                                                              \
    break

    SETUP_LAUNCH_CONFIG(num_sms, num_warp_groups * num_warps_per_group * 32, stream);
    SWITCH_HIDDEN(PAYLOAD_FLAGS_DISPATCH_LAUNCH_CASE);
#undef PAYLOAD_FLAGS_DISPATCH_LAUNCH_CASE
}

}  // namespace internode_ll

}  // namespace deep_ep::legacy
