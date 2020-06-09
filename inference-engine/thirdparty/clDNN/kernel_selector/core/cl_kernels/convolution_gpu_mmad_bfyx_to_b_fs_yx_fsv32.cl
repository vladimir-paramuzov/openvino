// Copyright (c) 2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "include/common.cl"

#include "include/data_types.cl"

#include "include/fetch.cl"
#include "include/imad.cl"
#include "include/mmad.cl"

#define CEIL_DIV(x, y) (1 + ((x) - 1) / (y))
#define AS_TYPE(type, val) CAT(as_, type)(val)

#define ISV 4

#ifdef ACCUMULATOR_TYPE
#undef ACCUMULATOR_TYPE
#endif

#ifdef TO_ACCUMULATOR_TYPE
#undef TO_ACCUMULATOR_TYPE
#endif

#if QUANTIZATION_TERM
#define ACCUMULATOR_TYPE int
#define TO_ACCUMULATOR_TYPE(x) convert_int(x)
#define ACTIVATION_TYPE float
#define TO_ACTIVATION_TYPE(x) convert_float(x)

#if OUTPUT_X_BLOCK_SIZE == 8
    #define PACKED_TYPE_VEC MAKE_VECTOR_TYPE(PACKED_IN_TYPE, 8)
    #define ACCUMULATOR_TYPE_VEC int8
    #define TO_ACCUMULATOR_TYPE_VEC(x) convert_int8(x)
    #define ACTIVATION_TYPE_VEC float8
    #define TO_ACTIVATION_TYPE_VEC(x) convert_float8(x)
#if OUTPUT_LAYOUT_B_FS_YX_FSV32
    #define BLOCK_WRITE(ptr, val) intel_sub_group_block_write_us8((__global ushort*)(ptr), as_ushort8(val));
#else // OUTPUT_LAYOUT_B_FS_YX_FSV32
    #define BLOCK_WRITE(ptr, val) BLOCK_WRITE_UC_8((__global uchar*)(ptr), as_uchar8(val))
#endif // OUTPUT_LAYOUT_B_FS_YX_FSV32
#elif OUTPUT_X_BLOCK_SIZE == 4
    #define PACKED_TYPE_VEC MAKE_VECTOR_TYPE(PACKED_IN_TYPE, 4)
    #define ACCUMULATOR_TYPE_VEC int4
    #define TO_ACCUMULATOR_TYPE_VEC(x) convert_int4(x)
    #define ACTIVATION_TYPE_VEC float4
    #define TO_ACTIVATION_TYPE_VEC(x) convert_float4(x)
#if OUTPUT_LAYOUT_B_FS_YX_FSV32
    #define BLOCK_WRITE(ptr, val) intel_sub_group_block_write_us4((__global ushort*)(ptr), as_ushort4(val));
#else // OUTPUT_LAYOUT_B_FS_YX_FSV32
    #define BLOCK_WRITE(ptr, val) BLOCK_WRITE_UC_4((__global uchar*)(ptr), as_uchar4(val))
#endif // OUTPUT_LAYOUT_B_FS_YX_FSV32
#else
#error "convolution_gpu_mmad_bfyx_b_fs_yx_fsv32: Unsupported block size"
#endif

#else // QUANTIZATION_TERM
#error "convolution_gpu_mmad_bfyx_b_fs_yx_fsv32: invalid parameters: quantization term is expected to be true"
#endif

#define AS_TYPE_N_(type, n, x) as_##type##n(x)
#define AS_TYPE_N(type, n, x) AS_TYPE_N_(type, n, x)
#define AS_INPUT0_TYPE_4(x) AS_TYPE_N(INPUT0_TYPE, 4, x)

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(LWS0, LWS1, LWS2)))
KERNEL(convolution_mmad_bfyx_to_b_fs_yx_fsv32)(
    __global INPUT0_TYPE* input,
#if OUTPUT_LAYOUT_B_FS_YX_FSV32
    __global PACKED_OUT_TYPE* output,
#else // OUTPUT_LAYOUT_B_FS_YX_FSV32
    __global OUTPUT_TYPE* output,
#endif //OUTPUT_LAYOUT_B_FS_YX_FSV32
    __global FILTER_TYPE* weights,
#if BIAS_TERM
    __global BIAS_TYPE* biases,
#endif
#if ASYMMETRIC_WEIGHTS_QUANTIZATION
    const __global WEIGHTS_ZERO_POINTS_TYPE *weights_zp,
#endif
#if ASYMMETRIC_DATA_QUANTIZATION
    const __global ACTIVATIONS_ZERO_POINTS_TYPE *activations_zp,
    const __global COMPENSATION_TYPE *compensation,
#endif
#if HAS_FUSED_OPS_DECLS
    FUSED_OPS_DECLS,
#endif
    uint split_idx)
{
    const int fg = get_group_id(0);
    const int x = (int)get_global_id(1) * OUTPUT_X_BLOCK_SIZE;
    const int b = (int)get_global_id(2) / OUTPUT_SIZE_Y;
    const int y = (int)get_global_id(2) % OUTPUT_SIZE_Y;

    const int lid = get_sub_group_local_id();
    const int group_id = get_group_id(1);
    const int sg = get_sub_group_id();

    const int x_wg_start = (group_id * GROUP_SIZE) * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

    ACCUMULATOR_TYPE_VEC acc[2] = { 0 }; // 2*16 packed channels * OUTPUT_X_BLOCK_SIZE
#if ASYMMETRIC_WEIGHTS_QUANTIZATION
    ACCUMULATOR_TYPE_VEC acc_assym_weights = 0;
#endif

#if INPUT0_LAYOUT_BFYX
    const int input_offset = b*INPUT0_BATCH_PITCH + INPUT0_OFFSET + input_y * INPUT0_Y_PITCH;
#elif INPUT0_LAYOUT_B_FS_YX_FSV4
    const int fsv = 4;
    const int input_x_pitch = fsv;
    const int input_y_pitch = input_x_pitch * (INPUT0_PAD_BEFORE_SIZE_X + INPUT0_SIZE_X + INPUT0_PAD_AFTER_SIZE_X);
    const int input_fs_pitch = input_y_pitch * (INPUT0_PAD_BEFORE_SIZE_Y + INPUT0_SIZE_Y + INPUT0_PAD_AFTER_SIZE_Y);
    const int input_total_f_size = INPUT0_PAD_BEFORE_FEATURE_NUM + INPUT0_FEATURE_NUM + INPUT0_PAD_AFTER_FEATURE_NUM;
    const int input_b_pitch = input_fs_pitch * ((input_total_f_size + fsv - 1) / fsv);
    const int input_offset = b*input_b_pitch + input_y * input_y_pitch;
#endif

    int filter_idx = fg * FILTER_SIZE_X * FILTER_SIZE_Y * ISV * OSV;
#if ASYMMETRIC_WEIGHTS_QUANTIZATION
    char4 multiplier;
    for (int i = 0; i < INPUT0_FEATURE_NUM; i++)
        multiplier[i] = 1;
#endif // ASYMMETRIC_WEIGHTS_QUANTIZATION

#if ASYMMETRIC_DATA_QUANTIZATION
    char4 zp = as_char4(((const __global uint*)(activations_zp))[0]);
#if INPUT0_FEATURE_NUM == 3
    zp[3] = 0;
#endif // INPUT0_FEATURE_NUM == 3
#endif // ASYMMETRIC_DATA_QUANTIZATION

    __local PACKED_IN_TYPE slm[SLM_LINE_SIZE*FILTER_SIZE_Y];

    for (int kh = 0; kh < FILTER_SIZE_Y ; ++kh) {
        __local PACKED_IN_TYPE* slm_block = slm + kh*SLM_LINE_SIZE + sg*SLM_CHUNK_SIZE;
        bool y_cross_fm = input_y + kh*DILATION_SIZE_Y < 0 || input_y + kh*DILATION_SIZE_Y >= INPUT0_SIZE_Y;
        if (y_cross_fm) {
#if ASYMMETRIC_DATA_QUANTIZATION
            for (int c = 0; c < SLM_CHUNK_SIZE; c += SUB_GROUP_SIZE) {
                if (sg*SLM_CHUNK_SIZE + c + lid < SLM_LINE_SIZE)
                    slm_block[c + lid] = AS_PACKED_IN_TYPE(zp);
            }
#if SLM_TAIL > 0
            if (sg == LWS1 - 1) {
                __local PACKED_IN_TYPE* slm_block_tail = slm + kh*SLM_LINE_SIZE + LWS1*SLM_CHUNK_SIZE;
                slm_block_tail[lid] = AS_PACKED_IN_TYPE(zp);
            }
#endif // SLM_TAIL > 0
#endif // ASYMMETRIC_DATA_QUANTIZATION
            continue;
        }

        {
            for (int c = 0; c < SLM_CHUNK_SIZE; c += SUB_GROUP_SIZE) {
                const int x_chunk = x_wg_start + sg*SLM_CHUNK_SIZE + c;
                bool x_cross_fm = x_chunk + lid < 0 || x_chunk + lid >= INPUT0_SIZE_X;

                if (!x_cross_fm) {
                #if INPUT0_LAYOUT_BFYX
                    MAKE_VECTOR_TYPE(INPUT0_TYPE, ISV) src = 0;
                    __attribute__((opencl_unroll_hint(INPUT0_FEATURE_NUM)))
                    for (int i = 0; i < INPUT0_FEATURE_NUM; i++) {
                        src[i] = input[input_offset + i * INPUT0_FEATURE_PITCH
                                                    + kh * DILATION_SIZE_Y * INPUT0_Y_PITCH
                                                    + (x_chunk + lid)* INPUT0_X_PITCH];
                    }
                    slm_block[c + lid] = AS_PACKED_IN_TYPE(src);
                #elif INPUT0_LAYOUT_B_FS_YX_FSV4
                    const __global uint* ptr = input + input_offset + kh * DILATION_SIZE_Y * input_y_pitch + (x_chunk + lid) * input_x_pitch;
                    PACKED_IN_TYPE src = AS_PACKED_IN_TYPE(ptr[0]);
                    slm_block[c + lid] = src;
                #endif
                } else {
#if ASYMMETRIC_DATA_QUANTIZATION
                    slm_block[c + lid] = AS_PACKED_IN_TYPE(zp);
#else  // ASYMMETRIC_DATA_QUANTIZATION
                    slm_block[c + lid] = 0;
#endif  // ASYMMETRIC_DATA_QUANTIZATION
                }
            }
#if SLM_TAIL > 0
            if (sg == LWS1 - 1) {
                __local PACKED_IN_TYPE* slm_block_tail = slm + kh*SLM_LINE_SIZE + LWS1*SLM_CHUNK_SIZE;
                const int x_chunk = x_wg_start + LWS1*SLM_CHUNK_SIZE;
                bool x_cross_fm = x_chunk + lid >= INPUT0_SIZE_X;
                if (!x_cross_fm) {
                #if INPUT0_LAYOUT_BFYX
                    MAKE_VECTOR_TYPE(INPUT0_TYPE, ISV) src = 0;
                    __attribute__((opencl_unroll_hint(INPUT0_FEATURE_NUM)))
                    for (int i = 0; i < INPUT0_FEATURE_NUM; i++) {
                        src[i] = input[input_offset + i * INPUT0_FEATURE_PITCH
                                                    + kh * DILATION_SIZE_Y * INPUT0_Y_PITCH
                                                    + (x_chunk + lid)* INPUT0_X_PITCH];
                    }
                    slm_block_tail[lid] = AS_PACKED_IN_TYPE(src);
                #elif INPUT0_LAYOUT_B_FS_YX_FSV4
                    const __global uint* ptr = input + input_offset + kh * DILATION_SIZE_Y * input_y_pitch + (x_chunk + lid) * input_x_pitch;
                    PACKED_IN_TYPE src = AS_PACKED_IN_TYPE(ptr[0]);
                    slm_block_tail[lid] = src;
                #endif
                } else {
#if ASYMMETRIC_DATA_QUANTIZATION
                    slm_block_tail[lid] = AS_PACKED_IN_TYPE(zp);
#else  // ASYMMETRIC_DATA_QUANTIZATION
                    slm_block_tail[lid] = 0;
#endif  // ASYMMETRIC_DATA_QUANTIZATION
                }
            }
#endif
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    __attribute__((opencl_unroll_hint(FILTER_SIZE_Y)))
    for (int kh = 0; kh < FILTER_SIZE_Y ; ++kh) {
        bool y_cross_fm = input_y + kh*DILATION_SIZE_Y < 0 || input_y + kh*DILATION_SIZE_Y >= INPUT0_SIZE_Y;
#if !ASYMMETRIC_DATA_QUANTIZATION
        if (y_cross_fm)
            continue;
#endif
        PACKED_IN_TYPE line_cache[INPUT_LINE_SIZE];
        for (int xb = 0; xb < INPUT_LINE_SIZE; xb++) {
            line_cache[xb] = slm[kh*SLM_LINE_SIZE + sg*OUTPUT_X_BLOCK_SIZE*STRIDE_SIZE_X + xb];
        }

        __attribute__((opencl_unroll_hint(FILTER_SIZE_X)))
        for (uint kw = 0; kw < FILTER_SIZE_X ; ++kw) {
            const uint f_off = filter_idx
                             + kh * OSV * ISV * FILTER_SIZE_X
                             + kw * OSV * ISV;

            int weights_data0 = as_int(intel_sub_group_block_read((const __global uint*)(weights + f_off)));
            int weights_data1 = as_int(intel_sub_group_block_read((const __global uint*)(weights + f_off + SUB_GROUP_SIZE*ISV)));

            PACKED_TYPE_VEC src;

            __attribute__((opencl_unroll_hint(OUTPUT_X_BLOCK_SIZE)))
            for (int i = 0; i < OUTPUT_X_BLOCK_SIZE; i++) {
                // src[i] = slm[kh*SLM_LINE_SIZE + (sg*OUTPUT_X_BLOCK_SIZE + i)*STRIDE_SIZE_X + kw*DILATION_SIZE_X];
                src[i] = line_cache[kw*DILATION_SIZE_X + STRIDE_SIZE_X*i];
                acc[0][i] = IMAD(acc[0][i], AS_INPUT0_TYPE_4(src[i]), as_char4(weights_data0));
                acc[1][i] = IMAD(acc[1][i], AS_INPUT0_TYPE_4(src[i]), as_char4(weights_data1));

#if ASYMMETRIC_WEIGHTS_QUANTIZATION
                acc_assym_weights[i] = IMAD(acc_assym_weights[i], AS_INPUT0_TYPE_4(src[i]), multiplier);
#endif
            }
        }
    }

#if BIAS_TERM
    const uint bias_index = fg*OSV;
#endif

#if OUTPUT_IS_FP
    MAKE_VECTOR_TYPE(PACKED_OUT_TYPE, OUTPUT_X_BLOCK_SIZE) dst[2];

    for (int i = 0; i < OUTPUT_X_BLOCK_SIZE; i++) {
#if BIAS_TERM
        ACTIVATION_TYPE res0 = TO_ACTIVATION_TYPE(acc[0][i]) + (ACTIVATION_TYPE)(biases[bias_index + lid]);
        ACTIVATION_TYPE res1 = TO_ACTIVATION_TYPE(acc[1][i]) + (ACTIVATION_TYPE)(biases[bias_index + lid + SUB_GROUP_SIZE]);
#else
        ACTIVATION_TYPE res0 = TO_ACTIVATION_TYPE(acc[0][i]);
        ACTIVATION_TYPE res1 = TO_ACTIVATION_TYPE(acc[1][i]);
#endif

#if ASYMMETRIC_WEIGHTS_QUANTIZATION
        res0 -= acc_assym_weights[i] * TO_ACCUMULATOR_TYPE(weights_zp[fg * OSV + lid + 0]);
        res1 -= acc_assym_weights[i] * TO_ACCUMULATOR_TYPE(weights_zp[fg * OSV + lid + 16]);
#endif  // ASYMMETRIC_WEIGHTS_QUANTIZATION

#if ASYMMETRIC_DATA_QUANTIZATION
        res0 += compensation[fg*OSV + lid];
        res1 += compensation[fg*OSV + lid + SUB_GROUP_SIZE];
#endif  // ASYMMETRIC_DATA_QUANTIZATION
#if HAS_FUSED_OPS
        { FUSED_OPS_0; dst[0][i] = FUSED_OPS_RESULT_0; };
        { FUSED_OPS_1; dst[1][i] = FUSED_OPS_RESULT_1; };
#else
        dst[0][i] = TO_OUTPUT_TYPE(res0);
        dst[1][i] = TO_OUTPUT_TYPE(res1);
#endif
    }

#if OUTPUT_LAYOUT_B_FS_YX_FSV32
    for (int i = 0; i < OUTPUT_X_BLOCK_SIZE; i++) {
        for (int ofm = 0; ofm < 2; ofm++) {
            const uint dst_index = OUTPUT_GET_INDEX(b, fg*OSV + SUB_GROUP_SIZE*ofm + lid, y, x+i);
            if (x + i < OUTPUT_SIZE_X) {
                output[dst_index] = dst[ofm][i];
            }
        }
    }
#else // OUTPUT_LAYOUT_B_FS_YX_FSV32
    for (int i = 0; i < OUTPUT_X_BLOCK_SIZE; i++) {
#if OUTPUT_FEATURE_NUM > 16
        for (int ofm = 0; ofm < 2; ofm++) {
            const uint dst_index = OUTPUT_GET_INDEX(b, fg*OSV + SUB_GROUP_SIZE*ofm + lid, y, x+i);
            if (x + i < OUTPUT_SIZE_X && fg*OSV + SUB_GROUP_SIZE*ofm + lid < OUTPUT_FEATURE_NUM) {
                output[dst_index] = dst[ofm][i];
            }
        }
#else // OUTPUT_FEATURE_NUM > 16
        const uint dst_index = OUTPUT_GET_INDEX(b, fg*OSV + lid, y, x+i);
        if (x + i < OUTPUT_SIZE_X && fg*OSV + lid < OUTPUT_FEATURE_NUM) {
            output[dst_index] = dst[0][i];
        }
#endif // OUTPUT_FEATURE_NUM > 16
    }
#endif // OUTPUT_LAYOUT_B_FS_YX_FSV32

#else  // OUTPUT_IS_FP
#if OUTPUT_LAYOUT_B_FS_YX_FSV32
    MAKE_VECTOR_TYPE(PACKED_OUT_TYPE, OUTPUT_X_BLOCK_SIZE) dst;
    #define CHANNEL0_OFFSET (2*lid+0)
    #define CHANNEL1_OFFSET (2*lid+1)
#else // OUTPUT_LAYOUT_B_FS_YX_FSV32
    MAKE_VECTOR_TYPE(OUTPUT_TYPE, OUTPUT_X_BLOCK_SIZE) dst[2];
    #define CHANNEL0_OFFSET (lid)
    #define CHANNEL1_OFFSET (lid+16)
#endif // OUTPUT_LAYOUT_B_FS_YX_FSV32


    for (int i = 0; i < OUTPUT_X_BLOCK_SIZE; i++) {
#if BIAS_TERM
        ACTIVATION_TYPE res0 = TO_ACTIVATION_TYPE(acc[0][i]) + (ACTIVATION_TYPE)(biases[bias_index + CHANNEL0_OFFSET]);
        ACTIVATION_TYPE res1 = TO_ACTIVATION_TYPE(acc[1][i]) + (ACTIVATION_TYPE)(biases[bias_index + CHANNEL1_OFFSET]);
#else
        ACTIVATION_TYPE res0 = TO_ACTIVATION_TYPE(acc[0][i]);
        ACTIVATION_TYPE res1 = TO_ACTIVATION_TYPE(acc[1][i]);
#endif

#if ASYMMETRIC_WEIGHTS_QUANTIZATION
        res0 -= acc_assym_weights[i] * TO_ACCUMULATOR_TYPE(weights_zp[fg * OSV + CHANNEL0_OFFSET]);
        res1 -= acc_assym_weights[i] * TO_ACCUMULATOR_TYPE(weights_zp[fg * OSV + CHANNEL1_OFFSET]);
#endif  // ASYMMETRIC_WEIGHTS_QUANTIZATION

#if ASYMMETRIC_DATA_QUANTIZATION
        res0 += compensation[fg*OSV + CHANNEL0_OFFSET];
        res1 += compensation[fg*OSV + CHANNEL1_OFFSET];
#endif  // ASYMMETRIC_DATA_QUANTIZATION

        MAKE_VECTOR_TYPE(OUTPUT_TYPE, 2) pack;
#if HAS_FUSED_OPS
        { FUSED_OPS_0; pack[0] = FUSED_OPS_RESULT_0; };
        { FUSED_OPS_1; pack[1] = FUSED_OPS_RESULT_1; };
#else
        pack[0] = TO_OUTPUT_TYPE(res0);
        pack[1] = TO_OUTPUT_TYPE(res1);
#endif
#if OUTPUT_LAYOUT_B_FS_YX_FSV32
        dst[i] = AS_PACKED_OUT_TYPE(pack);
#else // OUTPUT_LAYOUT_B_FS_YX_FSV32
        dst[0][i] = pack[0];
        dst[1][i] = pack[1];
#endif // OUTPUT_LAYOUT_B_FS_YX_FSV32
    }

    const bool full_x = OUTPUT_SIZE_X % OUTPUT_X_BLOCK_SIZE == 0 || x + OUTPUT_X_BLOCK_SIZE <= OUTPUT_SIZE_X;
    const bool full_f = OUTPUT_FEATURE_NUM % OSV == 0 || (fg + 1) * OSV <= OUTPUT_FEATURE_NUM;
#if OUTPUT_LAYOUT_B_FS_YX_FSV32
    if (full_x && full_f) {
        const uint dst_index = OUTPUT_GET_INDEX(b, fg*OSV, y, x) / 2;
        BLOCK_WRITE(output + dst_index, dst);
    } else {
        for (int i = 0; i < OUTPUT_X_BLOCK_SIZE; i++) {
            const bool full_it_x = OUTPUT_SIZE_X % OUTPUT_X_BLOCK_SIZE == 0 || x + i < OUTPUT_SIZE_X;
            const bool full_sgl_f = OUTPUT_FEATURE_NUM % OSV == 0 || fg * OSV + 2 * lid < OUTPUT_FEATURE_NUM;
            if (full_it_x && full_sgl_f) {
                const uint dst_index = OUTPUT_GET_INDEX(b, fg*OSV + 2*lid, y, x+i);
                output[dst_index/2] = dst[i];
            }
        }
    }
#else // OUTPUT_LAYOUT_B_FS_YX_FSV32
    if (full_x && full_f) {
        const uint dst_index0 = OUTPUT_GET_INDEX(b, fg*OSV, y, x);
        const uint dst_index1 = OUTPUT_GET_INDEX(b, fg*OSV+16, y, x);
        BLOCK_WRITE(output + dst_index0, dst[0]);
        BLOCK_WRITE(output + dst_index1, dst[1]);
    } else {
        for (int ofm = 0; ofm < 2; ofm++) {
            for (int i = 0; i < OUTPUT_X_BLOCK_SIZE; i++) {
                const bool full_it_x = OUTPUT_SIZE_X % OUTPUT_X_BLOCK_SIZE == 0 || x + i < OUTPUT_SIZE_X;
                const bool full_sgl_f = OUTPUT_FEATURE_NUM % OSV == 0 || 16*ofm + lid < OUTPUT_FEATURE_NUM % OSV;
                if (full_it_x && full_sgl_f) {
                    const uint dst_index = OUTPUT_GET_INDEX(b, fg*OSV + 16*ofm + lid, y, x+i);
                    output[dst_index] = dst[ofm][i];
                }
            }
        }
    }
#endif // OUTPUT_LAYOUT_B_FS_YX_FSV32

#endif  // OUTPUT_IS_FP
}
#undef CEIL_DIV
#undef PACKED_TYPE_VEC
#undef ACCUMULATOR_TYPE_VEC
#undef TO_ACCUMULATOR_TYPE_VEC
#undef ACTIVATION_TYPE_VEC
#undef TO_ACTIVATION_TYPE_VEC
#undef MMAD

#undef AS_TYPE_N_
#undef AS_TYPE_N
#undef AS_INPUT0_TYPE_4
