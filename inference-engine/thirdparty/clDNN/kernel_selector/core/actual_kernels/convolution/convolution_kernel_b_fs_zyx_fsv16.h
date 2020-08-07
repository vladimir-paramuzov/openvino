﻿//
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
//

#pragma once

#include "convolution_kernel_base.h"
#include <vector>

namespace kernel_selector {

class ConvolutionKernel_b_fs_zyx_fsv16 : public ConvolutionKernelBase {
public:
    using Parent = ConvolutionKernelBase;

    explicit ConvolutionKernel_b_fs_zyx_fsv16() : ConvolutionKernelBase("gen9_common_conv_fwd_data") {}

    virtual ~ConvolutionKernel_b_fs_zyx_fsv16() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    WeightsLayout GetPreferredWeightsLayout(const convolution_params& params) const override {
        bool is_3d_case = params.inputs[0].Dimentions() == 5;
        if (params.inputs[0].Feature().v == 3) {
            if (is_3d_case)
                return WeightsLayout::os_zyxi_osv16;
            else
                return WeightsLayout::os_yxi_osv16;
        } else if (params.inputs[0].GetDType() == Datatype::F32 && params.inputs[0].Batch().v % 16 == 0) {
            if (is_3d_case)
                return (params.groups > 1) ? WeightsLayout::g_is_os_zyx_osv16_isv16 : WeightsLayout::is_os_zyx_osv16_isv16;
            else
                return (params.groups > 1) ? WeightsLayout::g_is_os_yx_osv16_isv16 : WeightsLayout::is_os_yx_osv16_isv16;
        } else if (params.inputs[0].GetDType() == Datatype::F16 && params.inputs[0].Batch().v % 32 == 0) {
            if (is_3d_case)
                return (params.groups > 1) ? WeightsLayout::g_os_is_zyx_isv8_osv16_isv2 : WeightsLayout::os_is_zyx_isv8_osv16_isv2;
            else
                return (params.groups > 1) ? WeightsLayout::g_os_is_yx_isv8_osv16_isv2 : WeightsLayout::os_is_yx_isv8_osv16_isv2;
        } else {
            if (is_3d_case)
                return (params.groups > 1) ? WeightsLayout::g_os_is_zyx_isv16_osv16 : WeightsLayout::os_is_zyx_isv16_osv16;
            else
                return (params.groups > 1) ? WeightsLayout::g_os_is_yx_isv16_osv16 : WeightsLayout::os_is_yx_isv16_osv16;
        }
    }
    bool Validate(const Params& p, const optional_params& o) const override;
    DispatchData SetDefault(const convolution_params& arg, int autoTuneIndex = -1) const override;
    JitConstants GetJitConstants(const convolution_params& params, const DispatchData& kd) const override;

    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::ELTWISE,
                 FusedOpType::QUANTIZE,
                 FusedOpType::SCALE,
                 FusedOpType::ACTIVATION };
    }
};
}  // namespace kernel_selector
