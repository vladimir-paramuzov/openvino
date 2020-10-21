// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <functional>
#include <cmath>
#include <memory>
#include <functional_test_utils/skip_tests_config.hpp>

#include "ie_core.hpp"
#include "ie_precision.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"

#include "single_layer_tests/prior_box_clustered.hpp"
#include "legacy/ngraph_ops/prior_box_clustered_ie.hpp"

namespace LayerTestsDefinitions {
std::string PriorBoxClusteredLayerTest::getTestCaseName(const testing::TestParamInfo<priorBoxClusteredLayerParams>& obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    InferenceEngine::SizeVector inputShapes, imageShapes;
    std::string targetDevice;
    priorBoxClusteredSpecificParams specParams;
    std::tie(specParams,
        netPrecision,
        inPrc, outPrc, inLayout, outLayout,
        inputShapes,
        imageShapes,
        targetDevice) = obj.param;

    std::vector<float> widths, heights, variances;
    float step_width, step_height, offset;
    bool clip;
    std::tie(widths,
        heights,
        clip,
        step_width,
        step_height,
        offset,
        variances) = specParams;

    std::ostringstream result;
    const char separator = '_';

    result << "IS="      << CommonTestUtils::vec2str(inputShapes) << separator;
    result << "imageS="  << CommonTestUtils::vec2str(imageShapes) << separator;
    result << "netPRC="  << netPrecision.name()   << separator;
    result << "inPRC="   << inPrc.name() << separator;
    result << "outPRC="  << outPrc.name() << separator;
    result << "inL="     << inLayout << separator;
    result << "outL="    << outLayout << separator;
    result << "widths="  << CommonTestUtils::vec2str(widths)  << separator;
    result << "heights=" << CommonTestUtils::vec2str(heights) << separator;
    result << "variances=";
    if (variances.empty())
        result << "()" << separator;
    else
        result << CommonTestUtils::vec2str(variances) << separator;
    result << "stepWidth="  << step_width  << separator;
    result << "stepHeight=" << step_height << separator;
    result << "offset="     << offset      << separator;
    result << "clip=" << std::boolalpha << clip << separator;
    result << "trgDev=" << targetDevice;
    return result.str();
}

void PriorBoxClusteredLayerTest::SetUp() {
    SetRefMode(LayerTestsUtils::RefMode::CONSTANT_FOLDING);
    priorBoxClusteredSpecificParams specParams;
    std::tie(specParams, netPrecision,
        inPrc, outPrc, inLayout, outLayout,
        inputShapes, imageShapes, targetDevice) = GetParam();

    std::tie(widths,
        heights,
        clip,
        step_width,
        step_height,
        offset,
        variances) = specParams;

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, { inputShapes, imageShapes });

    ngraph::op::PriorBoxClusteredAttrs attributes;
    attributes.widths = widths;
    attributes.heights = heights;
    attributes.clip = clip;
    attributes.step_widths = step_width;
    attributes.step_heights = step_height;
    attributes.offset = offset;
    attributes.variances = variances;

    auto shape_of_1 = std::make_shared<ngraph::opset3::ShapeOf>(params[0]);
    auto shape_of_2 = std::make_shared<ngraph::opset3::ShapeOf>(params[1]);
    auto priorBoxClustered = std::make_shared<ngraph::op::PriorBoxClustered>(
        shape_of_1,
        shape_of_2,
        attributes);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(priorBoxClustered) };
    function = std::make_shared<ngraph::Function>(results, params, "PB_Clustered");
}

TEST_P(PriorBoxClusteredLayerTest, CompareWithRefs) {
    Run();
};
}  // namespace LayerTestsDefinitions
