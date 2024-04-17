// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/convolution_transformation.hpp"
#include "low_precision_transformations/convolution_with_incorrect_weights.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ov::element::Type> netPrecisions = {
    ov::element::f32,
    ov::element::f16
};

const std::vector<ov::pass::low_precision::LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams()
};

const std::vector<LayerTestsDefinitions::ConvolutionTransformationParam> params = {
    {
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        false,
        {},
        false,
        "Convolution",
        ov::element::f32.get_type_name()
    },
    {
        {},
        false,
        { 255ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -12.7f }, { 12.7f } },
        false,
        "Convolution",
        ov::element::f32.get_type_name()
    },
    {
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        false,
        { 255ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -12.7f }, { 12.7f } },
        false,
        "Convolution",
        ov::element::u8.get_type_name()
    },
    {
        { 256ul, ov::Shape {}, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        false,
        { 255ul, ov::Shape {}, { 0.f }, { 254.f }, { -12.7f }, { 12.7f } },
        false,
        "Convolution",
        ov::element::u8.get_type_name()
    },
    {
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { -12.75f }, { 6.375f } },
        true,
        { 255ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -12.7f }, { 12.7f } },
        false,
        "Convolution",
        ov::element::u8.get_type_name()
    },
    {
        { 256ul, ov::Shape { 1 }, { 0.f }, { 255.f }, { -18.7f }, { 18.8f } },
        true,
        {
            255ul, ov::Shape { 6, 1, 1, 1 }, { -0.6f }, { 0.6f },
            { -1.52806e-39f, -0.2, -0.3, -0.3, -0.2, -0.1 }, { 1.52806e-39f, 0.2, 0.3, 0.3, 0.2, 0.1 }
        },
        false,
        "Convolution",
        ov::element::u8.get_type_name()
    },
    {
        { 256ul, ov::Shape { 1 }, { 0.f }, { 255.f }, { -18.7f }, { 18.8f } },
        true,
        {
            255ul, ov::Shape { 6, 1, 1, 1 }, { -0.6f }, { 0.6f },
            { -1.52806e-39f, -1.52806e-39f, -1.52806e-39f, -1.52806e-39f, -1.52806e-39f, -1.52806e-39f },
            { 1.52806e-39f, 1.52806e-39f, 1.52806e-39f, 1.52806e-39f, 1.52806e-39f, 1.52806e-39f }
        },
        false,
        "Convolution",
        ov::element::u8.get_type_name()
    },
    // not supported quantization level on data
    {
        { 65536ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
        false,
        { 255ul, ov::Shape{1, 1, 1, 1}, {0.f}, {254.f}, {-12.7f}, {12.7f}},
        false,
        "Convolution",
        ov::element::f32.get_type_name()
    },
    // not supported quantization level on data & weights
    {
        { 65536ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        false,
        { 65536ul, ov::Shape{1, 1, 1, 1}, {0.f}, {254.f}, {-12.7f}, {12.7f}},
        false,
        "Convolution",
        ov::element::f32.get_type_name()
    },
    // not supported quantization level on weights
    {
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        false,
        { 65536ul, ov::Shape{1, 1, 1, 1}, {0.f}, {254.f}, {-12.7f}, {12.7f}},
        false,
        "Convolution",
        ov::element::f32.get_type_name()
    },
    {
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 0.f }, { 0.f }, { 0.f } },
        false,
        { 255ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -12.7f }, { 12.7f } },
        false,
        "Convolution",
        ov::element::u8.get_type_name()
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, ConvolutionTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::Shape({ 1, 3, 16, 16 })),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(params)),
    ConvolutionTransformation::getTestCaseName);

const std::vector<LayerTestsDefinitions::ConvolutionWIthIncorrectWeightsParam> incorrectWeightsParams = {
    // incorrect weights
    {
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        { 255ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
        false
    },
    // correct weights
    {
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 25.5f } },
        { 255ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
        true
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, ConvolutionWIthIncorrectWeightsTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::Shape({ 1, 3, 16, 16 })),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(incorrectWeightsParams)),
    ConvolutionWIthIncorrectWeightsTransformation::getTestCaseName);
}  // namespace
