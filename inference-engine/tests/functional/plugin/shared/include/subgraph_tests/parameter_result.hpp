// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
            std::string                        // Device name
> parameterResultParams;

class ParameterResultSubgraphTest : public testing::WithParamInterface<parameterResultParams>,
                                    virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<parameterResultParams> obj);
protected:
    void SetUp() override;
};
}  // namespace LayerTestsDefinitions
