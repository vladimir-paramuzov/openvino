// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "pass_manager.h"
#include "cldnn/graph/program_node.hpp"
#include "layout_optimizer.h"
#include "cldnn/graph/program.hpp"
#include "program_helpers.h"
#include "runtime/cldnn_itt.hpp"
#include <vector>
#include <memory>
#include <list>
#include <map>
#include <set>

using namespace cldnn;

void basic_memory_dependencies::run(program& p) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNN, "CLDNN::pass::BasicMemoryDependencies");
    auto itr = p.get_processing_order().begin();
    std::vector<primitive_id> past_outputs;
    while (itr != p.get_processing_order().end()) {
        auto& node = *itr;
        itr++;

        // data primitive can't be reused
        if (node->is_type<data>())
            continue;

        // add my dependencies to restriction list (can't share input.output buffers)
        for (auto it : node->get_dependencies()) {
            add_memory_dependency(node, it);
            add_memory_dependency(it, node);
        }

        // Note we iterate over processing order, it means if primitve has processing num greater than any of outputs,
        // this output has to land on the primitve restriction list. Otherwise memory reuse can corrupt final results.
        node->add_memory_dependency(past_outputs);
        // if current node is an output add it to the outputs list after restriction.
        if (node->is_output())
            past_outputs.push_back(node->id());
    }
}
