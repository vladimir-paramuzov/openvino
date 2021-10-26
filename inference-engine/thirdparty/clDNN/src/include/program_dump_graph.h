// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "cldnn/graph/program.hpp"
#include "cldnn/graph/program_node.hpp"
#include <fstream>
#include <string>

namespace cldnn {
std::string get_dir_path(build_options);
std::string get_serialization_network_name(build_options);

void dump_graph_optimized(std::ofstream&, const program&);
void dump_graph_processing_order(std::ofstream&, const program&);
void dump_graph_init(std::ofstream&, const program&, std::function<bool(program_node const&)> const&);
void dump_graph_info(std::ofstream&, const program&, std::function<bool(program_node const&)> const&);
}  // namespace cldnn
