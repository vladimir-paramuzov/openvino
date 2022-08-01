// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/runtime/lru_cache.hpp"
#include <vector>

using namespace cldnn;
using namespace ::tests;

class lru_cache_test_data {
public:
    lru_cache_test_data(int a, int b, int c) : x(a), y(b), z(c) {
        key = "key_" + std::to_string(a) + "_" + std::to_string(b) + "_" + std::to_string(c);
    }

    bool operator==(const lru_cache_test_data&rhs) {
        return (this->x == rhs.x && this->y == rhs.y && this->z == rhs.z);
    }

    bool operator!=(const lru_cache_test_data&rhs) {
        return (this->x != rhs.x || this->y != rhs.y || this->z != rhs.z);
    }

    operator std::string() {
        return "(" + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z) + ")";
    }

    std::string key;
    int x;
    int y;
    int z;

};

TEST(lru_cache, basic_data_type)
{
    const size_t cap = 4;
    LRUCache<int, int> ca(cap * sizeof(int));

    std::vector<int> inputs = {1, 2, 3, 4, 2, 1, 5};
    std::vector<std::pair<int, int>> input_values;
    for (auto i :  inputs) {
        input_values.push_back(std::make_pair(i, i + 10));
    }

    std::vector<bool> expected_hitted = {false, false, false, false, true, true, false};
    for (size_t i = 0; i < input_values.size(); i++) {
        auto& in = input_values[i];
        int data = 0;
        bool hitted = ca.has(in.first);
        if (hitted)
            data = ca.get(in.first);
        else
            data = ca.add(in.first, [in](){
                return LRUCache<int, int>::CacheEntry{in.second, sizeof(in.second)};});
        EXPECT_EQ(data, in.second);
        EXPECT_EQ(hitted, (bool)expected_hitted[i]);
    }

    EXPECT_EQ(cap, ca.count());

    std::vector<std::pair<int, int>> expected_value;
    for (size_t i = cap; i > 0; i--) {  // 5, 1, 2, 4
        int idx = input_values.size() - i;
        expected_value.push_back(input_values[idx]);
    }

    int idx = expected_value.size() - 1;
    for (auto key : ca.get_all_keys()) {
        EXPECT_EQ(key, expected_value[idx--].first);
    }
}

TEST(lru_cache, custom_data_type) {
    const size_t cap = 4;
    LRUCache<std::string, std::shared_ptr<lru_cache_test_data>> ca(4 * sizeof(lru_cache_test_data));

    std::vector<std::shared_ptr<lru_cache_test_data>> inputs;
    inputs.push_back(std::make_shared<lru_cache_test_data>(1, 21, 11));
    inputs.push_back(std::make_shared<lru_cache_test_data>(2, 22, 12));
    inputs.push_back(std::make_shared<lru_cache_test_data>(3, 23, 13));
    inputs.push_back(std::make_shared<lru_cache_test_data>(4, 24, 14));
    inputs.push_back(std::make_shared<lru_cache_test_data>(2, 22, 12));
    inputs.push_back(std::make_shared<lru_cache_test_data>(1, 21, 11));
    inputs.push_back(std::make_shared<lru_cache_test_data>(3, 23, 13));
    inputs.push_back(std::make_shared<lru_cache_test_data>(5, 25, 15));

    std::vector<bool> expected_hitted = {false, false, false, false, true, true, true, false};

    for (size_t i = 0; i < inputs.size(); i++) {
        auto& in = inputs[i];
        std::shared_ptr<lru_cache_test_data> p_data;
        bool hitted = ca.has(in->key);
        if (hitted)
            p_data = ca.get(in->key);
        else
            p_data = ca.add(in->key, [in](){
                return LRUCache<std::string, std::shared_ptr<lru_cache_test_data>>::CacheEntry{in, sizeof(lru_cache_test_data)};});
        EXPECT_EQ(p_data->key, in->key);
        EXPECT_EQ(hitted, (bool)expected_hitted[i]);
    }

    EXPECT_EQ(cap, ca.count());

    std::vector<std::string> expected_keys;
    for (size_t i = cap; i > 0; i--) {
        expected_keys.push_back(inputs[inputs.size() - i]->key);
    }

    int idx = expected_keys.size() - 1;
    for (auto key : ca.get_all_keys()) {
        EXPECT_EQ(key, expected_keys[idx--]);
    }
}
