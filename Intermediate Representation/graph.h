//
// Created by acheb on 2026-09-19.
//

#ifndef IR_ML_GRAPH_COMPILER_GRAPH_H
#define IR_ML_GRAPH_COMPILER_GRAPH_H

#include "node.h"

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

struct Graph {
    std::string format;
    int version;
    std::vector<Node> nodes;
    nlohmann::json input;
    nlohmann::json output;
};

#endif // IR_ML_GRAPH_COMPILER_GRAPH_H