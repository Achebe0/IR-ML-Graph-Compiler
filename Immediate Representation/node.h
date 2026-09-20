//
// Created by acheb on 2026-09-19.
//

#ifndef IR_ML_GRAPH_COMPILER_NODE_H
#define IR_ML_GRAPH_COMPILER_NODE_H

#endif //IR_ML_GRAPH_COMPILER_NODE_H
#include  <nlohmann/json.hpp>

struct Node {
    std::string id;
    std::string op;
    std::target target;
    std::nlohmann::json args;
    std::nlohmann::json kwargs;

};

struct Graph {
    //format
    //version
    //nodes
    //inputs and outputs
};