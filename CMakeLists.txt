cmake_minimum_required(VERSION 3.20)

project(IR_ML_Graph_Compiler)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(IR_ML_Graph_Compiler
        main.cpp
        "Immediate Representation/node.h"
        "Immediate Representation/graph.h"
        "Immediate Representation/json_parser.cpp"
        "Immediate Representation/json_parser.h"
)