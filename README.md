# IR-ML-Graph-Compiler

A custom machine-learning graph compiler built in C++. The project exports
PyTorch models into an intermediate graph representation for processing by the
C++ compiler.

## PyTorch Frontend

`PyTorch frontend/model.py` defines a feed-forward network and captures its
operations with `torch.export.export()`. Export records the model graph without
running the model on input data.

Run the frontend with:

```bash
python "PyTorch frontend/model.py"
```

The script prints the captured graph as formatted JSON. The exported graph
contains placeholders, linear operations, the ReLU activation, and the output
node.

## Workflow

```text
PyTorch model
    -> torch.export.export()
    -> exported graph
    -> JSON representation
    -> C++ graph compiler
```

The JSON representation is intended as the interface between the Python
frontend and the C++ compiler.


**PyTorch model (nn.Module)
     ↓
FX export.export() — records ops without executing them
     ↓
Serialize FX graph → JSON (your schema: op, inputs, shape/dtype, attrs)
     ↓
C++ parses JSON → builds Node/Graph IR objects
     ↓
Pass 1: Dead node elimination — DFS backward from outputs, delete unreached nodes
     ↓
Pass 2: Constant folding — evaluate all-constant subgraphs once, replace with constant node
     ↓
Pass 3: Operator fusion — merge matmul→relu (single consumer) into one fused node
     ↓
Topological sort — determine valid execution order
     ↓
Backend lowering:
   - CPU: naive per-node execution
   - CUDA: fused nodes → hand-written kernel; rest → naive
     ↓
Benchmark: your pipeline vs. PyTorch eager vs. torch.compile
**
