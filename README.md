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


## Compilation & Execution Pipeline

```text
PyTorch model (`nn.Module`)
        │
        ▼
`FX export.export()`
Records operations without executing them
        │
        ▼
Serialize FX graph → JSON
Schema: `op`, `inputs`, `shape/dtype`, `attrs`
        │
        ▼
C++ parses JSON
Builds `Node` / `Graph` IR objects
        │
        ▼
Pass 1: Dead Node Elimination
DFS backward from outputs and remove unreached nodes
        │
        ▼
Pass 2: Constant Folding
Evaluate all-constant subgraphs once and replace them
with a constant node
        │
        ▼
Pass 3: Operator Fusion
Fuse `matmul → relu` (single consumer) into one fused node
        │
        ▼
Topological Sort
Determine a valid execution order
        │
        ▼
Backend Lowering
├── CPU
│   └── Naive per-node execution
│
└── CUDA
    ├── Fused nodes → Hand-written CUDA kernel
    └── Remaining nodes → Naive execution
        │
        ▼
Benchmark
Compare against:
├── PyTorch eager
└── `torch.compile`
