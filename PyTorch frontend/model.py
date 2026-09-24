import torch
import torch.nn as nn
from torch.export import export, ExportedProgram
import json
import pickle


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.act = nn.ReLU()
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w_2(self.act(self.w_1(x)))


## recording the operations without execution
def torch_export_ffn(d_model=512, d_ff=2048, batch_size=1):
    model = FeedForwardNetwork(d_model= d_model, d_ff=d_ff).eval()
    example_input = torch.randn(batch_size, d_model)
    return torch.export.export(model, (example_input,))


def arg_to_ref(arg):
    if isinstance(arg, torch.fx.Node):
        return  {"kind" : "node_ref", "name": arg.name}

    if isinstance(arg, (list,tuple)):
        return {"kind": "literal", "value": [arg_to_ref(item) for item in arg]
                }

    if isinstance(arg, dict):
        return {
            "kind":"dict",
            "value": {
                str(key): arg_to_ref(value)
                for key, value in arg.items()
        }
        }

    if isinstance(arg, (int, float, bool)) or arg is None:
        return {"kind": "literal", "value": arg}

    if isinstance(arg, torch.dtype):
        return {"kind": "dtype", "value": str(arg)}
    if isinstance(arg, torch.memory_format):
        return {"kind": "memory_format", "value": str(arg)}

    raise TypeError(f"Unhandled arg type in serializer: {type(arg)} = {arg!r}")

# gets the underlying graph objects
def serialize_graph_json(exported_program):
    return {
        "nodes":[
            {
                "name":node.name,
                "op" : node.op,
                "target" : str(node.target),
                "args" : [arg_to_ref(arg) for arg in node.args],
                "kwargs" : {
                    key: arg_to_ref(value)
                    for key, value in node.kwargs.items()
                }
            }
            for node in exported_program.graph_module.graph.nodes
        ]
    }

def to_JSON_file(exported_program, pathway):

    graph_data = serialize_graph_json(exported_program)

    with open(pathway, "w") as file:
        json.dump(graph_data, file, indent=2)


if __name__ == "__main__":
    exported_program = torch_export_ffn()
    to_JSON_file(exported_program,"ffn_graph.json")
