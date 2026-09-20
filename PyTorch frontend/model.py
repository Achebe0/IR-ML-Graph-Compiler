import torch
import torch.nn as nn
import json


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.act = nn.ReLU()
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w_2(self.act(self.w_1(x)))



class CNNClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(32 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = x.flatten(1)
        return self.fc(x)


def _arg_to_ref(arg):
    """Convert an FX node arg to a JSON-safe reference: node name, literal, or nested list/tuple."""
    if isinstance(arg, torch.fx.Node):
        return {"ref": arg.name}
    if isinstance(arg, (list, tuple)):
        return [_arg_to_ref(a) for a in arg]
    if isinstance(arg, (int, float, bool)) or arg is None:
        return arg
    return str(arg)


def serialize_graph(exported_model) -> dict:
    gm = exported_model.graph_module
    nodes = []

    for node in gm.graph.nodes:
        entry = {
            "name": node.name,
            "op": node.op,                      # placeholder / call_function / output / get_attr
            "target": str(node.target),
            "args": [_arg_to_ref(a) for a in node.args],
            "kwargs": {k: _arg_to_ref(v) for k, v in node.kwargs.items()},
            "num_users": len(node.users),
        }

        # Pull shape/dtype from the FakeTensor metadata FX attaches during tracing
        val = node.meta.get("val")
        if isinstance(val, torch.Tensor):
            entry["shape"] = list(val.shape)
            entry["dtype"] = str(val.dtype)

        nodes.append(entry)

    return {"nodes": nodes}


def export_and_serialize(model, example_args, out_path, name):
    model.eval()
    exported = torch.export.export(model, example_args)

    print("Exported Graph")
    print(exported.graph)

    graph_json = serialize_graph(exported)
    with open(out_path, "w") as f:
        json.dump(graph_json, f, indent=2)

    print(f"Wrote {out_path} ({len(graph_json['nodes'])} nodes)")
    return exported


if __name__ == "__main__":
    # Simple model: batch=32, seq_len=128, d_model=512
    ffn_model = FeedForwardNetwork()
    ffn_args = (torch.randn(32, 128, 512),)
    export_and_serialize(ffn_model, ffn_args, "ffn_graph.json", "FFN")

    # Complex model: batch=4, 3-channel 32x32 images (CIFAR-scale)
    cnn_model = CNNClassifier()
    cnn_args = (torch.randn(4, 3, 32, 32),)
    export_and_serialize(cnn_model, cnn_args, "cnn_graph.json", "CNN")