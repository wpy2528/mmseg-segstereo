import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

    def forward(self, x, y):
        z = 2 * x + y
        return z

if __name__ == "__main__":
    model = SimpleNet()
    model.eval()
    x = torch.randn(1, 3, 544, 960)
    y = torch.randn(1, 3, 544, 960)
    input_names = ["left", "right"]
    output_names = ["output"]
    torch.onnx.export(
        model,
        (x, y),
        "cursor_utils/simple_net.onnx",
        input_names=input_names,
        output_names=output_names,
        opset_version=11
    )
