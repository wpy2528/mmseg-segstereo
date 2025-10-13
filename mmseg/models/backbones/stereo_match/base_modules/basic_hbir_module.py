import torch
import torch.nn as nn

from .utils.package_helper import require_packages

try:
    from hbdk4.compiler import load
    from horizon_plugin_pytorch.quantization.hbdk4 import (
        get_hbir_input_flattener,
        get_hbir_output_unflattener,
    )
except ImportError:
    load = None
    get_hbir_input_flattener = None
    get_hbir_output_unflattener = None


class BaseHbirModule(nn.Module):
    @require_packages("hbdk4")
    def __init__(
        self,
        model_path,
    ):
        super(BaseHbirModule, self).__init__()

        self.model = load(model_path)
        self.input_flattener = get_hbir_input_flattener(self.model)
        self.output_unflattener = get_hbir_output_unflattener(self.model)

    def flatten_hbir_input(self, batched_data):

        flat_input = self.input_flattener(batched_data)
        flat_input = [x.cpu().numpy() for x in flat_input]

        return flat_input

    def unflatten_hbir_output(self, hbir_output):
        hbir_output = self.output_unflattener(hbir_output)
        return hbir_output

    def output_to_tensor(self, hbir_output):
        if isinstance(hbir_output, list):
            for k in range(len(hbir_output)):
                hbir_output[k] = self.output_to_tensor(hbir_output[k])
        elif isinstance(hbir_output, tuple):
            hbir_output = list(self.output_to_tensor(list(hbir_output)))
        elif isinstance(hbir_output, dict):
            for k in hbir_output:
                hbir_output[k] = self.output_to_tensor(hbir_output[k])
        else:
            hbir_output = torch.from_numpy(hbir_output.copy())
        return hbir_output

    def forward(self, batched_data):

        hbir_output = self.model.functions[0](
            *self.flatten_hbir_input(batched_data)
        )
        hbir_output = self.output_unflattener(hbir_output)
        hbir_output = self.output_to_tensor(hbir_output)

        return hbir_output
