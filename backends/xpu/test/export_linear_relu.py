"""Minimal Linear+ReLU export for XpuBackend (GEMM + elementwise). Run this on
the PTL (XPU-capable) machine, with a torch build that has torch.xpu working
(see session notes / conversation).
"""
import torch
from executorch.backends.xpu.xpu_backend import XpuBackend
from executorch.backends.xpu.xpu_partitioner import XpuPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.extension.export_util.utils import save_pte_program


class LinearReluModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x):
        return torch.relu(self.linear(x))


def main() -> None:
    model = LinearReluModule().eval()
    example_inputs = (torch.randn(4, 4),)
    exported_program = torch.export.export(model, example_inputs)

    partitioner = XpuPartitioner(
        [XpuBackend.generate_method_name_compile_spec("forward")]
    )
    edge_compile_config = EdgeCompileConfig(
        _check_ir_validity=False,
        _skip_dim_order=True,
    )
    et_program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[partitioner],
        compile_config=edge_compile_config,
    ).to_executorch()

    save_pte_program(et_program, "linear_relu_xpu", ".")
    print("Wrote linear_relu_xpu.pte")


if __name__ == "__main__":
    main()
