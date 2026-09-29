from typing import final, List, Optional

from executorch.backends.aoti.aoti_partitioner import AotiPartitioner
from executorch.backends.xpu.xpu_backend import XpuBackend  # usort: skip
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.passes.propagate_device_pass import TARGET_DEVICE_COMPILE_SPEC_KEY


@final
class XpuPartitioner(AotiPartitioner):
    """XPU partitioner driven by AOTInductor.

    Delegates the whole non-lowered subgraph (like CudaPartitioner does for CUDA)
    and lets AOTInductor's Triton-XPU codegen / extern-kernel fallback decide how
    each op (add, mm, ...) is actually implemented -- no per-op kernel code here.
    """

    def __init__(self, compile_spec: Optional[List[CompileSpec]] = None) -> None:
        compile_spec = list(compile_spec) if compile_spec else []
        has_target_device = any(
            spec.key == TARGET_DEVICE_COMPILE_SPEC_KEY for spec in compile_spec
        )
        if not has_target_device:
            compile_spec.append(
                CompileSpec(TARGET_DEVICE_COMPILE_SPEC_KEY, b"xpu:0")
            )
        super().__init__(XpuBackend.__name__, compile_spec)
