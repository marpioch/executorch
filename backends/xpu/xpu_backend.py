from typing import Any, Dict, List

from executorch.backends.aoti.aoti_backend import AotiBackend
from executorch.exir.backend.backend_details import BackendDetails
from executorch.exir.backend.compile_spec_schema import CompileSpec


class XpuBackend(AotiBackend, BackendDetails):
    """Compiles a model to run on Intel XPU via AOTInductor.

    No preprocess() override: AotiBackend.preprocess() (torch._inductor.aot_compile)
    is used as-is, so add/mm/etc. are whatever AOTInductor's Triton-XPU codegen
    (or its extern-kernel fallback) decides, exactly like CudaBackend does for CUDA.
    """

    @classmethod
    def get_device_name(cls) -> str:
        return "xpu"

    @classmethod
    def get_supported_fallback_kernels(cls) -> Dict[str, Any]:
        return {}

    @classmethod
    def get_decomposition_table(cls) -> Dict[Any, Any]:
        return {}

    @classmethod
    def get_custom_passes(cls, compile_specs: List[CompileSpec]) -> List[Any]:
        return []

    @classmethod
    def get_aoti_compile_options(
        cls, compile_specs: List[CompileSpec]
    ) -> Dict[str, Any]:
        return {
            "aot_inductor.package": True,
            "aot_inductor.package_constants_in_so": False,
            "aot_inductor.link_libtorch": False,
            "max_autotune": True,
            "max_autotune_gemm_backends": "TRITON",
        }
