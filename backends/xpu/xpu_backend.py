from typing import Any, Dict, List, Optional
from contextlib import contextmanager
from pathlib import Path

from executorch.backends.aoti.aoti_backend import AotiBackend
from executorch.exir.backend.backend_details import BackendDetails
from executorch.exir.backend.compile_spec_schema import CompileSpec

# Import-lib stub advertising every aoti_torch_* name the generated wrapper can
# reference (see backends/xpu/runtime/xpu_shims.def); mirrors CudaBackend's
# Windows-only aoti_cuda_shims.lib mechanism (cuda_backend.py get_aoti_compile_options).
_XPU_SHIM_LIBRARY_DIR = Path(__file__).parent / "runtime"


@contextmanager
def _skip_mingw_cross_compile_probe():
    """torch._inductor.cpp_builder.get_cpp_options() probes for a MinGW
    cross-compiler whenever aot_inductor.cross_target_platform == "windows",
    with no check for whether we're already natively on Windows (unlike
    get_cpp_compiler(), which correctly gates that same flag behind
    sys.platform != "win32"). We set cross_target_platform="windows" purely to
    get the xpu_shims.lib linking behavior below; there's no MinGW toolchain to
    probe for on a native Windows host. Stub the probe out for this compile.
    """
    from torch._inductor import cpp_builder

    orig = cpp_builder.check_mingw_win32_flavor
    cpp_builder.check_mingw_win32_flavor = lambda compiler: "win32"
    try:
        yield
    finally:
        cpp_builder.check_mingw_win32_flavor = orig


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
    def get_extra_aoti_compile_context_manager(
        cls, compile_specs: Optional[List[CompileSpec]] = None
    ):
        return _skip_mingw_cross_compile_probe()

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
            # Windows-only, but this PoC only targets Windows so no platform
            # branch yet (unlike CudaBackend, which parses a "platform" spec).
            # Without this, the AOTI-compiled DLL can't resolve aoti_torch_*
            # calls at link time -- Windows DLLs (unlike Linux .so) require
            # every import to resolve against a named library at link time.
            "aot_inductor.cross_target_platform": "windows",
            "aot_inductor.aoti_shim_library": "xpu_shims",
            "aot_inductor.aoti_shim_library_path": str(_XPU_SHIM_LIBRARY_DIR),
            "aot_inductor.precompile_headers": False,
        }


