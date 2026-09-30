"""Intel XPU backend manifest.

Phase 1: the backend owns TVM's built-in ``opencl`` target kind and delegates
device codegen to ``target.build.opencl`` (source-level OpenCL C, compiled by
the Intel NEO OpenCL driver at load time). Execution goes through the shared
``tvm_ffi`` backend and TVM's OpenCL runtime (dlopen'd libOpenCL).

Later phases replace the device codegen with a SYCL source codegen
(``codegen_sycl.cc``) plus a Level-Zero execution path for zero-copy interop
with ``torch.xpu`` USM tensors and XMX/DPAS matrix instructions.
"""

from tilelang.backend.device_codegen import DeviceCodegen
from tilelang.backend.execution_backend import ExecutionBackendSpec
from tilelang.backend.host_codegen import STANDARD_HOST_CODEGENS
from tilelang.backend.pass_pipeline import PassPipeline
from tilelang.backend.module import BackendModule, register_backend

from . import codegen, pipeline, target as _target  # noqa: F401  (target registers detector/normalizer)

BACKEND = register_backend(
    BackendModule(
        name="xpu",
        target_kinds=("opencl",),
        pipelines={"opencl": PassPipeline("opencl", pipeline.XPUPassPipelineBody)},
        device_codegens={
            "opencl": DeviceCodegen(
                "opencl",
                build=codegen.build_xpu,
                build_without_compile=codegen.build_xpu_without_compile,
            )
        },
        execution_backends=(
            ExecutionBackendSpec("tvm_ffi", enable_host_codegen=True, enable_device_compile=True),
        ),
        host_codegens=STANDARD_HOST_CODEGENS,
    )
)
