"""XPU dialect of ``T.Kernel``: grid of work-groups, SIMT inside."""

from __future__ import annotations

from tvm import tirx

from tilelang.language.kernel import KernelLaunchFrame, kernel_launch_factory, launch_kernel

__all__ = ["Kernel"]


@kernel_launch_factory
def Kernel(
    *blocks: int | tirx.PrimExpr,
    threads: int | list[int] | tuple[int, ...] | None = None,
) -> KernelLaunchFrame:
    """Construct a kernel launch frame for Intel XPU: a grid of work-groups.

    Code inside the launch operates at the work-group level: ``T.Parallel``,
    ``T.copy`` and friends are mapped onto work-items by the compiler.
    ``threads`` is the work-group size (default 128).
    """
    return launch_kernel(blocks, threads=threads)
