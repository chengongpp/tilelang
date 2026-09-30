# Intel XPU backend: operator lowering registrations.
#
# Phase 1 targets TVM's OpenCL codegen/runtime (pure C++, always safe to
# compile). A SYCL/Level-Zero native path will be added behind USE_XPU later.

file(GLOB TILE_LANG_XPU_SRCS
  src/xpu/op/*.cc
)
list(APPEND TILE_LANG_SRCS ${TILE_LANG_XPU_SRCS})
