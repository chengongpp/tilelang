"""2D block-IO helper wrappers for the XPU DPAS GEMM global-direct path.

The OpenCL 2D block read builtins take an ``int2 coord`` parameter which
TIR ``call_extern`` cannot construct; these tiny wrappers take flat scalar
coordinates and return the loaded vector by value.  Attached automatically
to every xpu-dialect kernel via ``T.Kernel`` (pragma_import_c); unused
helpers are eliminated by the compiler.
"""

XPU_DPAS_PRELUDE = r"""
static inline short8 tl_xpu_rd_a16(const __global void* p, int w, int h,
                                   int pitch, int x, int y) {
  ushort8 v;
  intel_sub_group_2d_block_read_16b_8r16x1c(p, w, h, pitch, (int2)(x, y),
                                            (private ushort*)&v);
  return as_short8(v);
}
static inline int8 tl_xpu_rd_b16(const __global void* p, int w, int h,
                                 int pitch, int x, int y) {
  uint8 v;
  intel_sub_group_2d_block_read_transform_16b_16r16x1c(p, w, h, pitch,
                                                       (int2)(x, y),
                                                       (private uint*)&v);
  return as_int8(v);
}
"""
