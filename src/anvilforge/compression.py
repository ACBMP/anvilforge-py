"""Port of AnvilToolkit.Compressions.Manager's decompression dispatch.

The original ships a custom `Libs/lzo.dll` whose exports (`lzo1x_decompress`,
`lzo2a_decompress`, `__lzo_init_v2`, ...) are the stock reference liblzo2 API,
so we bind straight to the system's liblzo2 via ctypes instead of shipping a
private copy. Zstd (CompAlgo 5) uses the `zstandard` package. Oodle (CompAlgo
8) is proprietary Epic Games tech with no available/redistributable decoder
here, so it's left unimplemented -- it's only used by titles this rewrite
doesn't target (Unity onward).
"""
from __future__ import annotations

import ctypes
import ctypes.util

_lzo = None


def _lzo_lib():
    global _lzo
    if _lzo is None:
        path = ctypes.util.find_library("lzo2") or "liblzo2.so.2"
        lib = ctypes.CDLL(path)
        lib.__lzo_init_v2.restype = ctypes.c_int
        if lib.__lzo_init_v2(1, -1, -1, -1, -1, -1, -1, -1, -1, -1) != 0:
            raise RuntimeError("failed to initialize liblzo2")
        _lzo = lib
    return _lzo


def _lzo_decompress(algorithm: int, data: bytes, uncompressed_size: int) -> bytes:
    lib = _lzo_lib()
    # The original tool P/Invokes the unsafe lzo1x_decompress/lzo2a_decompress
    # (no destination-bounds checking), which is fine for well-formed blocks
    # but can walk off the end of the destination buffer -- fatally, since
    # ctypes has no way to catch a native memory fault -- on malformed input
    # (observed on one edge-case entry in real ACB data that isn't actually a
    # compressed block). liblzo2 also ships "_safe" variants that are
    # bounds-checked and behave identically on valid input, so we use those
    # instead: same output, but a catchable error instead of a segfault.
    if algorithm in (0, 1):
        func = lib.lzo1x_decompress_safe
    elif algorithm == 2:
        func = lib.lzo2a_decompress_safe
    else:
        raise ValueError(f"unsupported LZO algorithm id {algorithm}")
    func.restype = ctypes.c_int

    src = ctypes.create_string_buffer(data, len(data))
    dst = ctypes.create_string_buffer(max(uncompressed_size, 1))
    dst_len = ctypes.c_ulong(uncompressed_size)
    wrkmem = ctypes.create_string_buffer(65536)

    rc = func(src, ctypes.c_ulong(len(data)), dst, ctypes.byref(dst_len), wrkmem)
    if rc != 0:
        raise RuntimeError(f"LZO decompression failed (return code {rc})")
    return dst.raw[: dst_len.value]


def _lzo_compress(algorithm: int, data: bytes) -> bytes:
    lib = _lzo_lib()
    if algorithm == 0:
        func = lib.lzo1x_1_compress
        extra = len(data) // 64 + 16 + 3 + 4
    elif algorithm == 2:
        func = lib.lzo2a_999_compress
        extra = len(data) // 8 + 128 + 3
    else:
        raise ValueError(f"unsupported LZO compress algorithm id {algorithm}")
    func.restype = ctypes.c_int

    src = ctypes.create_string_buffer(data, len(data)) if data else ctypes.create_string_buffer(1)
    dst = ctypes.create_string_buffer(len(data) + extra)
    dst_len = ctypes.c_ulong(0)
    # The original tool shares one fixed 512KB scratch buffer across every
    # algorithm (including lzo2a_999_compress, which is the hungriest); that
    # size is proven sufficient in production, so it's mirrored here rather
    # than trying to derive each algorithm's exact LZO*_MEM_COMPRESS macro.
    wrkmem = ctypes.create_string_buffer(524288)

    rc = func(src, ctypes.c_ulong(len(data)), dst, ctypes.byref(dst_len), wrkmem)
    if rc != 0:
        raise RuntimeError(f"LZO compression failed (return code {rc})")
    return dst.raw[: dst_len.value]


def compress(comp_algo: int, data: bytes) -> bytes:
    """Port of Compressions.LZO.Compress's algorithm dispatch (Oodle/Zstd
    compress paths aren't needed -- BlackFlag/AC3 use algorithm 0 and
    Brotherhood/Revelations use algorithm 2, both LZO). Falls back to storing
    data uncompressed if compression doesn't actually shrink it, matching the
    original's CompressionRatio-gated fallback."""
    if comp_algo in (0, 2):
        compressed = _lzo_compress(comp_algo, data)
        return compressed if len(compressed) < len(data) else data
    raise ValueError(f"unsupported compression algorithm id {comp_algo} for compress()")


def _zstd_decompress(data: bytes, uncompressed_size: int) -> bytes:
    import zstandard

    return zstandard.ZstdDecompressor().decompress(data, max_output_size=uncompressed_size)


def decompress(comp_algo: int, data: bytes, uncompressed_size: int) -> bytes:
    """Port of Compressions.Manager.Decompress."""
    if comp_algo in (0, 1, 2):
        return _lzo_decompress(comp_algo, data, uncompressed_size)
    if comp_algo == 5:
        return _zstd_decompress(data, uncompressed_size)
    if comp_algo == 8:
        raise NotImplementedError(
            "Oodle-compressed blocks (CompAlgo 8) are not supported by this rewrite; "
            "this shouldn't occur for ACB/ACR/AC3/AC4 content"
        )
    raise ValueError(f"unknown compression algorithm id {comp_algo}")
