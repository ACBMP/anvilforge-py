"""Binary helpers ported from AnvilToolkit.Utils.Extensions / StringHelper.

All reads/writes are little-endian, matching BinaryReader/BinaryWriter's
default behavior in the original C#. `endian`/byte-swap variants from the
C# source aren't needed here since every game this rewrite targets always
calls those helpers with endian=false.
"""
from __future__ import annotations

import math
from typing import BinaryIO

MASK64 = (1 << 64) - 1


def read_cstring(f: BinaryIO) -> str:
    """Port of StringHelper.ReadNullTerminatedString."""
    buf = bytearray()
    while True:
        b = f.read(1)
        if not b or b == b"\x00":
            break
        buf += b
    return buf.decode("utf-8", errors="replace")


def write_cstring(f: BinaryIO, s: str) -> None:
    """Port of StringHelper.WriteNullTerminatedString."""
    f.write(s.encode("utf-8"))
    f.write(b"\x00")


def round_up(number: int, increment: int, offset: int = 0) -> int:
    """Port of Extensions.RoundUp."""
    return math.ceil((number - offset) / increment) * increment + offset


def read_class_id(f: BinaryIO, legacy: bool) -> int:
    """Port of Extensions.ReadClassID (AC2/Brotherhood/Revelations/AC1 use a
    32-bit class id, every other supported game uses 64-bit)."""
    if legacy:
        return int.from_bytes(f.read(4), "little")
    return int.from_bytes(f.read(8), "little")


def write_class_id(f: BinaryIO, value: int, legacy: bool) -> None:
    """Port of Extensions.WriteClassID."""
    if legacy:
        f.write((value & 0xFFFFFFFF).to_bytes(4, "little"))
    else:
        f.write((value & MASK64).to_bytes(8, "little"))


def read_string32(f: BinaryIO) -> str:
    """Port of StringHelper.ReadString32: an int32 byte-length prefix followed
    by that many UTF-8 bytes."""
    length = int.from_bytes(f.read(4), "little", signed=True)
    return f.read(length).decode("utf-8", errors="replace")


def write_string32(f: BinaryIO, s: str) -> None:
    """Port of StringHelper.WriteString32. The original writes .NET's
    string.Length (a UTF-16 code unit count) as the prefix but then encodes
    the payload with the writer's default (UTF-8) encoding -- for any
    non-ASCII text the prefix and the actual encoded byte count silently
    diverge in the original tool too. Reproduced as-is (bug-for-bug) since
    real asset names are ASCII in practice and this is what ReadString32
    on the other side actually expects."""
    encoded = s.encode("utf-8")
    f.write(len(s).to_bytes(4, "little", signed=True))
    f.write(encoded)


def get_after_or_empty(text: str, sep: str = "-") -> str:
    """Port of StringHelper.GetAfterOrEmpty."""
    idx = text.find(sep)
    if idx > 0:
        return text[idx + len(sep):]
    return ""


def leading_index(filename: str, sep: str = "_-_") -> int:
    """Port of StringHelper.GetUntilOrEmptyInt, used to recover a loose
    file's original position from its `{index}{sep}{name}.ext` filename."""
    idx = filename.find(sep)
    if idx <= 0:
        return 0
    prefix = filename[:idx]
    return int(prefix) if prefix.isdigit() else 0


def fnv64(text: str) -> int:
    """Port of FNV.FNV64(input, bFlipEndian=false).

    Note this is the FNV-1 (not FNV-1a) mixing order used by the original:
    hash = (hash * prime) ^ byte -- and it hashes UTF-16 code units of the
    .NET string directly, not UTF-8 bytes.
    """
    prime = 1099511628211
    h = 14695981039346656037
    for ch in text:
        h = ((h * prime) & MASK64) ^ ord(ch)
    return h
