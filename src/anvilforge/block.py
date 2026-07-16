"""Port of AnvilToolkit.FileTypes.AnvilNext.Containers.Block.

Only the 16-bit-size-field variant is implemented: the wider 32-bit variant
in the original is only used by Unity/Syndicate/Origins/Odyssey/Steep/Ghost
Recon Breakpoint, none of which this rewrite targets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import BinaryIO

from . import compression


@dataclass
class Block:
    uncompressed_size: int = 0
    compressed_size: int = 0
    id: int = 0
    data: bytes = field(default=b"")

    @classmethod
    def read_header(cls, f: BinaryIO) -> "Block":
        uncompressed_size = int.from_bytes(f.read(2), "little")
        compressed_size = int.from_bytes(f.read(2), "little")
        return cls(uncompressed_size=uncompressed_size, compressed_size=compressed_size)

    def read_data(self, comp_algo: int, f: BinaryIO) -> None:
        self.id = int.from_bytes(f.read(4), "little")
        cdata = f.read(self.compressed_size)
        if self.uncompressed_size == self.compressed_size:
            self.data = cdata
        else:
            self.data = compression.decompress(comp_algo, cdata, self.uncompressed_size)
