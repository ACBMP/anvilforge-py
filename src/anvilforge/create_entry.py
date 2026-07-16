"""Port of AnvilToolkit.FileTypes.AnvilNext.Containers.DataFile.CreateEntry.

Recovers a loose file's ID/Extension by parsing its raw payload directly,
the same way the original tool does when repacking a folder from scratch
(as opposed to a manifest carried over from unpacking).

Only the code paths relevant to Brotherhood/Revelations/AC3/BlackFlag are
implemented. The original's per-game preamble switch used a `version - 7`
enum-arithmetic trick to route Unity/Syndicate/Origins/Odyssey/Steep/Ghost
Recon Breakpoint to wider (32-bit) block-count/list-length fields; all four
games this rewrite targets fall through that switch's default (16-bit)
branch, which is what's implemented below.
"""
from __future__ import annotations

import io
import os
from typing import BinaryIO

from .binio import fnv64, read_class_id
from .block import Block
from .entry import ForgeEntry
from .games import Game, is_legacy


def _u32(f: BinaryIO) -> int:
    return int.from_bytes(f.read(4), "little")


def _i32(f: BinaryIO) -> int:
    return int.from_bytes(f.read(4), "little", signed=True)


def _u16(f: BinaryIO) -> int:
    return int.from_bytes(f.read(2), "little")


def _u8(f: BinaryIO) -> int:
    return f.read(1)[0]


def _metadata_only_entry(entry: ForgeEntry, size: int, id_: int) -> ForgeEntry:
    entry.id = id_
    entry.length_on_disk = size
    entry.umac_hash = fnv64(entry.name)
    entry.scc_status_data = 2
    entry.scc_status_attributes = 2
    entry.is_hidden = 1
    return entry


def create_entry(file_path: str, timestamp: int, game: Game) -> ForgeEntry | None:
    if not os.path.exists(file_path):
        return None

    entry = ForgeEntry(name=file_path, timestamp=timestamp)
    size = os.path.getsize(file_path)
    ext = os.path.splitext(file_path)[1].lower()

    if size == 0:
        entry.is_hidden = 1
        return entry

    if ext == ".prefetchinfo":
        return _metadata_only_entry(entry, size, 145)
    if ext == ".metafile":
        return _metadata_only_entry(entry, size, 16)

    legacy = is_legacy(game)

    # A handful of entries in real retail data are degenerate/placeholder
    # payloads that merely look like a resource header (observed in real ACB
    # multiplayer content) -- block_count/num14 end up 0, which sends the
    # original's own algorithm down a "decompress 0 bytes into N bytes"
    # dead end. The original tool's CreateEntry has an outer try/catch that
    # turns any such failure into a `null` result (skip this entry); mirror
    # that here instead of letting a malformed block4 propagate.
    try:
        with open(file_path, "rb") as f:
            if game in (Game.BLACK_FLAG, Game.AC3):
                num1 = _u32(f)
                if num1 == 1:
                    return _metadata_only_entry(entry, size, 16)
                f.seek(num1, 1)
            elif game in (Game.BROTHERHOOD, Game.REVELATIONS):
                num2 = _u32(f)
                f.seek(num2 * 8, 1)
            else:
                raise NotImplementedError(f"create_entry() does not support {game!r}")

            f.read(8)  # unused
            f.read(2)  # unused
            f.read(1)  # unused
            f.read(2)  # unused
            f.read(2)  # unused

            block_count = _u16(f)
            blocks = [Block.read_header(f) for _ in range(block_count)]
            for block in blocks:
                f.seek(block.compressed_size + 4, 1)

            f.read(8)  # unused
            f.read(2)  # unused
            comp_algo = _u8(f)
            f.read(2)  # unused
            f.read(2)  # unused
            num14 = _u16(f)

            block1 = Block.read_header(f)
            f.seek((num14 - 1) * 4, 1)

            block1.read_data(comp_algo, f)

            sub = io.BytesIO(block1.data)
            sub.seek(8, 1)
            num15 = _i32(sub)
            sub.seek(num15, 1)
            num16 = _u8(sub)
            sub.seek(-1, 1)
            num17 = 0
            if num16 == 1:
                sub.seek(4, 1)
                n = _i32(sub)
                num17 = 12 * n + 7
            sub.seek(num17 + 14 + num15, 0)
            entry.id = read_class_id(sub, legacy)
            entry.extension = _u32(sub)
    except (OSError, ValueError, RuntimeError, IndexError):
        return None

    entry.length_on_disk = size
    entry.umac_hash = fnv64(file_path)
    entry.scc_status_data = 4
    entry.scc_status_attributes = 4
    entry.is_hidden = 0
    return entry
