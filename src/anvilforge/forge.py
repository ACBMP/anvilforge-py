"""Port of AnvilToolkit.FileTypes.AnvilNext.Containers.ForgeFile.

Top-level unpack()/repack() entry points tying together the header
read/write (Deserialize25/27, Serialize25/27) and per-FileSet logic in
fileset.py.
"""
from __future__ import annotations

import os
import shutil
import time
from typing import BinaryIO

from .binio import leading_index, read_cstring, write_cstring, round_up
from .create_entry import create_entry
from .entry import ForgeEntry
from .fileset import (
    LEGACY_DATA_HEADER_SIZE,
    read_fileset,
    write_fileset_legacy,
    write_fileset_modern,
)
from .games import Game, ENTRIES_PER_FILESET, forge_version, is_legacy

LOOSE_EXTENSIONS = {".data", ".metafile", ".prefetchinfo"}

_LEGACY_ROUND_UP = 2048
_MODERN_ROUND_UP = 32768
# Spacing used when pre-assigning FileSet offsets for the legacy header,
# mirroring Serialize25's `1086 + index * 738299742`. Harmless for the
# single-FileSet case (<=5000 entries) this rewrite has been validated
# against; multi-FileSet forges are untested.
_LEGACY_FILESET_STRIDE = 738_299_742


def _write_i32(f: BinaryIO, v: int) -> None:
    f.write(int(v).to_bytes(4, "little", signed=True))


def _write_i64(f: BinaryIO, v: int) -> None:
    f.write(int(v).to_bytes(8, "little", signed=True))


def _write_u32(f: BinaryIO, v: int) -> None:
    f.write((v & 0xFFFFFFFF).to_bytes(4, "little"))


# ----------------------------------------------------------------- unpack --


def read_header(f: BinaryIO, expected_version: int) -> int:
    """Reads the common scimitar header, seeks to the first FileSet, and
    returns the FileSet count."""
    magic = read_cstring(f)
    if magic.lower() != "scimitar":
        raise ValueError(f"not a forge file (bad magic: {magic!r})")
    version = int.from_bytes(f.read(4), "little")
    if version != expected_version:
        raise ValueError(f"unexpected forge version {version}, expected {expected_version}")

    header_end = int.from_bytes(f.read(8), "little")
    f.read(header_end - 21)  # opaque header bytes, not preserved on repack

    f.read(4)  # unused
    f.read(4)  # unused
    f.read(8 if expected_version == 25 else 12)
    f.read(8)  # unused (packed pair of ints when this file was written)
    f.read(4)  # unused

    fileset_count = int.from_bytes(f.read(4), "little")
    first_offset = int.from_bytes(f.read(8), "little", signed=True)
    f.seek(first_offset)
    return fileset_count


def unpack(forge_path: str, out_dir: str, game: Game) -> list[ForgeEntry]:
    """Unpacks a .forge file into out_dir, one loose file per entry.

    Returns the list of ForgeEntry metadata in original order (informational
    only -- repack() re-derives everything from the loose files' content, it
    does not consume this list).
    """
    version = forge_version(game)
    legacy = is_legacy(game)

    entries: list[ForgeEntry] = []
    with open(forge_path, "rb") as f:
        fileset_count = read_header(f, version)
        for set_index in range(fileset_count):
            set_entries, _next_ptr = read_fileset(f, set_index, out_dir, legacy)
            entries.extend(set_entries)
    return entries


def backup(forge_path: str, backup_dir: str | None = None) -> str:
    """Copies forge_path into a Backups/ directory alongside it (or into
    backup_dir if given), skipping if a backup already exists."""
    directory = backup_dir or os.path.join(os.path.dirname(forge_path) or ".", "Backups")
    os.makedirs(directory, exist_ok=True)
    dest = os.path.join(directory, os.path.basename(forge_path))
    if not os.path.exists(dest):
        shutil.copy2(forge_path, dest)
    return dest


# ------------------------------------------------------------------ repack --


_PRESERVED_FIELDS = (
    "umac_hash",
    "extension",
    "parent",
    "revision_number_data",
    "revision_number_attributes",
    "metafile_key",
    "timestamp",
    "scc_status_data",
    "scc_status_attributes",
    "is_hidden",
)


def _scan_entries(
    in_dir: str, game: Game, legacy: bool, original_entries: list[ForgeEntry] | None = None
) -> list[ForgeEntry]:
    names = sorted(
        (n for n in os.listdir(in_dir) if os.path.splitext(n)[1].lower() in LOOSE_EXTENSIONS),
        key=leading_index,
    )
    timestamp = int(time.time())
    original_by_id = {e.id: e for e in original_entries} if original_entries else {}
    seen_ids: set[int] = set()
    entries: list[ForgeEntry] = []
    for name in names:
        entry = create_entry(os.path.join(in_dir, name), timestamp, game)
        if entry is None or entry.id in seen_ids:
            continue
        original = original_by_id.get(entry.id)
        if original is not None:
            # create_entry() re-derives these fresh every call -- umac_hash
            # in particular is fnv64(loose file's *own transient disk
            # path*), not anything content-stable, so it's effectively
            # randomized on every repack unless restored here. Preserving
            # them for entries that already existed is what makes a
            # touch-almost-nothing round trip (repack.py's per-forge
            # unpack->edit one sub-part->repack) safe against a real,
            # live install instead of silently corrupting every other
            # entry in the forge.
            for field in _PRESERVED_FIELDS:
                setattr(entry, field, getattr(original, field))
        if entries:
            prev = entries[-1]
            entry.offset = prev.offset + prev.length_on_disk
            if legacy:
                entry.offset += LEGACY_DATA_HEADER_SIZE
        entries.append(entry)
        seen_ids.add(entry.id)
    return entries


def _chunk(entries: list[ForgeEntry]) -> list[list[ForgeEntry]]:
    if not entries:
        return [[]]
    return [
        entries[i : i + ENTRIES_PER_FILESET] for i in range(0, len(entries), ENTRIES_PER_FILESET)
    ]


def _write_header_legacy(bw: BinaryIO, entries_count: int, fileset_count: int) -> None:
    """Port of ForgeFile.Serialize25's header (up to, but not including, the
    per-FileSet writes)."""
    write_cstring(bw, "scimitar")
    _write_u32(bw, 25)
    _write_i64(bw, 1046)
    _write_i32(bw, 16)
    _write_i32(bw, 1)
    bw.seek(1017, 1)
    _write_i32(bw, entries_count)
    _write_i32(bw, fileset_count)
    bw.seek(8, 1)
    _write_i32(bw, -1 if entries_count < ENTRIES_PER_FILESET else entries_count - 1)
    _write_i32(bw, -1)
    _write_i32(bw, entries_count)
    _write_i32(bw, fileset_count)
    _write_i64(bw, 1086)


def _write_header_modern(bw: BinaryIO, entries_count: int, fileset_count: int) -> None:
    """Port of ForgeFile.Serialize27's header."""
    write_cstring(bw, "scimitar")
    _write_u32(bw, 27)
    _write_i64(bw, 1050)
    _write_i64(bw, 16)
    _write_i32(bw, 1)
    bw.seek(1017, 1)
    _write_i32(bw, entries_count)
    _write_i32(bw, 2)
    bw.seek(12, 1)
    _write_i32(bw, -1 if entries_count < ENTRIES_PER_FILESET else entries_count - 1)
    _write_i32(bw, -1)
    _write_i32(bw, entries_count)
    _write_i32(bw, fileset_count)
    _write_i64(bw, 1094)


def repack(
    in_dir: str, forge_path: str, game: Game, original_entries: list[ForgeEntry] | None = None
) -> None:
    """Builds a .forge file from a folder of loose .data/.MetaFile/.PrefetchInfo
    files, deriving each entry's ID/extension from its own content (matching
    the original tool's from-a-folder repack, not a manifest).

    `original_entries` (typically whatever `unpack()` returned for this
    exact forge) restores several per-entry metadata fields create_entry()
    can't recover from a loose file alone (notably umac_hash, which is
    otherwise derived from the loose file's own transient disk path and so
    is effectively randomized on every repack -- see _scan_entries) for
    every entry that already existed. Without this, replace.py's per-forge
    round trip would silently reset that metadata for every entry in the
    forge, not just the one it meant to touch -- confirmed to cause real,
    in-game crashes even with zero content changes."""
    version = forge_version(game)
    legacy = is_legacy(game)

    entries = _scan_entries(in_dir, game, legacy, original_entries)
    filesets = _chunk(entries)

    with open(forge_path, "wb") as bw:
        if legacy:
            _write_header_legacy(bw, len(entries), len(filesets))
            offsets = [1086 + i * _LEGACY_FILESET_STRIDE for i in range(len(filesets))] + [-1]
            for i, fs in enumerate(filesets):
                write_fileset_legacy(bw, fs, i, len(filesets), offsets[i])
                bw.seek(1024, 1)
            round_to = _LEGACY_ROUND_UP
        else:
            _write_header_modern(bw, len(entries), len(filesets))
            offsets = [write_fileset_modern(bw, fs, i) for i, fs in enumerate(filesets)]
            offsets.append(-1)
            round_to = _MODERN_ROUND_UP

        for i in range(len(offsets) - 1):
            bw.seek(offsets[i] + 16)
            _write_i64(bw, offsets[i + 1])

        end = bw.seek(0, 2)
        bw.seek(round_up(end, round_to) - 1)
        bw.write(b"\x00")
