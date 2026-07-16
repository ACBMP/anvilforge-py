"""Port of AnvilToolkit.FileTypes.AnvilNext.Containers.FileSet.

Handles one FileSet's worth of entries (up to games.ENTRIES_PER_FILESET) in
both directions:

- read_fileset(): mirrors the `FileSet(BinaryReader, ...)` constructor used
  during unpacking -- walks the offset/ID/length triplet table, cross-reads
  each entry's info record, and writes the raw payload out as a loose file.
- write_fileset_legacy() / write_fileset_modern(): mirror WriteToFile25 /
  WriteToFile27 used during repacking.

"Legacy" == format 25 (Brotherhood/Revelations): 32-bit IDs/class-ids,
188-byte info records, and a 440-byte "FILEDATA" header preceding each
entry's raw payload. "Modern" == format 27 (AC3/BlackFlag): 64-bit IDs,
192-byte info records, payload written with no per-entry header.
"""
from __future__ import annotations

import io
import os
from typing import BinaryIO

from .binio import read_cstring, write_cstring, read_class_id, write_class_id
from .entry import ForgeEntry, entry_size
from .games import ENTRIES_PER_FILESET
from .sanitize import sanitize_entry_name

LEGACY_DATA_HEADER_SIZE = 440
_LEGACY_DATA_BASE = 820_000
_MODERN_DATA_BASE = 880_000


def _u32(f: BinaryIO) -> int:
    return int.from_bytes(f.read(4), "little")


def _i32(f: BinaryIO) -> int:
    return int.from_bytes(f.read(4), "little", signed=True)


def _u64(f: BinaryIO) -> int:
    return int.from_bytes(f.read(8), "little")


def _i64(f: BinaryIO) -> int:
    return int.from_bytes(f.read(8), "little", signed=True)


def _wu32(f: BinaryIO, v: int) -> None:
    f.write((v & 0xFFFFFFFF).to_bytes(4, "little"))


def _wi32(f: BinaryIO, v: int) -> None:
    f.write(int(v).to_bytes(4, "little", signed=True))


def _wu64(f: BinaryIO, v: int) -> None:
    f.write((v & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little"))


def _wi64(f: BinaryIO, v: int) -> None:
    f.write(int(v).to_bytes(8, "little", signed=True))


def _name_suffix(name: str) -> str:
    """Port of Path.GetFileNameWithoutExtension(name).GetAfterOrEmpty("_-_")."""
    base = os.path.splitext(os.path.basename(name))[0]
    idx = base.find("_-_")
    if idx <= 0:
        return ""
    return base[idx + 3 :]


def _loose_file_name(global_index: int, entry: ForgeEntry) -> str:
    """Port of FileSet.WriteFileToDisk's naming rule."""
    ext = ".data"
    if entry.id == 16:
        ext = ".MetaFile"
    elif entry.id == 145 and "prefetchinginfo" in entry.name.lower():
        ext = ".PrefetchInfo"
    base = os.path.splitext(os.path.basename(entry.name))[0]
    return f"{global_index}_-_{base}{ext}"


def _read_entry_info(f: BinaryIO, entry: ForgeEntry, legacy: bool) -> None:
    entry.length_on_disk = _i32(f)
    entry.umac_hash = _u64(f)
    entry.engine_version = _i32(f)
    entry.extension = _u32(f)
    entry.revision_number_data = _i32(f)
    entry.revision_number_attributes = _i32(f)
    f.read(4)  # next-entry index, unused
    f.read(4)  # prev-entry index, unused
    entry.parent = _i32(f)
    entry.timestamp = _u32(f)
    raw_name = read_cstring(f)
    pad = 127 - len(raw_name)
    if pad > 0:
        f.read(pad)
    entry.scc_status_data = _i32(f)
    entry.metafile_key = read_class_id(f, legacy)
    entry.scc_status_attributes = _i32(f)
    entry.is_hidden = _i32(f)
    entry.name = sanitize_entry_name(raw_name)


def read_fileset(
    f: BinaryIO, set_index: int, out_dir: str, legacy: bool
) -> tuple[list[ForgeEntry], int]:
    """Reads one FileSet, writing each entry's raw payload into out_dir.

    Returns (entries, next_fileset_offset) where next_fileset_offset is -1
    when this was the last FileSet in the chain.
    """
    length = _u32(f)
    f.read(4)  # FileSetsCount (set 0 only) or 0, unused
    f.read(8)  # "position + 40" pointer field, unused
    next_fileset_ptr = _i64(f)
    f.read(4)  # start index, unused
    f.read(4)  # end index, unused
    info_table_start = _i64(f)
    f.read(8)  # data-region base value, unused

    esize = entry_size(legacy)
    entries: list[ForgeEntry] = []
    os.makedirs(out_dir, exist_ok=True)

    for index in range(length):
        entry = ForgeEntry()
        entry.offset = _i64(f)
        entry.id = _u32(f) if legacy else _u64(f)
        entry.length_on_disk = _i32(f)

        triplet_return_pos = f.tell()

        f.seek(info_table_start + index * esize)
        _read_entry_info(f, entry, legacy)

        entries.append(entry)

        f.seek(entry.offset + (LEGACY_DATA_HEADER_SIZE if legacy else 0))
        data = f.read(entry.length_on_disk)

        global_index = set_index * ENTRIES_PER_FILESET + index
        path = os.path.join(out_dir, _loose_file_name(global_index, entry))
        with open(path, "wb") as out:
            out.write(data)

        f.seek(triplet_return_pos)

    if next_fileset_ptr == -1:
        return entries, -1
    f.seek(next_fileset_ptr)
    return entries, next_fileset_ptr


def write_fileset_legacy(
    bw: BinaryIO, entries: list[ForgeEntry], set_index: int, filesets_count: int, offset: int
) -> int:
    """Port of FileSet.WriteToFile25. Mutates entry.offset to the final
    absolute value (matching the original, which does the same on the
    in-memory ForgeEntry objects)."""
    bw.seek(offset)
    position1 = bw.tell()
    num1 = set_index * ENTRIES_PER_FILESET
    n = len(entries)

    _wi32(bw, n)
    _wi32(bw, filesets_count if set_index == 0 else 0)
    _wi64(bw, bw.tell() + 40)
    _wi64(bw, -1)
    _wi32(bw, num1)
    _wi32(bw, num1 + n - 1)
    num2 = bw.tell() + n * 16 + 16
    num3 = num2 + n * 188
    _wi64(bw, num2)
    _wi64(bw, num3)
    num4 = num3 + _LEGACY_DATA_BASE
    position2 = bw.tell()

    for index, entry in enumerate(entries):
        entry.offset += num4

        bw.seek(position2 + index * 16)
        _wi64(bw, entry.offset)
        _wu32(bw, entry.id)
        _wi32(bw, entry.length_on_disk)

        bw.seek(num2 + index * 188)
        _wi32(bw, entry.length_on_disk)
        _wu64(bw, entry.umac_hash)
        _wi32(bw, entry.engine_version)
        _wu32(bw, entry.extension)
        _wi32(bw, entry.revision_number_data)
        _wi32(bw, entry.revision_number_attributes)
        if index == n and set_index + 1 == filesets_count:
            _wi32(bw, -1)
        else:
            _wi32(bw, set_index * ENTRIES_PER_FILESET + index + 1)
        _wi32(bw, set_index * ENTRIES_PER_FILESET + index - 1)
        _wi32(bw, entry.parent)
        _wu32(bw, entry.timestamp)
        after = _name_suffix(entry.name)
        write_cstring(bw, after)
        pad = 127 - len(after)
        if pad > 0:
            bw.seek(pad, 1)
        _wi32(bw, entry.scc_status_data)
        write_class_id(bw, entry.metafile_key, legacy=True)
        _wi32(bw, entry.scc_status_attributes)
        _wi32(bw, entry.is_hidden)

        bw.seek(entry.offset)
        write_cstring(bw, "FILEDATA" + after)
        bw.seek(382 - len(after), 1)
        _wu32(bw, entry.id)
        _wi32(bw, entry.length_on_disk)
        _wu64(bw, entry.umac_hash)
        bw.seek(8, 1)
        if set_index == 0 and index == 0:
            _wi32(bw, 1)
            _wi32(bw, 2)
            _wi32(bw, 2)
        else:
            _wi32(bw, 0)
            _wi32(bw, 4)
            _wi32(bw, 4)
        bw.seek(4, 1)
        _wu32(bw, entry.timestamp)
        bw.seek(1, 1)
        _wu32(bw, entry.extension)
        with open(entry.name, "rb") as src:
            bw.write(src.read())

    return position1


def write_fileset_modern(bw: BinaryIO, entries: list[ForgeEntry], set_index: int) -> int:
    """Port of FileSet.WriteToFile27."""
    position1 = bw.tell()
    num1 = set_index * ENTRIES_PER_FILESET
    n = len(entries)

    _wi32(bw, n)
    _wi32(bw, 2)
    _wi64(bw, bw.tell() + 40)
    _wi64(bw, -1)
    _wi32(bw, num1)
    _wi32(bw, num1 + n - 1)
    num2 = bw.tell() + n * 20 + 16
    num3 = num2 + n * 192
    _wi64(bw, num2)
    _wi64(bw, num3)
    num4 = num3 + _MODERN_DATA_BASE
    position2 = bw.tell()

    triplet_buf = io.BytesIO()
    info_buf = io.BytesIO()
    for index, entry in enumerate(entries):
        entry.offset += num4

        _wi64(triplet_buf, entry.offset)
        _wu64(triplet_buf, entry.id)
        _wi32(triplet_buf, entry.length_on_disk)

        _wi32(info_buf, entry.length_on_disk)
        _wu64(info_buf, entry.umac_hash)
        _wi32(info_buf, entry.engine_version)
        _wu32(info_buf, entry.extension)
        _wi32(info_buf, entry.revision_number_data)
        _wi32(info_buf, entry.revision_number_attributes)
        if index == n:
            _wi32(info_buf, -1)
        else:
            _wi32(info_buf, index + 1)
        _wi32(info_buf, index - 1)
        _wi32(info_buf, entry.parent)
        _wu32(info_buf, entry.timestamp)
        after = _name_suffix(entry.name)
        write_cstring(info_buf, after)
        pad = 127 - len(after)
        if pad > 0:
            info_buf.seek(pad, 1)
        _wi32(info_buf, entry.scc_status_data)
        write_class_id(info_buf, entry.metafile_key, legacy=False)
        _wi32(info_buf, entry.scc_status_attributes)
        _wi32(info_buf, entry.is_hidden)

        bw.seek(entry.offset)
        with open(entry.name, "rb") as src:
            bw.write(src.read())
        entry.offset -= num4

    bw.seek(position2)
    bw.write(triplet_buf.getvalue())
    bw.seek(num2)
    bw.write(info_buf.getvalue())

    return position1
