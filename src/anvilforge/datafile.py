"""Port of AnvilToolkit.FileTypes.AnvilNext.Containers.DataFile.

Every loose entry a ForgeFile unpacks (forge.py) is itself a small AnvilNext
"resource" container: a dependency table followed by one or more
magic-delimited, block-compressed sections. The first section is always a
Table of Contents (never consumed on unpack -- the original discards it too,
see below); any sections after that hold the resource's actual named
sub-parts (e.g. a mesh plus its LODs), each self-describing its own class ID
via the same "peek an import table, then read a class ID" idiom used by
create_entry.py.

Deliberate deviations from the original, all because the omitted behavior is
either unused by any reader (including the original's own) or is randomized
tool-signature junk that's regenerated fresh on every repack anyway:

- The true extension name (`GetHashedString`) is resolved via a ~5.5MB
  embedded, LZMA-compressed string table (`Resources/hashes.hl`) in the
  original tool. That table isn't ported here -- only its small (23-entry)
  fallback table is. Unresolved extension codes fall back to the numeric
  code as a string, matching the original's own final fallback. This is
  purely cosmetic (the loose sub-file's on-disk extension); repacking
  re-derives everything from content, never from a filename, so it doesn't
  affect round-trip fidelity.
- The Table of Contents section is written out (so the file is structurally
  valid and usable), but the original's own Deserialize never reads it back
  for anything -- it's skipped unconditionally. This port matches that.
- The random tool-signature footer (`WriteOffsetString` + timestamps) that
  Serialize appends after the content section is not reproduced. Nothing
  reads it back meaningfully either: the original's own Deserialize only
  ever recovers a partial, effectively-noise slice of it (a null-terminated
  scan starting 8 bytes past the content section, which usually just
  catches a few bytes of the non-string `ElapsedTime` field). Omitting it
  entirely still parses cleanly, since running out of bytes there is an
  expected, harmless loop-termination case.
"""
from __future__ import annotations

import io
import os
import zlib
from dataclasses import dataclass, field
from typing import BinaryIO

from . import compression
from .binio import (
    get_after_or_empty,
    leading_index,
    read_class_id,
    read_string32,
    write_string32,
)
from .games import Game, is_legacy
from .sanitize import sanitize_entry_name

DATA_MAGIC = 1154322941026740787

# (Version, Algorithm, BlockSize) -- port of DataStorage.DataVersions,
# restricted to the four games this rewrite targets.
DATA_VERSIONS: dict[Game, tuple[int, int, int]] = {
    Game.BLACK_FLAG: (1, 0, 32768),
    Game.AC3: (1, 0, 32768),
    Game.BROTHERHOOD: (1, 2, 32768),
    Game.REVELATIONS: (1, 2, 32768),
}

# Port of DataStorage.AnvilExtensions, HashedData.GetHashedString's fallback
# table (used when an extension code isn't found in the un-ported hashes.hl).
ANVIL_EXTENSIONS: dict[int, str] = {
    974816263: "AssassinAbilitySet",
    1501033885: "FightStrategyManager",
    1598516500: "MapIconLayer",
    4036801386: "NotorietyManager",
    153746181: "PatrolPath",
    1314581103: "UIPOIResourcesInfo",
    906511432: "AIStateOverrideSettings",
    1957128424: "Wrinkle Intensity",
    847111067: "Subsurface Intensity",
    403702418: "EYE_RIGHT",
    485040451: "EYE_LEFT",
    2735471887: "EYELID_LEFT_DOWN",
    4212489091: "EYELID_LEFT_UP",
    3754828124: "EYELID_RIGHT_DOWN",
    2764555285: "EYELID_RIGHT_UP",
    3124699560: "EYEBROW_LEFT_IN",
    1853759095: "EYEBROW_RIGHT_IN",
    4253370951: "JAW",
    3516996980: "P_SmallWeapon",
    2512990213: "P_BOW_TAG",
    2347475608: "P_CROSSBOW",
    332778049: "P_MUSKET",
    2858492965: "P_LEFT_HAND_PISTOL",
}


def resolve_extension(ext_code: int) -> str:
    """Port of HashedData.GetHashedString (minus the un-ported hashes.hl)."""
    return ANVIL_EXTENSIONS.get(ext_code, str(ext_code))


def _u8(f: BinaryIO) -> int:
    return f.read(1)[0]


def _u16(f: BinaryIO) -> int:
    return int.from_bytes(f.read(2), "little")


def _u32(f: BinaryIO) -> int:
    return int.from_bytes(f.read(4), "little")


def _u64(f: BinaryIO) -> int:
    return int.from_bytes(f.read(8), "little")


def _i32(f: BinaryIO) -> int:
    return int.from_bytes(f.read(4), "little", signed=True)


def _i16(v: int) -> bytes:
    return int(v).to_bytes(2, "little", signed=True)


# --------------------------------------------------------------- Dependency --


@dataclass
class Dependency:
    id: int = 0
    unk01: int = 0
    unk02: int = 0
    unk_shorts: list[int] = field(default_factory=list)


def _read_dependency(f: BinaryIO, extended: bool) -> Dependency:
    id_ = _u64(f)
    if not extended:
        return Dependency(id=id_)
    unk01 = _u8(f)
    n = _u8(f)
    unk02 = _u8(f)
    shorts = [int.from_bytes(f.read(2), "little", signed=True) for _ in range(n)]
    return Dependency(id_, unk01, unk02, shorts)


def _read_inline_dependencies(f: BinaryIO, game: Game) -> tuple[list[Dependency], bytes]:
    """Port of the dependency-table read at the top of DataFile.Deserialize."""
    extended = game in (Game.BLACK_FLAG, Game.AC3)
    if extended:
        num2 = _u32(f)
        num3 = _u16(f)
        num4 = _u16(f)
    else:
        num2 = num3 = 0
        num4 = _u32(f)
    if num3 == 0:
        return [_read_dependency(f, extended) for _ in range(num4)], b""
    f.seek(-4, 1)
    return [], f.read(num2)


def _write_inline_dependencies(
    f: BinaryIO, deps: list[Dependency], raw: bytes, game: Game
) -> None:
    """Port of the dependency-table write in DataFile.Serialize."""
    if game in (Game.BLACK_FLAG, Game.AC3):
        if raw:
            f.write(len(raw).to_bytes(4, "little"))
            f.write(raw)
            return
        f.write((len(deps) * 11 + 4).to_bytes(4, "little"))
        f.seek(2, 1)
        f.write(len(deps).to_bytes(2, "little", signed=True))
        for d in deps:
            f.write(d.id.to_bytes(8, "little"))
            f.write(bytes([d.unk01 & 0xFF, len(d.unk_shorts) & 0xFF, d.unk02 & 0xFF]))
            for s in d.unk_shorts:
                f.write(_i16(s))
    else:
        f.write(len(deps).to_bytes(4, "little"))
        for d in deps:
            f.write(d.id.to_bytes(8, "little"))


def _read_dependency_sidecar(path: str) -> tuple[list[Dependency], bytes]:
    """Port of the `*.Dependency` sidecar read at the top of DataFile.Serialize."""
    with open(path, "rb") as f:
        count = _u16(f)
        deps = [_read_dependency(f, extended=True) for _ in range(count)]
        length = _i32(f)
        raw = f.read(length)
        return deps, raw


def _write_dependency_sidecar(path: str, deps: list[Dependency], raw: bytes) -> None:
    """Port of the three `*.Dependency` sidecar-writing branches at the end of
    DataFile.Deserialize (MetaBytes is always empty here -- see module
    docstring)."""
    if not deps and not raw:
        return
    with open(path, "wb") as f:
        if deps:
            f.write(len(deps).to_bytes(2, "little"))
            for d in deps:
                f.write(d.id.to_bytes(8, "little"))
                f.write(bytes([d.unk01 & 0xFF, len(d.unk_shorts) & 0xFF, d.unk02 & 0xFF]))
                for s in d.unk_shorts:
                    f.write(_i16(s))
            f.write((0).to_bytes(4, "little"))
        else:
            f.write((0).to_bytes(2, "little"))
            f.write(len(raw).to_bytes(4, "little"))
            f.write(raw)


# ---------------------------------------------------------- block-set I/O --


def _read_block_set(f: BinaryIO) -> bytes:
    """Port of DataEntry's constructor: reads a block-list header, then
    decompresses and concatenates every block's payload."""
    f.read(2)  # dataVersion.Version, unused on read
    comp_algo = _u8(f)
    f.read(2)  # UncompressedBlockAlign, unused
    f.read(2)  # CompressedBlockAlign, unused
    block_count = _u16(f)
    if block_count == 0:
        raise ValueError("Blocks count = 0")
    headers = [(_u16(f), _u16(f)) for _ in range(block_count)]
    out = bytearray()
    for uncompressed_size, compressed_size in headers:
        f.read(4)  # block ID, unused on read (verified by CRC only in-game)
        cdata = f.read(compressed_size)
        if uncompressed_size == compressed_size:
            out += cdata
        else:
            out += compression.decompress(comp_algo, cdata, uncompressed_size)
    return bytes(out)


def _compress_blocks(
    data: bytes, algorithm: int, block_size: int
) -> list[tuple[int, int, int, bytes]]:
    blocks = []
    for i in range(0, len(data), block_size):
        chunk = data[i : i + block_size]
        compressed = compression.compress(algorithm, chunk)
        blocks.append((len(chunk), len(compressed), zlib.crc32(compressed), compressed))
    return blocks


def _write_block_set(
    f: BinaryIO, raw_data: bytes, version: int, algorithm: int, block_size: int
) -> None:
    """Port of the two Serialize block-writing branches (single tiny
    direct-stored block under 17 bytes, or a real compressed block list)."""
    if len(raw_data) < 17:
        f.write(version.to_bytes(2, "little", signed=True))
        f.write(bytes([algorithm]))
        f.write(block_size.to_bytes(2, "little"))
        f.write(block_size.to_bytes(2, "little"))
        f.write((1).to_bytes(2, "little", signed=True))
        f.write(len(raw_data).to_bytes(2, "little", signed=True))
        f.write(len(raw_data).to_bytes(2, "little", signed=True))
        f.write(zlib.crc32(raw_data).to_bytes(4, "little"))
        f.write(raw_data)
        return

    blocks = _compress_blocks(raw_data, algorithm, block_size)
    f.write(version.to_bytes(2, "little", signed=True))
    f.write(bytes([algorithm]))
    f.write(block_size.to_bytes(2, "little"))
    f.write(block_size.to_bytes(2, "little"))
    f.write(len(blocks).to_bytes(2, "little", signed=True))
    for uncompressed_size, compressed_size, _id, _data in blocks:
        f.write(uncompressed_size.to_bytes(2, "little"))
        f.write(compressed_size.to_bytes(2, "little"))
    for _unc, _comp, block_id, data in blocks:
        f.write(block_id.to_bytes(4, "little"))
        f.write(data)


# --------------------------------------------------- content record I/O --


def _extra_from_header(peek: bytes) -> int:
    """Port of the `num3==1` import-table-length peek used throughout this
    format (also in create_entry.py, at a different nesting depth)."""
    if len(peek) >= 8 and peek[0] == 1:
        n = int.from_bytes(peek[4:8], "little", signed=True)
        return 12 * n + 7
    return 0


def _derive_uid_and_ext(raw: bytes, legacy: bool) -> tuple[int, int]:
    """Port of the uID/extension recovery in DataFile.Serialize's per-loose-
    file read (the DataFile-internal analogue of create_entry.py's
    ID/Extension derivation, one nesting level shallower since these
    sub-parts aren't independently block-compressed)."""
    extra = _extra_from_header(raw[:8])
    buf = io.BytesIO(raw)
    buf.seek(extra + 2, 0)
    uid = read_class_id(buf, legacy)
    ext = _u32(buf)
    return uid, ext


def _read_content_records(data: bytes, start_index: int, out_dir: str) -> int:
    """Port of the per-DataEntry sub-file extraction loop in Deserialize."""
    buf = io.BytesIO(data)
    length = len(data)
    index = start_index
    while buf.tell() < length:
        ext_code = _u32(buf)
        size = _i32(buf) + 1
        name = read_string32(buf)
        name = sanitize_entry_name(name)
        peek_pos = buf.tell()
        size += _extra_from_header(data[peek_pos : peek_pos + 8])
        payload = buf.read(size)
        ext = resolve_extension(ext_code)
        fname = f"{index}_-_{name}.{ext}"
        with open(os.path.join(out_dir, fname), "wb") as out:
            out.write(payload)
        index += 1
    return index


def _build_content_and_toc(paths: list[str], game: Game) -> tuple[bytes, bytes]:
    """Port of DataFile.Serialize's loose-file scan (builds both the content
    blob and its Table of Contents in one pass, since both are derived from
    the same per-file uID/size)."""
    legacy = is_legacy(game)
    seen_ids: set[int] = set()
    content = bytearray()
    toc_entries: list[tuple[int, int]] = []

    for path in paths:
        with open(path, "rb") as fh:
            raw = fh.read()
        uid, ext_code = _derive_uid_and_ext(raw, legacy)
        if uid in seen_ids:
            continue
        seen_ids.add(uid)

        base = os.path.splitext(os.path.basename(path))[0]
        name = get_after_or_empty(base, "_-_")
        if name.lower() == "unnamed":
            name = ""

        extra = _extra_from_header(raw[:8])
        size_field = len(raw) - 1 - extra

        record_start = len(content)
        content += ext_code.to_bytes(4, "little")
        content += size_field.to_bytes(4, "little", signed=True)
        name_buf = io.BytesIO()
        write_string32(name_buf, name)
        content += name_buf.getvalue()
        content += raw
        toc_entries.append((uid, len(content) - record_start))

    return bytes(content), _build_toc(toc_entries, legacy)


def _build_toc(entries: list[tuple[int, int]], legacy: bool) -> bytes:
    out = bytearray()
    out += len(entries).to_bytes(2, "little")
    for uid, size in entries:
        if legacy:
            out += (uid & 0xFFFFFFFF).to_bytes(4, "little")
            out += size.to_bytes(4, "little")
            out += b"\xff\xff"
        else:
            out += (uid & ((1 << 64) - 1)).to_bytes(8, "little")
            out += size.to_bytes(4, "little")
            out += b"\x00\x00"
    if legacy:
        out += b"\x00\x00\x00\x00"
    return bytes(out)


# --------------------------------------------------------------- top level --


def unpack_datafile(data_path: str, out_dir: str, game: Game) -> None:
    """Unpacks one loose `.data` entry (already extracted from a .forge by
    forge.unpack) into its named sub-parts, plus a `.Dependency` sidecar if
    the file carries dependency info."""
    os.makedirs(out_dir, exist_ok=True)
    with open(data_path, "rb") as f:
        end = f.seek(0, 2)
        f.seek(0)
        deps, raw_dep = _read_inline_dependencies(f, game)

        index = 0
        first_block = True
        while end - f.tell() >= 8:
            magic = _u64(f)
            if magic != DATA_MAGIC:
                break
            blob = _read_block_set(f)
            if not first_block:
                index = _read_content_records(blob, index, out_dir)
            first_block = False

    base = os.path.splitext(os.path.basename(data_path))[0]
    _write_dependency_sidecar(os.path.join(out_dir, f"{base}.Dependency"), deps, raw_dep)


def repack_datafile(in_dir: str, data_path: str, game: Game) -> None:
    """Builds a `.data` file from a folder of its named sub-parts (as produced
    by unpack_datafile), deriving each sub-part's ID/extension from its own
    content, matching the original tool's from-a-folder repack."""
    version, algorithm, block_size = DATA_VERSIONS[game]

    deps: list[Dependency] = []
    raw_dep = b""
    names = []
    for name in os.listdir(in_dir):
        if os.path.splitext(name)[1].lower() == ".dependency":
            deps, raw_dep = _read_dependency_sidecar(os.path.join(in_dir, name))
        else:
            names.append(name)
    names.sort(key=leading_index)
    paths = [os.path.join(in_dir, n) for n in names]

    content, toc = _build_content_and_toc(paths, game)

    with open(data_path, "wb") as f:
        _write_inline_dependencies(f, deps, raw_dep, game)
        f.write(DATA_MAGIC.to_bytes(8, "little"))
        _write_block_set(f, toc, version, algorithm, block_size)
        f.write(DATA_MAGIC.to_bytes(8, "little"))
        _write_block_set(f, content, version, algorithm, block_size)
