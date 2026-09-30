"""Parser for the SCHM reflection-schema files bundled under
anvilforge/schemas/ (one per game's multiplayer content: AC3_MP.schema,
AC4_MP.schema, ACB_MP.schema, ACR_MP.schema).

These describe the property layout of every AnvilNext engine object type
(Entity, Material, TextureMap, ...) so that the binary blobs unpacked from a
.data container (see datafile.py) can be decoded into editable XML and back
(objectxml.py).

Unlike forge.py/datafile.py, none of this is ported from AnvilToolkit's C#
source -- AnvilToolkit has no code for this format at all, since it only
implements a separate, hardcoded-per-class serialization system used for
single-player content (ScimitarClassRegistry). Everything here was instead
reverse engineered directly from the four .schema files and cross-validated
against real, retail Assassin's Creed Brotherhood multiplayer .data content:
header framing, primitive/vector/matrix sizes, array and object-reference
encodings, and the Property.packed_type bit layout below were all confirmed
by decoding real extracted objects (TextureSet, TextureMap, GridCellDataBlock,
LODSelector, RealTreeMesh, ...) and checking that decoding consumes the
sample's *entire* byte length with a nested type-hash cross-check passing --
see objectxml.py's module docstring for the full method and results.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from enum import IntEnum
from functools import lru_cache
from importlib import resources
from typing import BinaryIO

from .games import Game, SCHEMA_NAMES

SCHM_MAGIC = 0x4D484353  # b"SCHM"


class Kind(IntEnum):
    """Property.kind values (bits 48-52 of packed_type). Confirmed against a
    real compiled type table in AnvilToolkit's PropertyRegistry.cs for
    indices 0-22 and 25-28; indices 23-24 and 29-31 (the array kinds, plus
    SloppyHandle/DataDrivenEnum) have no equivalent in that hardcoded system
    and were positioned by elimination + empirical validation (StaticArray,
    BigArray, SmallArray all confirmed; SloppyHandle/DataDrivenEnum never
    observed in real ACB MP content, position unconfirmed)."""

    BOOL = 0
    CHAR = 1
    INT8 = 2
    UINT8 = 3
    INT16 = 4
    UINT16 = 5
    INT32 = 6
    UINT32 = 7
    INT64 = 8
    UINT64 = 9
    FLOAT = 10
    VECTOR2 = 11
    VECTOR3 = 12
    VECTOR4 = 13
    QUATERNION = 14
    MATRIX3X3 = 15
    MATRIX4X4 = 16
    OBJECT_ID = 17
    HANDLE = 18
    OBJECT = 19
    OBJECT_PTR = 20
    BASE_OBJECT_PTR = 21
    BASE_OBJECT = 22
    STATIC_ARRAY = 23
    BIG_ARRAY = 24
    ENUM = 25
    STRING = 26
    LSTRING = 27
    REFERENCE = 28
    SMALL_ARRAY = 29
    SLOPPY_HANDLE = 30
    DATA_DRIVEN_ENUM = 31


ARRAY_KINDS = frozenset({Kind.STATIC_ARRAY, Kind.BIG_ARRAY, Kind.SMALL_ARRAY})

# Fixed encoded size in bytes for the primitive/math kinds (everything not an
# array, reference, string, enum, or object/pointer kind, which need
# object-aware or length-prefixed reads -- see objectxml.py). Confirmed by
# byte-exact round trip against real samples for BOOL, FLOAT, MATRIX4X4;
# the rest follow directly from their component count/width and weren't
# independently exercised.
PRIMITIVE_SIZE: dict[Kind, int] = {
    Kind.BOOL: 1,
    Kind.CHAR: 1,
    Kind.INT8: 1,
    Kind.UINT8: 1,
    Kind.INT16: 2,
    Kind.UINT16: 2,
    Kind.INT32: 4,
    Kind.UINT32: 4,
    Kind.INT64: 8,
    Kind.UINT64: 8,
    Kind.FLOAT: 4,
    Kind.VECTOR2: 8,
    Kind.VECTOR3: 12,
    Kind.VECTOR4: 16,
    Kind.QUATERNION: 16,
    Kind.MATRIX3X3: 36,
    Kind.MATRIX4X4: 64,
}


@dataclass
class GameInfo:
    version: int
    game_name_hash: int
    code_cl: int
    parent_code_cl: int


@dataclass
class EnumValue:
    name_hash: int
    value: int


@dataclass
class EnumDef:
    name_hash: int
    values: list[EnumValue] = field(default_factory=list)


@dataclass
class Property:
    flags: int
    name_hash: int
    packed_type: int

    @property
    def kind(self) -> int:
        """Primary storage kind -- a Kind value, or an unrecognized 5-bit
        code for schema versions/types this port doesn't know about yet."""
        return (self.packed_type >> 48) & 0x1F

    @property
    def elem_kind(self) -> int:
        """For array kinds, the element type (also a Kind value). For a
        scalar BOOL property specifically, 1 means "not present in the byte
        stream at all" (confirmed empirically: TextureMap.Dynamic/
        MctCompressionEnabled/IgnoreSkipMips/DirtyFlag and ~30 of Entity's
        own bool flags all carry elem_kind==1 and consume zero bytes, while
        TextureSet.DirtyFlag carries elem_kind==0 and is always present as
        one byte). Meaning for any other kind is unconfirmed and unused by
        the decoder."""
        return (self.packed_type >> 55) & 0x1F

    @property
    def static_count(self) -> int:
        """Element count for a STATIC_ARRAY property (no length prefix in
        the byte stream -- the count is only known from here)."""
        return (self.packed_type >> 32) & 0xFFFF

    @property
    def object_hash(self) -> int:
        """Type-name hash of the referenced/element TypeDef, for kinds that
        point at another type (OBJECT/OBJECT_PTR/BASE_OBJECT/BASE_OBJECT_PTR,
        REFERENCE, array-of-those, ENUM)."""
        return self.packed_type & 0xFFFFFFFF


@dataclass
class TypeDef:
    type_hash: int
    base_type_hash: int
    flags: int
    properties: list[Property] = field(default_factory=list)
    enums: list[EnumDef] = field(default_factory=list)


def _u8(f: BinaryIO) -> int:
    return f.read(1)[0]


def _u32(f: BinaryIO) -> int:
    return int.from_bytes(f.read(4), "little")


def _i32(f: BinaryIO) -> int:
    return int.from_bytes(f.read(4), "little", signed=True)


def _u64(f: BinaryIO) -> int:
    return int.from_bytes(f.read(8), "little")


def _read_cstring(f: BinaryIO) -> str:
    buf = bytearray()
    while True:
        b = f.read(1)
        if not b or b == b"\x00":
            break
        buf += b
    return buf.decode("utf-8", errors="replace")


def _read_enum(f: BinaryIO) -> EnumDef:
    name_hash = _u32(f)
    count = _i32(f)
    values = [EnumValue(_u32(f), _i32(f)) for _ in range(count)]
    return EnumDef(name_hash, values)


def _read_type(f: BinaryIO) -> TypeDef:
    type_hash = _u32(f)
    base_type_hash = _u32(f)
    flags = _u32(f)
    prop_count = _i32(f)
    enum_count = _i32(f)
    properties = [Property(_u32(f), _u32(f), _u64(f)) for _ in range(prop_count)]
    enums = [_read_enum(f) for _ in range(enum_count)]
    return TypeDef(type_hash, base_type_hash, flags, properties, enums)


class Schema:
    def __init__(
        self,
        game_info: GameInfo,
        types: list[TypeDef],
        global_enums: list[EnumDef],
        names: dict[int, str],
    ) -> None:
        self.game_info = game_info
        self.types_by_hash: dict[int, TypeDef] = {t.type_hash: t for t in types}
        self.global_enums_by_hash: dict[int, EnumDef] = {e.name_hash: e for e in global_enums}
        self.names = names

    def name_of(self, name_hash: int) -> str:
        """Resolves a hash to its string via the schema's own dictionary,
        falling back to the numeric hash (as hex) when unresolved -- the
        same fallback convention as datafile.resolve_extension, and a much
        larger table (thousands of entries vs. its ~20-entry hardcoded
        fallback list)."""
        return self.names.get(name_hash, f"0x{name_hash:08X}")

    def type_by_hash(self, type_hash: int) -> TypeDef | None:
        return self.types_by_hash.get(type_hash)

    def property_chain(self, type_hash: int) -> list[Property]:
        """All properties for type_hash, base-class-first -- the order
        fields are actually laid out in the byte stream (confirmed: a
        BaseEntity-derived Entity object serializes BaseEntity's own
        properties before Entity's)."""
        chain: list[TypeDef] = []
        cur = self.types_by_hash.get(type_hash)
        while cur is not None:
            chain.append(cur)
            cur = self.types_by_hash.get(cur.base_type_hash)
        chain.reverse()
        props: list[Property] = []
        for t in chain:
            props.extend(t.properties)
        return props

    def enum_value_name(self, enum_hash: int, value: int) -> str | None:
        """Best-effort display name for an enum's raw integer value, checked
        against both the global enum table and every type's own local enum
        list (schema files carry both)."""
        e = self.global_enums_by_hash.get(enum_hash)
        if e is None:
            for t in self.types_by_hash.values():
                for local in t.enums:
                    if local.name_hash == enum_hash:
                        e = local
                        break
                if e is not None:
                    break
        if e is None:
            return None
        for v in e.values:
            if v.value == value:
                return self.name_of(v.name_hash)
        return None

    @classmethod
    def load(cls, path: str) -> "Schema":
        with open(path, "rb") as f:
            magic = _u32(f)
            if magic != SCHM_MAGIC:
                raise ValueError(f"not a SCHM schema file (bad magic in {path!r})")
            version = _i32(f)
            game_name_hash = _u32(f)
            code_cl = _u64(f)
            parent_code_cl = _u64(f)
            game_info = GameInfo(version, game_name_hash, code_cl, parent_code_cl)
            compression_algo = _u8(f)
            data_length = _i32(f)
            body = f.read(data_length - 33)

        if compression_algo != 0:
            raise NotImplementedError(
                f"compressed schema body (algorithm {compression_algo}) not supported; "
                "every real .schema file seen so far is stored uncompressed"
            )

        buf = io.BytesIO(body)
        n = len(body)

        type_count = _i32(buf)
        types = [_read_type(buf) for _ in range(type_count)]

        global_enum_count = _i32(buf) if buf.tell() + 4 <= n else 0
        global_enums = [_read_enum(buf) for _ in range(global_enum_count)]

        dict_count = _i32(buf) if buf.tell() + 4 <= n else 0
        names: dict[int, str] = {}
        for _ in range(dict_count):
            h = _u32(buf)
            names[h] = _read_cstring(buf)

        return cls(game_info, types, global_enums, names)

    def with_placeholders_filled(self, donor: "Schema") -> "Schema":
        """Returns a copy of this schema in which every type carrying a
        nameless placeholder property (name hash 0) is replaced by `donor`'s
        definition of that type, plus every donor-only type they may
        reference.

        ACB_MP.schema has such placeholders exactly where the shipped ACBMP
        binary serializes a field the schema dump lost (e.g.
        CrowdFraction.CrowdFractionName: ac2::CrowdFraction::FastLoad reads a
        string there). ACR_MP.schema names those fields with the same layout,
        so `ACB.with_placeholders_filled(ACR)` is the schema that actually
        matches retail ACB data -- verified by fastload.py round-tripping
        every object of the ACB MtStMichel/Alhambra maps."""
        s = Schema.__new__(Schema)
        s.game_info = self.game_info
        s.global_enums_by_hash = dict(self.global_enums_by_hash)
        s.names = dict(donor.names)
        s.names.update(self.names)
        s.types_by_hash = dict(self.types_by_hash)
        for h, t in self.types_by_hash.items():
            if any(p.name_hash == 0 for p in t.properties) and h in donor.types_by_hash:
                s.types_by_hash[h] = donor.types_by_hash[h]
        for h, t in donor.types_by_hash.items():
            s.types_by_hash.setdefault(h, t)
        return s

    @classmethod
    def load_default(cls, game: Game) -> "Schema":
        """Loads the .schema file bundled with this package for `game`,
        so callers no longer need to track down and pass a matching
        .schema file by hand."""
        return _load_bundled(SCHEMA_NAMES[game])


@lru_cache(maxsize=None)
def _load_bundled(name: str) -> Schema:
    ref = resources.files("anvilforge") / "schemas" / f"{name}.schema"
    with resources.as_file(ref) as path:
        return Schema.load(path)
