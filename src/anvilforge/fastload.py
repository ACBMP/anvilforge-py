"""Lossless codec for AnvilNext "FastLoad" object payloads (the loose sub-parts
of a .data entry), following the engine's own FastLoadSerializer rules.

objectxml.py's decoder was reverse engineered from sample bytes alone and
stops early on most complex types (0 of ~3000 Entities decode fully). The
rules here were instead taken from the Mac build of ACBMP (full symbols) --
FastLoadSerializer::SerializePropertyGeneric, ::SerializeObjectPropertyInternal
(Object** and ManagedObject** overloads), ::SerializeObject,
::SerializeProperty(Reference), ::SerializeProperty(SimpleStringTemplate),
::SerializeDynamicProperties and the generated per-class ::FastLoad routines
(Entity, EntityGroup, Visual, InertComponent, CrowdDutyRegion, FXCommand, ...)
-- and validated by decoding ~20k retail objects from ACB and ACR multiplayer
maps to exactly their byte length. The rules:

- A property is on disk iff ``flags & 0x2000000``. BOOLs are always one byte
  (``elem_kind`` does NOT mean "absent" -- objectxml.py's rule is wrong).
- Root object: optional import-table pre-header, then ``status(1) flag(1)
  id(4) type_hash(4)`` and its properties.
- Pointer (OBJECT_PTR / BASE_OBJECT_PTR): status byte. 3 = null;
  1/2/5 = link, followed by a 4-byte object id; 0/4 = inline object:
  ``[flag(1) if the *declared* class derives from ManagedObject] id(4)
  type_hash(4) props``.
- Embedded OBJECT: ``[flag(1) if ManagedObject-derived] id(4) hash(4) props``.
  Embedded BASE_OBJECT: ``id(4) hash(4) props``.
- REFERENCE: ``tag(1) extra(1) id(4)``; tag 0 means the referenced object is
  inlined right after: ``hash(4) props`` (EntityGroup.Entities does this).
- HANDLE: ``tag(1) id(4)``.
- STRING/LSTRING: ``u32 len``; if len > 0, ``len + 1`` code units (NUL incl.).
- SMALL/BIG_ARRAY: ``u32 count`` + elements; STATIC_ARRAY: fixed count.
- Classes with DynamicProperties (FXCommand, Material, BuildRow, ...) append,
  after their own properties: ``u32 count`` then per entry ``name(4)
  descriptor(8) value`` (value encoded by the generic per-kind rules, where an
  OBJECT_PTR is a bare 4-byte id).

- A few classes append a hand-written tail (their CustomSerializeFastLoad) after
  their properties; CUSTOM_TAILS reads and writes those (NavMeshTriangle,
  WayPoint, ObjectWayPoint: the navmesh), kept in ``Obj.custom``.

Anything else (compiled animation tracks, FX code tables, compiled shader permutations, and a few cinematic components --
Scene clips, SceneSpawner, CameraRuleBook, MultiInertComponent -- seen only in
the Whiteroom lobby map) uses hand-written serializers these rules don't cover;
decode() raises DecodeError for those and callers should treat the payload as
opaque. Every Entity/EntityGroup/CrowdDutyRegion in the gameplay maps checked
(ACB MtStMichel, Alhambra; ACR Dyers) decodes.

The tree keeps every byte that isn't derivable from the schema (status/flag
bytes, ids, raw primitive bytes), so ``encode(decode(b, S), S) == b``. Encoding
against a *different* schema re-lays fields by name: fields the target schema
doesn't have are dropped, fields it has that the source lacks must be supplied
(see ``Obj.fields``) -- this is what lets a Revelations object be rewritten
into Brotherhood's layout.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

from .schema import Kind, PRIMITIVE_SIZE, Schema

SERIALIZED = 0x2000000

# Classes whose FastLoad appends a DynamicProperties block after their own
# properties (callers of FastLoadSerializer::SerializeDynamicProperties).
DYNAMIC_PROPERTY_CLASSES = frozenset({
    "FXCommand", "Material", "BuildColumn", "BuildRow", "FXProperty", "FXDefaultTable",
    "FXDefaultTableInstance", "FXConstantTable", "FXConstantTableInstance", "GenericObject",
    "PropertyControllerEntry", "DynamicPropertiesSet",
})


def _u16s(b: bytes) -> list[int]:
    return [int.from_bytes(b[i:i + 2], "little") for i in range(0, len(b), 2)]


def _pack_u16s(v) -> bytes:
    return b"".join(int(x).to_bytes(2, "little") for x in v)


def _tail_navmesh_triangle(read):
    """NavMeshTriangle::CustomSerializeFastLoad (Mac 0x0062c450): four u8 counts, then that many u16 in total."""
    counts = read(4)
    return {"counts": list(counts), "data": _u16s(read(2 * sum(counts)))}


def _enc_navmesh_triangle(c) -> bytes:
    return bytes(c["counts"]) + _pack_u16s(c["data"])


WAYPOINT_WORDS = ("Flags", "Bits", "TriangleIndex", "X", "Y", "Z", "ManagerIndex")


def _tail_waypoint(read):
    """WayPoint::CustomSerializeFastLoad (Mac 0x0085e3d0): u8 link count, links as (u16, u16), then seven u16:
    flags (low 4 bits), bits (12 bits), triangle index, x and y (10-bit steps across the manager's cell), z (half
    float), manager index."""
    n = read(1)[0]
    links = _u16s(read(4 * n))
    out = {"Links": [tuple(links[i:i + 2]) for i in range(0, len(links), 2)]}
    out.update(zip(WAYPOINT_WORDS, _u16s(read(14))))
    return out


def _enc_waypoint(c) -> bytes:
    return (bytes([len(c["Links"])]) + _pack_u16s([x for l in c["Links"] for x in l])
            + _pack_u16s(c[k] for k in WAYPOINT_WORDS))


def _tail_links(read):
    """ObjectWayPoint::CustomSerializeFastLoad (Mac 0x0062d890): u8 count, then that many (u16, u16) links."""
    links = _u16s(read(4 * read(1)[0]))
    return [tuple(links[i:i + 2]) for i in range(0, len(links), 2)]


def _enc_links(c) -> bytes:
    return bytes([len(c)]) + _pack_u16s([x for l in c for x in l])


# class name -> (read tail via read(n) -> value, encode value -> bytes)
CUSTOM_TAILS = {
    "NavMeshTriangle": (_tail_navmesh_triangle, _enc_navmesh_triangle),
    "WayPoint": (_tail_waypoint, _enc_waypoint),
    "ObjectWayPoint": (_tail_links, _enc_links),
}


class DecodeError(Exception):
    def __init__(self, msg: str, pos: int = -1, path: str = "") -> None:
        super().__init__(f"{msg} at {pos} ({path})")
        self.pos = pos
        self.path = path


@dataclass
class Obj:
    """One serialized object. ``fields`` maps property name -> value, in the
    source schema's order; ``dyn`` holds a DynamicProperties block if any."""

    type_hash: int
    id: bytes
    fields: dict[str, Any] = field(default_factory=dict)
    flag: int | None = None  # ManagedObject flag byte, when present
    dyn: list[tuple[int, int, int, Any]] | None = None  # (name, obj_hash, type_bits, value)
    custom: Any = None  # hand-written tail (CUSTOM_TAILS), when the class has one


@dataclass
class Ptr:
    status: int
    link: bytes | None = None  # 4-byte id for link statuses
    obj: Obj | None = None


@dataclass
class Ref:
    tag: int
    extra: int
    id: bytes
    obj: Obj | None = None  # inline object when tag == 0


@dataclass
class Handle:
    tag: int
    id: bytes


@dataclass
class Root:
    pre_header: bytes
    status: int
    obj: Obj


class Codec:
    def __init__(self, schema: Schema) -> None:
        self.S = schema
        self._levels: dict[int, list] = {}
        self._managed: dict[int, bool] = {}

    # ------------------------------------------------------------- schema --

    def levels(self, h: int) -> list[tuple[str, list]]:
        """[(class_name, serialized props)] base-class first."""
        lv = self._levels.get(h)
        if lv is None:
            cls = []
            t = self.S.type_by_hash(h)
            while t is not None:
                cls.append(t)
                t = self.S.type_by_hash(t.base_type_hash)
            lv = [(self.S.name_of(t.type_hash), [p for p in t.properties if p.flags & SERIALIZED])
                  for t in reversed(cls)]
            self._levels[h] = lv
        return lv

    def managed(self, h: int) -> bool:
        m = self._managed.get(h)
        if m is None:
            m = False
            t = self.S.type_by_hash(h)
            while t is not None:
                if self.S.name_of(t.type_hash) == "ManagedObject":
                    m = True
                    break
                t = self.S.type_by_hash(t.base_type_hash)
            self._managed[h] = m
        return m

    # ------------------------------------------------------------- decode --

    def decode(self, data: bytes) -> Root:
        f = io.BytesIO(data)
        pre = 0
        if data and data[0] == 1 and len(data) >= 8:
            pre = 12 * int.from_bytes(data[4:8], "little", signed=True) + 7
        pre_header = self._read(f, pre, "hdr")
        status, flag = self._read(f, 2, "hdr")
        oid = self._read(f, 4, "hdr")
        h = self._u32(f, "hdr")
        obj = self._props(f, h, oid, self.S.name_of(h))
        obj.flag = flag
        if f.tell() != len(data):
            raise DecodeError(f"{len(data) - f.tell()} trailing bytes", f.tell(), self.S.name_of(h))
        return Root(pre_header, status, obj)

    def _read(self, f: io.BytesIO, n: int, path: str) -> bytes:
        b = f.read(n)
        if len(b) < n:
            raise DecodeError(f"eof reading {n}", f.tell(), path)
        return b

    def _u32(self, f: io.BytesIO, path: str) -> int:
        return int.from_bytes(self._read(f, 4, path), "little")

    def _props(self, f: io.BytesIO, h: int, oid: bytes, path: str) -> Obj:
        if self.S.type_by_hash(h) is None:
            raise DecodeError(f"unknown type {h:#x}", f.tell(), path)
        obj = Obj(h, oid)
        for cname, props in self.levels(h):
            for p in props:
                nm = self.S.name_of(p.name_hash)
                obj.fields[nm] = self._value(f, p.kind, p.elem_kind, p.static_count, p.object_hash,
                                             f"{path}.{nm}")
            if cname in DYNAMIC_PROPERTY_CLASSES:
                obj.dyn = self._dyn(f, f"{path}._dyn")
            if cname in CUSTOM_TAILS:
                obj.custom = CUSTOM_TAILS[cname][0](lambda n: self._read(f, n, f"{path}._custom"))
        return obj

    def _inline(self, f: io.BytesIO, declared: int, path: str, with_flag: bool) -> Obj:
        flag = self._read(f, 1, path)[0] if with_flag else None
        oid = self._read(f, 4, path)
        h = self._u32(f, path)
        obj = self._props(f, h, oid, f"{path}<{self.S.name_of(h)}>")
        obj.flag = flag
        return obj

    def _value(self, f, kind, elem, count, oh, path, generic=False):
        if kind == Kind.BOOL or kind in PRIMITIVE_SIZE:
            return self._read(f, PRIMITIVE_SIZE.get(kind, 1), path)
        if kind == Kind.ENUM:
            return self._read(f, 4, path)
        if kind == Kind.OBJECT_ID:
            return self._read(f, 4, path)
        if kind == Kind.HANDLE:
            t = self._read(f, 1, path)[0]
            return Handle(t, self._read(f, 4, path))
        if kind == Kind.REFERENCE:
            tag, extra = self._read(f, 2, path)
            rid = self._read(f, 4, path)
            obj = None
            if tag == 0:
                h = self._u32(f, path)
                obj = self._props(f, h, rid, f"{path}<{self.S.name_of(h)}>")
            return Ref(tag, extra, rid, obj)
        if kind in (Kind.STRING, Kind.LSTRING):
            n = self._u32(f, path)
            if n > 10_000_000:
                raise DecodeError(f"bad string length {n}", f.tell(), path)
            w = 2 if kind == Kind.LSTRING else 1
            return self._read(f, (n + 1) * w, path)[:-w] if n else b""
        if kind == Kind.OBJECT:
            return self._inline(f, oh, path, self.managed(oh))
        if kind == Kind.BASE_OBJECT:
            return self._inline(f, oh, path, False)
        if kind in (Kind.OBJECT_PTR, Kind.BASE_OBJECT_PTR):
            if generic and kind == Kind.OBJECT_PTR:
                return Ptr(-1, self._read(f, 4, path))
            t = self._read(f, 1, path)[0]
            if t == 3:
                return Ptr(3)
            if t in (1, 2, 5):
                return Ptr(t, self._read(f, 4, path))
            if t in (0, 4):
                return Ptr(t, obj=self._inline(f, oh, path, self.managed(oh)))
            raise DecodeError(f"unknown pointer status {t}", f.tell() - 1, path)
        if kind in (Kind.STATIC_ARRAY, Kind.BIG_ARRAY, Kind.SMALL_ARRAY):
            n = count if kind == Kind.STATIC_ARRAY else self._u32(f, path)
            if n > 10_000_000:
                raise DecodeError(f"bad array count {n}", f.tell(), path)
            if elem == Kind.BOOL or elem in PRIMITIVE_SIZE or elem in (Kind.ENUM, Kind.OBJECT_ID):
                # fixed-size elements (vertex/index buffers etc.): one bulk read instead of n calls
                w = 4 if elem in (Kind.ENUM, Kind.OBJECT_ID) else PRIMITIVE_SIZE.get(elem, 1)
                b = self._read(f, n * w, path)
                return [b[i:i + w] for i in range(0, n * w, w)]
            return [self._value(f, elem, 0, 0, oh, f"{path}[{i}]") for i in range(n)]
        raise DecodeError(f"unhandled kind {kind}", f.tell(), path)

    def _dyn(self, f, path):
        n = self._u32(f, path)
        if n > 100_000:
            raise DecodeError(f"bad dynamic property count {n}", f.tell(), path)
        out = []
        for i in range(n):
            name = self._u32(f, path)
            oh = self._u32(f, path)
            bits = self._u32(f, path)
            kind = (bits >> 16) & 0x3F
            val = self._value(f, kind, (bits >> 23) & 0x1F, bits & 0xFFFF, oh, f"{path}[{i}]", generic=True)
            out.append((name, oh, bits, val))
        return out

    # ------------------------------------------------------------- encode --

    def encode(self, root: Root) -> bytes:
        out = io.BytesIO()
        out.write(root.pre_header)
        out.write(bytes([root.status, root.obj.flag if root.obj.flag is not None else 1]))
        out.write(root.obj.id)
        out.write(root.obj.type_hash.to_bytes(4, "little"))
        self._enc_props(out, root.obj, self.S.name_of(root.obj.type_hash))
        return out.getvalue()

    def _enc_props(self, out, obj: Obj, path: str) -> None:
        if self.S.type_by_hash(obj.type_hash) is None:
            raise ValueError(f"{path}: type {obj.type_hash:#x} not in target schema")
        for cname, props in self.levels(obj.type_hash):
            for p in props:
                nm = self.S.name_of(p.name_hash)
                if nm not in obj.fields:
                    raise KeyError(f"{path}.{nm}: missing field required by target schema")
                self._enc_value(out, p.kind, p.elem_kind, p.object_hash, obj.fields[nm], f"{path}.{nm}")
            if cname in DYNAMIC_PROPERTY_CLASSES:
                self._enc_dyn(out, obj.dyn or [], f"{path}._dyn")
            if cname in CUSTOM_TAILS:
                out.write(CUSTOM_TAILS[cname][1](obj.custom))

    def _enc_inline(self, out, obj: Obj, oh: int, with_flag: bool, path: str) -> None:
        if with_flag:
            out.write(bytes([obj.flag if obj.flag is not None else 1]))
        out.write(obj.id)
        out.write(obj.type_hash.to_bytes(4, "little"))
        self._enc_props(out, obj, f"{path}<{self.S.name_of(obj.type_hash)}>")

    def _enc_value(self, out, kind, elem, oh, v, path, generic=False):
        if kind == Kind.BOOL or kind in PRIMITIVE_SIZE or kind in (Kind.ENUM, Kind.OBJECT_ID):
            out.write(v)
        elif kind == Kind.HANDLE:
            out.write(bytes([v.tag])); out.write(v.id)
        elif kind == Kind.REFERENCE:
            out.write(bytes([v.tag, v.extra])); out.write(v.id)
            if v.tag == 0:
                out.write(v.obj.type_hash.to_bytes(4, "little"))
                self._enc_props(out, v.obj, f"{path}<{self.S.name_of(v.obj.type_hash)}>")
        elif kind in (Kind.STRING, Kind.LSTRING):
            w = 2 if kind == Kind.LSTRING else 1
            n = len(v) // w
            out.write(n.to_bytes(4, "little"))
            if n:
                out.write(v); out.write(b"\0" * w)
        elif kind == Kind.OBJECT:
            self._enc_inline(out, v, oh, self.managed(oh), path)
        elif kind == Kind.BASE_OBJECT:
            self._enc_inline(out, v, oh, False, path)
        elif kind in (Kind.OBJECT_PTR, Kind.BASE_OBJECT_PTR):
            if v.status == -1:
                out.write(v.link)
            else:
                out.write(bytes([v.status]))
                if v.status in (1, 2, 5):
                    out.write(v.link)
                elif v.status in (0, 4):
                    self._enc_inline(out, v.obj, oh, self.managed(oh), path)
        elif kind in (Kind.STATIC_ARRAY, Kind.BIG_ARRAY, Kind.SMALL_ARRAY):
            if kind != Kind.STATIC_ARRAY:
                out.write(len(v).to_bytes(4, "little"))
            for i, item in enumerate(v):
                self._enc_value(out, elem, 0, oh, item, f"{path}[{i}]")
        else:
            raise ValueError(f"{path}: unhandled kind {kind}")

    def _enc_dyn(self, out, dyn, path):
        out.write(len(dyn).to_bytes(4, "little"))
        for name, oh, bits, val in dyn:
            out.write(name.to_bytes(4, "little"))
            out.write(oh.to_bytes(4, "little"))
            out.write(bits.to_bytes(4, "little"))
            self._enc_value(out, (bits >> 16) & 0x3F, (bits >> 23) & 0x1F, oh, val, path, generic=True)


def walk(obj: Obj):
    """Yields every Obj in the tree (depth first, including ``obj``)."""
    yield obj
    stack = list(obj.fields.values())
    if obj.dyn:
        stack += [d[3] for d in obj.dyn]
    while stack:
        v = stack.pop()
        if isinstance(v, Obj):
            yield from walk(v)
        elif isinstance(v, (Ptr, Ref)) and v.obj is not None:
            yield from walk(v.obj)
        elif isinstance(v, list):
            stack.extend(v)
