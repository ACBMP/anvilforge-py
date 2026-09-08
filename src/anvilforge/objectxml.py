"""Binary <-> XML conversion for individual schema-described objects (the
named sub-parts unpacked from a .data container by datafile.unpack_datafile
-- things like Entity, TextureMap, Material, GridCellDataBlock).

## Wire format, and how it was determined

AnvilToolkit's C# source has no code for this serialization system (see
schema.py's module docstring) -- everything below was reverse engineered by
decoding real files extracted from retail Assassin's Creed Brotherhood
multiplayer .forge archives (DataPC_AC2MP_Firenze.forge and its Cell*.data
sub-containers) and checking three things for every hypothesis: (1)
decoding consumes the sample's *entire* byte length with nothing left over
or negative, (2) any nested object's own type-hash cross-checks against the
schema's type table, and (3) the same hypothesis holds across every sample
of a given type, not just one. Specific results:

- Root/nested object header (`_read_header`/`_read_nested_header`): the
  fixed 2-byte marker `00 01` immediately before ID+Hash was found to be
  byte-for-byte identical across all 4508 real sub-entries sampled. The
  optional "import table" prefix before it (captured verbatim, never
  decoded -- see PreHeader below) occurred in 38 of those.
- TextureSet (3 properties: an 11-element StaticArray of References, a
  Handle, a bool): 210/210 real samples decoded byte-exact.
- LODSelector (7 properties incl. a 5-element StaticArray of *inline*
  BaseObject structs): 617/617 byte-exact, with every nested struct's own
  type hash cross-checked against LODDescriptor's real schema hash.
- GridCellDataBlock (a SmallArray of References): its one property's
  element count and 6-byte-per-element size were confirmed against a real
  cell's own 283-object listing (length prefix + 283 x 6-byte References
  landed exactly on EOF).
- ObjectPtr/BaseObjectPtr's tag byte (0=null, 1=link/ID-only, 4=inline) was
  found via RealTreeMesh.CompiledMesh (tag 4, nested header cross-checked
  against SubRealTreeMesh's real type hash) and confirmed as the dominant
  tag across ~1600 real Entity instances.
- The rule "a BOOL property with elem_kind==1 is not present in the byte
  stream at all" was found by contrasting TextureSet.DirtyFlag (elem_kind
  0, always present, part of the 210/210 exact match above) with
  TextureMap.DirtyFlag/Dynamic/MctCompressionEnabled/IgnoreSkipMips
  (elem_kind 1): treating them as absent makes 49/50 real TextureMap
  samples decode exactly, including PixelFormat cross-checked against the
  schema's own PixelFormat enum (2=DXT1, 4=DXT3, 5=DXT5) against the
  trailing payload's byte count for dozens of real textures spanning many
  resolutions and mip counts (a block-compressed mip chain is always
  4/3 x base-level-bytes for power-of-two dimensions, which matched).
- After every declared property is decoded, any bytes left before EOF are a
  real, intentional non-reflected payload (compiled mesh geometry, compiled
  texture pixels), not a decoding gap -- see `RawTail` below and
  `_texture_sidecar`.

Genuinely unconfirmed: SLOPPY_HANDLE/DATA_DRIVEN_ENUM kinds (never observed
in real ACB MP content), any ObjectPtr tag byte other than 0/1/4, and the
AC3/AC4 (8-byte ID) games specifically -- every sample validated above is
Assassin's Creed Brotherhood (4-byte "legacy" IDs; AC3/AC4 use the same
header/property scheme per games.is_legacy, but that hasn't been checked
against real AC3/AC4 content). Hitting any of these during decode is
reaching the edge of current understanding, not a bug: decoding stops at
that exact point and everything from there to EOF is preserved as an opaque
tail, so binary -> XML -> binary stays byte-exact even for object kinds or
types this port doesn't fully understand yet.

## XML shape

Every property becomes a `<Property Name="..." Kind="...">` child (not a
generic hash-keyed `<Value>` the way AnvilToolkit's hardcoded XmlUtils.cs
does it -- this schema has real names for everything, so there's no reason
to obscure them). Object references (OBJECT_PTR/BASE_OBJECT_PTR) carry a
`State` of "null", "link" (an external ID-only reference) or "inline" (a
nested object, recursively the same shape as the root `<Object>`). Arrays
wrap `<Item>` children. Anything past the point decoding stopped --
intentional trailing payload, or a construct this port doesn't recognize --
is preserved verbatim as base64 in a `<RawTail>` element (or externalized
to a sidecar file by `write_object_xml`, see below) so the round trip is
always exact regardless of how much of the object was actually understood.
"""
from __future__ import annotations

import base64
import io
import os
import struct
import xml.etree.ElementTree as ET

from .schema import ARRAY_KINDS, Kind, PRIMITIVE_SIZE, Schema

_HEADER_MARKER = b"\x00\x01"  # confirmed constant, see module docstring
_PTR_INLINE_PAD = b"\x00"  # confirmed constant (0/42 real samples nonzero)

_FLOAT_COUNT = {
    Kind.VECTOR2: 2,
    Kind.VECTOR3: 3,
    Kind.VECTOR4: 4,
    Kind.QUATERNION: 4,
    Kind.MATRIX3X3: 9,
    Kind.MATRIX4X4: 16,
}
_UNSIGNED_KINDS = {Kind.CHAR, Kind.UINT8, Kind.UINT16, Kind.UINT32, Kind.UINT64}
_SIGNED_KINDS = {Kind.INT8, Kind.INT16, Kind.INT32, Kind.INT64}


class DecodeError(Exception):
    """Raised to abandon decoding the current object at the current byte
    position. Deliberately left uncaught by every recursive decode helper
    except decode_root's own top-level loop: once any property's true
    encoded length is unknown, every byte after it -- including the rest of
    whatever array/object contains it -- is unrecoverable too, so the
    failure has to propagate all the way out before anything decides what
    to do with the remaining bytes (see decode_root)."""


def _kind_name(kind: int) -> str:
    try:
        return Kind(kind).name
    except ValueError:
        return f"Unknown{kind}"


def _kind_from_name(name: str) -> Kind:
    try:
        return Kind[name]
    except KeyError:
        raise ValueError(f"cannot encode unrecognized Kind {name!r}") from None


def _read_header(f: io.BytesIO, id_width: int) -> tuple[bytes, int, bytes]:
    """Reads one root object header: an optional, unparsed "import table"
    prefix (captured verbatim -- see PreHeader in write_object_xml), the
    fixed 2-byte marker, the object's ID, and its type-name hash. Mirrors
    datafile._extra_from_header/derive_uid_and_ext, which recover the same
    header for a different purpose (repack bookkeeping)."""
    start = f.tell()
    peek = f.read(8)
    if len(peek) < 8:
        raise DecodeError("truncated object header")
    extra = 0
    if peek[0] == 1:
        n = int.from_bytes(peek[4:8], "little", signed=True)
        extra = 12 * n + 7
    f.seek(start)
    pre_header = f.read(extra)
    marker = f.read(2)
    obj_id = f.read(id_width)
    hash_bytes = f.read(4)
    if len(marker) < 2 or len(obj_id) < id_width or len(hash_bytes) < 4:
        raise DecodeError("truncated object header")
    return pre_header, int.from_bytes(hash_bytes, "little"), obj_id


def _read_nested_header(f: io.BytesIO, id_width: int) -> tuple[int, bytes]:
    """Header for a nested object (OBJECT/BASE_OBJECT fields, and the inline
    case of OBJECT_PTR/BASE_OBJECT_PTR): just ID + type hash, no import-table
    prefix or marker -- those are root-object-only."""
    obj_id = f.read(id_width)
    hash_bytes = f.read(4)
    if len(obj_id) < id_width or len(hash_bytes) < 4:
        raise DecodeError("truncated nested object header")
    return int.from_bytes(hash_bytes, "little"), obj_id


class _Decoder:
    def __init__(self, schema: Schema, id_width: int) -> None:
        self.schema = schema
        self.id_width = id_width

    def decode_root(self, data: bytes) -> ET.Element:
        f = io.BytesIO(data)
        root = ET.Element("Object")
        try:
            pre_header, type_hash, obj_id = _read_header(f, self.id_width)
        except DecodeError:
            root.set("Type", "Unrecognized")
            self._attach_raw_tail(root, data, 0)
            return root

        root.set("Type", self.schema.name_of(type_hash))
        root.set("TypeHash", str(type_hash))
        root.set("ID", obj_id.hex())
        if pre_header:
            root.set("PreHeader", pre_header.hex())

        for prop in self.schema.property_chain(type_hash):
            pos = f.tell()
            try:
                el = self._decode_value(f, prop.kind, prop.elem_kind, prop.static_count, prop.object_hash)
            except DecodeError:
                # Seek back to before this property started: whatever bytes
                # it (or something nested inside it) already consumed before
                # failing are otherwise neither represented in the XML (the
                # property is dropped) nor covered by the RawTail attached
                # below (which only starts at the *current* position) --
                # silently losing them from the round trip. Only the
                # outermost loop needs this: a nested _decode_props call
                # propagating a failure discards its whole partial result
                # the same way, once its own containing property is dropped
                # here.
                f.seek(pos)
                break
            if el is not None:
                el.tag = "Property"
                el.set("Name", self.schema.name_of(prop.name_hash))
                if prop.kind == Kind.ENUM and el.text is not None:
                    enum_name = self.schema.enum_value_name(prop.object_hash, int(el.text))
                    if enum_name is not None:
                        el.set("EnumName", enum_name)
                root.append(el)
        self._attach_raw_tail(root, data, f.tell())
        return root

    def _decode_props(self, f: io.BytesIO, props: list, parent: ET.Element) -> None:
        """Decodes every property in order, appending each as a child
        <Property> of `parent`. Does NOT catch DecodeError -- see
        decode_root, which is the only place that needs to (and does)
        recover from a partial failure."""
        for prop in props:
            el = self._decode_value(f, prop.kind, prop.elem_kind, prop.static_count, prop.object_hash)
            if el is None:
                continue
            el.tag = "Property"
            el.set("Name", self.schema.name_of(prop.name_hash))
            if prop.kind == Kind.ENUM and el.text is not None:
                enum_name = self.schema.enum_value_name(prop.object_hash, int(el.text))
                if enum_name is not None:
                    el.set("EnumName", enum_name)
            parent.append(el)

    def _decode_value(
        self, f: io.BytesIO, kind: int, elem_kind: int, static_count: int, object_hash: int
    ) -> ET.Element | None:
        el = ET.Element("Value")
        el.set("Kind", _kind_name(kind))

        if kind == Kind.BOOL:
            if elem_kind == 1:
                return None  # not serialized -- see module docstring
            raw = f.read(1)
            if not raw:
                raise DecodeError("truncated bool")
            # Text carries the literal byte (retail data isn't always a
            # canonical 0/1 -- values like 223 or 87 show up on some real
            # ACB properties, presumably uninitialized memory at save time)
            # so re-encoding stays byte-exact; Bool is just for readability.
            el.text = str(raw[0])
            el.set("Bool", "true" if raw[0] else "false")
            return el

        if kind in PRIMITIVE_SIZE:
            size = PRIMITIVE_SIZE[kind]
            raw = f.read(size)
            if len(raw) < size:
                raise DecodeError(f"truncated {_kind_name(kind)}")
            if kind == Kind.FLOAT:
                el.text = repr(struct.unpack("<f", raw)[0])
                if struct.pack("<f", float(el.text)) != raw:
                    # repr()/float() isn't bit-exact for this value (seen on
                    # real retail data: uninitialized fields holding a NaN
                    # with a non-canonical payload). Hex is the ground
                    # truth on encode when present -- see _encode_value.
                    el.set("Hex", raw.hex())
            elif kind in _FLOAT_COUNT:
                count = _FLOAT_COUNT[kind]
                el.text = " ".join(repr(v) for v in struct.unpack(f"<{count}f", raw))
                if struct.pack(f"<{count}f", *(float(x) for x in el.text.split())) != raw:
                    el.set("Hex", raw.hex())
            elif kind in _SIGNED_KINDS:
                el.text = str(int.from_bytes(raw, "little", signed=True))
            else:
                el.text = str(int.from_bytes(raw, "little"))
            return el

        if kind == Kind.OBJECT_ID:
            raw = f.read(self.id_width)
            if len(raw) < self.id_width:
                raise DecodeError(f"truncated {_kind_name(kind)}")
            el.set("ID", raw.hex())
            return el

        if kind == Kind.HANDLE:
            # Confirmed against real content (TeamVIPNavflowPathNode.NavFlow,
            # a HANDLE targeting Entity): a 1-byte tag (always seen as 0)
            # precedes the id_width-byte ID -- without it, every HANDLE
            # property is off by one byte, corrupting itself and everything
            # after it. See schema.py's property table for this type: only
            # tag 0 has been observed, so any other value is treated as
            # unconfirmed territory rather than guessed at.
            tag = f.read(1)
            if not tag:
                raise DecodeError("truncated handle tag")
            if tag[0] != 0:
                raise DecodeError(f"unrecognized handle tag {tag[0]}")
            raw = f.read(self.id_width)
            if len(raw) < self.id_width:
                raise DecodeError("truncated HANDLE")
            el.set("Tag", str(tag[0]))
            el.set("ID", raw.hex())
            return el

        if kind in (Kind.OBJECT, Kind.BASE_OBJECT):
            type_hash, obj_id = _read_nested_header(f, self.id_width)
            el.set("Type", self.schema.name_of(type_hash))
            el.set("TypeHash", str(type_hash))
            el.set("ID", obj_id.hex())
            self._decode_props(f, self.schema.property_chain(type_hash), el)
            return el

        if kind in (Kind.OBJECT_PTR, Kind.BASE_OBJECT_PTR):
            tag = f.read(1)
            if not tag:
                raise DecodeError("truncated pointer tag")
            tag_val = tag[0]
            if tag_val == 0:
                el.set("State", "null")
                return el
            if tag_val == 1:
                raw = f.read(self.id_width)
                if len(raw) < self.id_width:
                    raise DecodeError("truncated link ID")
                el.set("State", "link")
                el.set("ID", raw.hex())
                return el
            if tag_val == 4:
                pad = f.read(1)
                if not pad:
                    raise DecodeError("truncated pointer padding")
                type_hash, obj_id = _read_nested_header(f, self.id_width)
                el.set("State", "inline")
                el.set("Type", self.schema.name_of(type_hash))
                el.set("TypeHash", str(type_hash))
                el.set("ID", obj_id.hex())
                self._decode_props(f, self.schema.property_chain(type_hash), el)
                return el
            raise DecodeError(f"unrecognized pointer tag {tag_val}")

        if kind in ARRAY_KINDS:
            if kind == Kind.STATIC_ARRAY:
                count = static_count
            else:
                raw = f.read(4)
                if len(raw) < 4:
                    raise DecodeError("truncated array length")
                count = int.from_bytes(raw, "little", signed=True)
                if not (0 <= count < 1_000_000):
                    raise DecodeError(f"implausible array length {count}")
            el.set("ElementKind", _kind_name(elem_kind))
            el.set("Count", str(count))
            for _ in range(count):
                item = self._decode_value(f, elem_kind, 0, 0, object_hash)
                if item is not None:
                    item.tag = "Item"
                    el.append(item)
            return el

        if kind == Kind.ENUM:
            raw = f.read(4)
            if len(raw) < 4:
                raise DecodeError("truncated enum")
            el.text = str(int.from_bytes(raw, "little", signed=True))
            return el

        if kind in (Kind.STRING, Kind.LSTRING):
            raw = f.read(4)
            if len(raw) < 4:
                raise DecodeError("truncated string length")
            length = int.from_bytes(raw, "little", signed=True)
            if not (0 <= length < 1_000_000):
                raise DecodeError(f"implausible string length {length}")
            width = 2 if kind == Kind.LSTRING else 1
            payload = f.read(length * width)
            if len(payload) < length * width:
                raise DecodeError("truncated string payload")
            el.text = payload.decode("utf-16-le" if kind == Kind.LSTRING else "utf-8", errors="replace")
            return el

        if kind == Kind.REFERENCE:
            header = f.read(2)
            raw_id = f.read(self.id_width)
            if len(header) < 2 or len(raw_id) < self.id_width:
                raise DecodeError("truncated reference")
            el.set("Tag", str(header[0]))
            el.set("IsGlobal", str(header[1]))
            el.set("ID", raw_id.hex())
            return el

        raise DecodeError(f"unhandled property kind {kind}")

    def _attach_raw_tail(self, parent: ET.Element, data: bytes, pos: int) -> None:
        tail = data[pos:]
        if not tail:
            return
        el = ET.SubElement(parent, "RawTail")
        el.set("Bytes", str(len(tail)))
        el.text = base64.b64encode(tail).decode("ascii")


def decode_object(data: bytes, schema: Schema, legacy: bool) -> ET.Element:
    """Decodes one schema object's raw bytes (a loose sub-part produced by
    datafile.unpack_datafile) into an XML tree. `legacy` selects the 4-byte
    (Brotherhood/Revelations) vs. 8-byte (AC3/BlackFlag) ID width, matching
    games.is_legacy."""
    return _Decoder(schema, 4 if legacy else 8).decode_root(data)


def _encode_props(buf: io.BytesIO, parent: ET.Element, id_width: int) -> None:
    for el in parent.findall("Property"):
        _encode_value(buf, el, id_width)


def _encode_value(buf: io.BytesIO, el: ET.Element, id_width: int) -> None:
    kind = _kind_from_name(el.get("Kind"))

    if kind == Kind.BOOL:
        buf.write(bytes([int(el.text) & 0xFF]))
        return

    if kind in PRIMITIVE_SIZE:
        hex_override = el.get("Hex")
        if hex_override is not None:
            buf.write(bytes.fromhex(hex_override))
        elif kind == Kind.FLOAT:
            buf.write(struct.pack("<f", float(el.text)))
        elif kind in _FLOAT_COUNT:
            values = [float(x) for x in el.text.split()]
            buf.write(struct.pack(f"<{len(values)}f", *values))
        elif kind in _SIGNED_KINDS:
            buf.write(int(el.text).to_bytes(PRIMITIVE_SIZE[kind], "little", signed=True))
        else:
            buf.write(int(el.text).to_bytes(PRIMITIVE_SIZE[kind], "little"))
        return

    if kind == Kind.OBJECT_ID:
        buf.write(bytes.fromhex(el.get("ID")))
        return

    if kind == Kind.HANDLE:
        buf.write(bytes([int(el.get("Tag"))]))
        buf.write(bytes.fromhex(el.get("ID")))
        return

    if kind in (Kind.OBJECT, Kind.BASE_OBJECT):
        buf.write(bytes.fromhex(el.get("ID")))
        buf.write(int(el.get("TypeHash")).to_bytes(4, "little"))
        _encode_props(buf, el, id_width)
        return

    if kind in (Kind.OBJECT_PTR, Kind.BASE_OBJECT_PTR):
        state = el.get("State")
        if state == "null":
            buf.write(b"\x00")
        elif state == "link":
            buf.write(b"\x01")
            buf.write(bytes.fromhex(el.get("ID")))
        elif state == "inline":
            buf.write(b"\x04")
            buf.write(_PTR_INLINE_PAD)
            buf.write(bytes.fromhex(el.get("ID")))
            buf.write(int(el.get("TypeHash")).to_bytes(4, "little"))
            _encode_props(buf, el, id_width)
        else:
            raise ValueError(f"unknown pointer State {state!r}")
        return

    if kind in ARRAY_KINDS:
        items = el.findall("Item")
        if kind != Kind.STATIC_ARRAY:
            buf.write(len(items).to_bytes(4, "little", signed=True))
        for item in items:
            _encode_value(buf, item, id_width)
        return

    if kind == Kind.ENUM:
        buf.write(int(el.text).to_bytes(4, "little", signed=True))
        return

    if kind in (Kind.STRING, Kind.LSTRING):
        text = el.text or ""
        if kind == Kind.LSTRING:
            payload = text.encode("utf-16-le")
            buf.write(len(text).to_bytes(4, "little", signed=True))
        else:
            payload = text.encode("utf-8")
            buf.write(len(payload).to_bytes(4, "little", signed=True))
        buf.write(payload)
        return

    if kind == Kind.REFERENCE:
        buf.write(bytes([int(el.get("Tag")), int(el.get("IsGlobal"))]))
        buf.write(bytes.fromhex(el.get("ID")))
        return

    raise ValueError(f"cannot encode Kind {el.get('Kind')!r}")


def encode_object(root: ET.Element, legacy: bool) -> bytes:
    """Inverse of decode_object. Purely structural -- every Kind/Type/State
    needed to reproduce the bytes is already carried in the XML itself, so
    (unlike decoding) this needs no schema, just the same `legacy` id-width
    selection used to decode it."""
    id_width = 4 if legacy else 8
    buf = io.BytesIO()
    pre_header = root.get("PreHeader")
    if pre_header:
        buf.write(bytes.fromhex(pre_header))
    buf.write(_HEADER_MARKER)
    buf.write(bytes.fromhex(root.get("ID")))
    buf.write(int(root.get("TypeHash")).to_bytes(4, "little"))
    _encode_props(buf, root, id_width)
    raw_tail = root.find("RawTail")
    if raw_tail is not None and raw_tail.text:
        buf.write(base64.b64decode(raw_tail.text))
    return buf.getvalue()


# ------------------------------------------------------- texture sidecars --

# TextureMap.PixelFormat values that map onto a real DXT/BC FourCC -- see
# the schema's own PixelFormat global enum. Formats without a confirmed
# fixed-function DDS mapping here (uncompressed RGBA8888, A8, I8, I16,
# A8I8, R32F, DXN) fall back to a plain .bin sidecar rather than guessing at
# channel masks nothing here has verified.
_DXT_FOURCC = {2: b"DXT1", 3: b"DXT1", 4: b"DXT3", 5: b"DXT5"}


def _texture_sidecar(obj_el: ET.Element, tail: bytes) -> tuple[str, bytes] | None:
    """If `obj_el` looks like a TextureMap-shaped object (has Width/Height/
    PixelFormat/NbMipMaps properties with a PixelFormat this port can map to
    a DDS FourCC), returns (".dds", full_dds_bytes) so the trailing compiled
    pixel payload can be written out as a directly viewable image file.
    Returns None for anything else, and the caller falls back to a generic
    .bin sidecar."""

    def prop_int(name: str) -> int | None:
        p = obj_el.find(f"Property[@Name='{name}']")
        if p is None or p.text is None:
            return None
        try:
            return int(p.text)
        except ValueError:
            return None

    width = prop_int("Width")
    height = prop_int("Height")
    pixfmt = prop_int("PixelFormat")
    mips = prop_int("NbMipMaps") or 1
    if width is None or height is None or pixfmt is None or not tail:
        return None
    fourcc = _DXT_FOURCC.get(pixfmt)
    if fourcc is None:
        return None

    flags = 0x1 | 0x2 | 0x4 | 0x1000 | 0x20000 | 0x80000  # CAPS|HEIGHT|WIDTH|PIXELFORMAT|MIPMAPCOUNT|LINEARSIZE
    block_size = 8 if fourcc == b"DXT1" else 16
    pitch = max(1, (width + 3) // 4) * max(1, (height + 3) // 4) * block_size
    caps = 0x1000 | (0x8 | 0x400000 if mips > 1 else 0)  # TEXTURE | (COMPLEX|MIPMAP if mipped)

    # Standard 128-byte DDS header: "DDS " magic + a 124-byte DDS_HEADER,
    # which itself embeds a 32-byte DDS_PIXELFORMAT (dwSize/dwFlags/
    # dwFourCC/dwRGBBitCount/four channel masks -- left zeroed here since
    # every format this port maps to a FourCC is block-compressed, so the
    # uncompressed-format mask fields don't apply).
    header = b"DDS " + struct.pack(
        "<7I44x2I4s5I5I",
        124, flags, height, width, pitch, 0, mips,  # DDS_HEADER core fields
        # 44x: dwReserved1[11]
        32, 0x4, fourcc, 0, 0, 0, 0, 0,  # DDS_PIXELFORMAT
        caps, 0, 0, 0, 0,  # dwCaps, dwCaps2-4, dwReserved2
    )
    return ".dds", header + tail


def _externalize_raw_tails(root: ET.Element, out_path: str, threshold: int) -> list[str]:
    """Replaces any <RawTail> longer than `threshold` bytes with a `File`
    attribute pointing at a sidecar written next to `out_path`, for every
    Object-shaped element in the tree (the root, and any inline nested
    object). Returns the sidecar paths written."""
    base, _ = os.path.splitext(out_path)
    written: list[str] = []
    counter = [0]

    def visit(el: ET.Element) -> None:
        for child in list(el):
            if child.tag == "RawTail":
                tail_len = int(child.get("Bytes", "0"))
                if tail_len > threshold and child.text:
                    tail = base64.b64decode(child.text)
                    ext_and_data = _texture_sidecar(el, tail)
                    if ext_and_data is not None:
                        ext, sidecar_bytes = ext_and_data
                    else:
                        ext, sidecar_bytes = ".bin", tail
                    suffix = "" if counter[0] == 0 else f"_{counter[0]}"
                    counter[0] += 1
                    sidecar_path = f"{base}{suffix}{ext}"
                    with open(sidecar_path, "wb") as f:
                        f.write(sidecar_bytes)
                    child.text = None
                    child.set("File", os.path.basename(sidecar_path))
                    child.set("FileFormat", ext.lstrip("."))
                    written.append(sidecar_path)
            else:
                visit(child)

    visit(root)
    return written


def _internalize_raw_tails(root: ET.Element, xml_path: str) -> None:
    """Inverse of _externalize_raw_tails: loads every sidecar `File`
    reference back in as inline base64, stripping the synthetic DDS header
    _texture_sidecar added (if any) so the original raw tail bytes are
    restored exactly."""
    directory = os.path.dirname(xml_path)

    def visit(el: ET.Element) -> None:
        for child in el.iter("RawTail"):
            file_name = child.get("File")
            if not file_name:
                continue
            with open(os.path.join(directory, file_name), "rb") as f:
                sidecar_bytes = f.read()
            if child.get("FileFormat") == "dds":
                sidecar_bytes = sidecar_bytes[128:]  # strip the header we added back off
            child.text = base64.b64encode(sidecar_bytes).decode("ascii")
            del child.attrib["File"]
            del child.attrib["FileFormat"]

    visit(root)


def write_object_xml(data: bytes, schema: Schema, legacy: bool, out_path: str, *, sidecar_threshold: int = 64) -> list[str]:
    """Decodes `data` and writes it as an XML file at `out_path`. RawTail
    blobs longer than `sidecar_threshold` bytes are externalized to sibling
    files next to `out_path` (e.g. foo.xml -> foo.dds or foo.bin) instead of
    being inlined as base64 -- this is what surfaces embedded texture/mesh
    payloads as directly usable image files. Returns the sidecar paths
    written, if any."""
    root = decode_object(data, schema, legacy)
    written = _externalize_raw_tails(root, out_path, sidecar_threshold)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return written


def read_object_xml(xml_path: str, legacy: bool) -> bytes:
    """Inverse of write_object_xml: loads the XML (resolving any sidecar
    `File` references relative to xml_path's directory) and encodes it back
    to the original binary form."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    _internalize_raw_tails(root, xml_path)
    return encode_object(root, legacy)
