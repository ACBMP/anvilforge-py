"""Validates schema-driven binary<->XML object conversion (objectxml.py)
against real, retail Assassin's Creed Brotherhood multiplayer .forge files,
if present on disk. Skipped if the sample directory isn't found -- see
test_real_datafile_samples.py.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import pytest

from anvilforge.datafile import repack_datafile, unpack_datafile
from anvilforge.forge import unpack
from anvilforge.games import Game
from anvilforge.objectxml import encode_object, decode_object, read_object_xml
from anvilforge.schema import Schema

SAMPLE_DIR = (
    "/home/a/Games/assassins-creed-brotherhood/drive_c/Program Files (x86)/Ubisoft/"
    "Ubisoft Game Launcher/games/Assassin's Creed Brotherhood/multi"
)
SAMPLES = [
    "DataPC_extraparams.forge",
    "DataPC_Fire.forge",
    "DataPC_skins_0000_00000001_dlc.forge",
    "DataPC_skins_0001_00000002_dlc.forge",
]
SCHEMA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "src", "anvilforge", "schemas", "ACB_MP.schema"
)

pytestmark = pytest.mark.skipif(
    not os.path.isdir(SAMPLE_DIR), reason="real ACB MP sample forges not available"
)


@pytest.fixture(scope="module")
def schema() -> Schema:
    return Schema.load(SCHEMA_PATH)


def _unpack_all_datafiles(tmp_path, filename: str, schema: Schema):
    src = os.path.join(SAMPLE_DIR, filename)
    if not os.path.exists(src):
        pytest.skip(f"{filename} not present")
    forge_out = tmp_path / "forge_out"
    unpack(src, str(forge_out), Game.BROTHERHOOD)
    out_dirs = []
    for name in os.listdir(forge_out):
        if not name.lower().endswith(".data") or (forge_out / name).stat().st_size == 0:
            continue
        out_dir = tmp_path / f"unpacked_{name}"
        unpack_datafile(str(forge_out / name), str(out_dir), Game.BROTHERHOOD, schema=schema)
        out_dirs.append(out_dir)
    return out_dirs


@pytest.mark.parametrize("filename", SAMPLES)
def test_decoded_objects_re_encode_byte_exact(tmp_path, filename, schema):
    """The core round-trip guarantee objectxml.py is built around: whatever
    a sub-part decodes to (fully or only partially, falling back to a
    RawTail -- see objectxml.py's module docstring), encoding it back must
    reproduce the exact original bytes."""
    out_dirs = _unpack_all_datafiles(tmp_path, filename, schema)
    tested = 0
    for out_dir in out_dirs:
        for name in os.listdir(out_dir):
            if not name.endswith(".xml"):
                continue
            object_path = out_dir / name[: -len(".xml")]
            if not object_path.exists():
                continue
            with open(object_path, "rb") as f:
                original = f.read()
            root = decode_object(original, schema, legacy=True)
            assert encode_object(root, legacy=True) == original, f"{object_path}: not byte-exact"
            tested += 1
    assert tested, "expected at least one schema-recognized sub-part to test"


@pytest.mark.parametrize("filename", SAMPLES)
def test_xml_files_are_well_formed_and_file_round_trip(tmp_path, filename, schema):
    """Same guarantee as above, but through the on-disk XML (+ sidecar)
    files unpack_datafile actually writes, exercising ET parsing and
    _internalize_raw_tails' sidecar resolution too."""
    out_dirs = _unpack_all_datafiles(tmp_path, filename, schema)
    tested = 0
    for out_dir in out_dirs:
        for name in os.listdir(out_dir):
            if not name.endswith(".xml"):
                continue
            xml_path = out_dir / name
            ET.parse(xml_path)  # raises on malformed XML
            object_path = out_dir / name[: -len(".xml")]
            if not object_path.exists():
                continue
            with open(object_path, "rb") as f:
                original = f.read()
            assert read_object_xml(str(xml_path), legacy=True) == original
            tested += 1
    assert tested, "expected at least one schema-recognized sub-part to test"


@pytest.mark.parametrize("filename", SAMPLES)
def test_repack_with_schema_unpack_preserves_content(tmp_path, filename, schema):
    """unpack_datafile(..., schema=...) writes extra .xml/.dds files
    alongside the usual raw sub-parts; repack_datafile has to ignore all of
    that and still reproduce the same .data content as the schema-less
    path (see repack_datafile's `.xml`/sidecar exclusion and
    _sync_xml_siblings)."""
    out_dirs = _unpack_all_datafiles(tmp_path, filename, schema)
    tested = 0
    for out_dir in out_dirs:
        repacked = tmp_path / f"{out_dir.name}.repacked.data"
        repack_datafile(str(out_dir), str(repacked), Game.BROTHERHOOD)
        reunpacked = tmp_path / f"{out_dir.name}.reunpacked"
        # Same `schema=` as the first unpack, or extension names (part of
        # the filename) won't match between the two directories even though
        # the underlying sub-part content is identical.
        unpack_datafile(str(repacked), str(reunpacked), Game.BROTHERHOOD, schema=schema)

        skip_suffixes = (".xml", ".dds", ".bin", ".Dependency")
        raw_names_before = {n for n in os.listdir(out_dir) if not n.endswith(skip_suffixes)}
        raw_names_after = {n for n in os.listdir(reunpacked) if not n.endswith(skip_suffixes)}
        assert raw_names_before == raw_names_after
        for sub in raw_names_before:
            with open(out_dir / sub, "rb") as f1, open(reunpacked / sub, "rb") as f2:
                assert f1.read() == f2.read(), f"{sub}: content changed across schema-aware repack"
        tested += 1
    assert tested, "expected at least one non-empty .data entry to test"
