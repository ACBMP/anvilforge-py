"""Synthetic, always-on tests for search.extract_match and replace.py --
same self-built-fixture style as test_search.py (no real game files needed).
"""
from __future__ import annotations

import base64
import io
import os
import xml.etree.ElementTree as ET

import pytest

from anvilforge import search as search_mod
from anvilforge.binio import write_class_id
from anvilforge.cli import main as cli_main
from anvilforge.datafile import repack_datafile
from anvilforge.forge import repack, unpack
from anvilforge.games import Game
from anvilforge.objectxml import read_object_xml
from anvilforge.replace import plan_replace, replace as run_replace

GAME = Game.BROTHERHOOD
LEGACY = True
MARKER = b"\x00\x01"  # objectxml._HEADER_MARKER -- see its module docstring;
# the real marker value round-trips byte-exact through decode+encode, unlike
# an arbitrary placeholder (encode_object always re-emits the real constant).


def _synthetic_object_payload(uid: int, type_hash: int, tail: bytes = b"hello world") -> bytes:
    """A loose sub-part payload that's simultaneously valid .data container
    framing (datafile._derive_uid_and_ext's peek) *and* a valid
    objectxml.decode_root object header -- confirmed the same header,
    reused for two purposes (see objectxml._read_header's docstring)."""
    buf = io.BytesIO()
    buf.write(MARKER)
    write_class_id(buf, uid, LEGACY)
    buf.write((type_hash & 0xFFFFFFFF).to_bytes(4, "little"))
    buf.write(tail)
    return buf.getvalue()


def _build_data_file(build_dir: str, name: str, subparts: list[tuple[int, str, int, int, bytes]]) -> str:
    """subparts: list of (index, name, uid, type_hash, tail)."""
    in_dir = os.path.join(build_dir, f"loose_{name}")
    os.makedirs(in_dir)
    for index, sub_name, uid, type_hash, tail in subparts:
        path = os.path.join(in_dir, f"{index}_-_{sub_name}.FX")
        with open(path, "wb") as f:
            f.write(_synthetic_object_payload(uid, type_hash, tail))
    data_path = os.path.join(build_dir, name)
    repack_datafile(in_dir, data_path, GAME)
    return data_path


def _build_forge(
    build_dir: str,
    forge_path: str,
    data_entries: dict[str, list[tuple[int, str, int, int, bytes]]],
) -> None:
    loose_dir = os.path.join(build_dir, "loose_forge_" + os.path.basename(forge_path))
    os.makedirs(loose_dir)
    for i, (loose_name, subparts) in enumerate(data_entries.items()):
        built = _build_data_file(build_dir, f"{i}_{loose_name}", subparts)
        with open(built, "rb") as src, open(os.path.join(loose_dir, loose_name), "wb") as dst:
            dst.write(src.read())
    with open(os.path.join(loose_dir, "9_-_SomeMeta.MetaFile"), "wb") as f:
        f.write(b"meta-payload")
    os.makedirs(os.path.dirname(forge_path), exist_ok=True)
    repack(loose_dir, forge_path, GAME)


@pytest.fixture
def dup_root(tmp_path):
    """The same FX asset (matching type hash, different payload) duplicated
    across two forges, plus an unrelated sibling sub-part that must survive
    untouched through a `replace`."""
    root = tmp_path / "game_root"
    root.mkdir()
    build = tmp_path / "build"
    build.mkdir()

    _build_forge(
        str(build),
        str(root / "DataPC.forge"),
        {
            "83_-_AC2MP_ID20_MainTemplate_Player.data": [
                (0, "AC2MP_FX_SmokeBomb_01", 1001, 555111, b"payload-A"),
                (1, "DecoySibling", 1002, 999888, b"decoy-untouched"),
            ],
        },
    )
    _build_forge(
        str(build),
        str(root / "sub" / "Extra.forge"),
        {
            "5_-_AC2MP_ID99_Other.data": [
                (0, "AC2MP_FX_SmokeBomb_01", 1003, 555111, b"payload-B"),
            ],
        },
    )
    return str(root)


@pytest.fixture
def mismatch_root(tmp_path):
    """Two matches for the same query, but with different type hashes --
    replace must refuse to touch either."""
    root = tmp_path / "game_root"
    root.mkdir()
    build = tmp_path / "build"
    build.mkdir()

    _build_forge(
        str(build),
        str(root / "DataPC.forge"),
        {
            "83_-_AC2MP_ID20_MainTemplate_Player.data": [
                (0, "AC2MP_FX_SmokeBomb_01", 1001, 555111, b"payload-A"),
            ],
        },
    )
    _build_forge(
        str(build),
        str(root / "bad" / "Bad.forge"),
        {
            "9_-_AC2MP_ID77_Bad.data": [
                (0, "AC2MP_FX_SmokeBomb_01", 1004, 999999, b"wrong-type"),
            ],
        },
    )
    return str(root)


def test_extract_round_trips_exactly(dup_root, tmp_path):
    matches = search_mod.search(dup_root, GAME, "SmokeBomb_01")
    assert len(matches) == 2
    match = next(m for m in matches if "DataPC.forge" in m)

    out_dir = tmp_path / "extracted"
    xml_path, sidecars = search_mod.extract_match(dup_root, GAME, match, str(out_dir))
    assert sidecars == []
    assert os.path.isfile(xml_path)

    root = ET.parse(xml_path).getroot()
    assert root.get("TypeHash") == "555111"

    original = _synthetic_object_payload(1001, 555111, b"payload-A")
    assert read_object_xml(xml_path, LEGACY) == original


def test_extract_whole_entry_match_raises(dup_root, tmp_path):
    matches = search_mod.search(dup_root, GAME, "AC2MP_ID99_Other")
    whole_entry_match = next(m for m in matches if m.endswith(".data"))
    with pytest.raises(ValueError, match="not a sub-part"):
        search_mod.extract_match(dup_root, GAME, whole_entry_match, str(tmp_path / "out"))


def test_extract_externalizes_large_tail_to_sidecar(tmp_path):
    root_dir = tmp_path / "game_root"
    root_dir.mkdir()
    build = tmp_path / "build"
    build.mkdir()
    big_tail = b"Z" * 200

    _build_forge(
        str(build),
        str(root_dir / "Big.forge"),
        {"1_-_BigOne.data": [(0, "AC2MP_FX_BigAsset", 2001, 555111, big_tail)]},
    )

    matches = search_mod.search(str(root_dir), GAME, "BigAsset")
    assert len(matches) == 1
    xml_path, sidecars = search_mod.extract_match(
        str(root_dir), GAME, matches[0], str(tmp_path / "extracted_big")
    )
    assert len(sidecars) == 1
    assert os.path.isfile(sidecars[0])
    assert read_object_xml(xml_path, LEGACY) == _synthetic_object_payload(2001, 555111, big_tail)


def test_replace_propagates_and_preserves_siblings(dup_root, tmp_path):
    matches = search_mod.search(dup_root, GAME, "SmokeBomb_01")
    assert len(matches) == 2
    source_match = next(m for m in matches if "DataPC.forge" in m)

    xml_path, _ = search_mod.extract_match(dup_root, GAME, source_match, str(tmp_path / "extracted"))

    tree = ET.parse(xml_path)
    raw_tail = tree.getroot().find("RawTail")
    new_tail = b"EDITED-CONTENT"
    raw_tail.text = base64.b64encode(new_tail).decode("ascii")
    raw_tail.set("Bytes", str(len(new_tail)))
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    plan = run_replace(dup_root, GAME, "SmokeBomb_01", str(xml_path))
    assert plan.subpart_count == 2
    assert len(plan.forges) == 2

    reindexed_matches = search_mod.search(dup_root, GAME, "SmokeBomb_01", reindex=True)
    assert len(reindexed_matches) == 2
    seen_ids = set()
    for i, match in enumerate(reindexed_matches):
        xml_out, _ = search_mod.extract_match(dup_root, GAME, match, str(tmp_path / f"check_{i}"))
        root = ET.parse(xml_out).getroot()
        assert base64.b64decode(root.find("RawTail").text) == new_tail
        seen_ids.add(root.get("ID"))
    # Each target keeps its *own* original ID -- content is replaced, identity is not
    # (a prior version of this code cloned the source's ID onto every target,
    # which broke other objects' by-ID references to it and crashed the game).
    assert seen_ids == {(1001).to_bytes(4, "little").hex(), (1003).to_bytes(4, "little").hex()}

    decoy_matches = search_mod.search(dup_root, GAME, "DecoySibling", reindex=True)
    assert len(decoy_matches) == 1
    xml_out, _ = search_mod.extract_match(dup_root, GAME, decoy_matches[0], str(tmp_path / "check_decoy"))
    assert base64.b64decode(ET.parse(xml_out).getroot().find("RawTail").text) == b"decoy-untouched"


def test_replace_dry_run_does_not_write(dup_root, tmp_path):
    matches = search_mod.search(dup_root, GAME, "SmokeBomb_01")
    source_match = next(m for m in matches if "DataPC.forge" in m)
    xml_path, _ = search_mod.extract_match(dup_root, GAME, source_match, str(tmp_path / "extracted"))

    forge_paths = [
        os.path.join(dup_root, "DataPC.forge"),
        os.path.join(dup_root, "sub", "Extra.forge"),
    ]
    before = {p: os.path.getmtime(p) for p in forge_paths}

    plan = run_replace(dup_root, GAME, "SmokeBomb_01", str(xml_path), dry_run=True)
    assert plan.subpart_count == 2

    after = {p: os.path.getmtime(p) for p in forge_paths}
    assert before == after


def test_replace_exclude_skips_named_match(dup_root, tmp_path):
    matches = search_mod.search(dup_root, GAME, "SmokeBomb_01")
    source_match = next(m for m in matches if "DataPC.forge" in m)
    other_match = next(m for m in matches if m != source_match)

    plan = plan_replace(dup_root, GAME, "SmokeBomb_01", exclude=[other_match])
    assert plan.subpart_count == 1


def test_replace_aborts_on_type_mismatch(mismatch_root, tmp_path):
    matches = search_mod.search(mismatch_root, GAME, "SmokeBomb_01")
    assert len(matches) == 2
    source_match = next(m for m in matches if "DataPC.forge" in m)
    xml_path, _ = search_mod.extract_match(
        mismatch_root, GAME, source_match, str(tmp_path / "extracted")
    )

    forge_paths = [
        os.path.join(mismatch_root, "DataPC.forge"),
        os.path.join(mismatch_root, "bad", "Bad.forge"),
    ]
    before = {p: os.path.getmtime(p) for p in forge_paths}

    with pytest.raises(ValueError, match="type mismatch"):
        run_replace(mismatch_root, GAME, "SmokeBomb_01", str(xml_path))

    after = {p: os.path.getmtime(p) for p in forge_paths}
    assert before == after


def test_cli_extract_and_replace_dry_run(dup_root, tmp_path, capsys):
    rc = cli_main(["search", "--game", "acb", dup_root, "SmokeBomb_01"])
    assert rc == 0
    matches = capsys.readouterr().out.strip().splitlines()
    assert len(matches) == 2
    source_match = next(m for m in matches if "DataPC.forge" in m)

    out_dir = tmp_path / "cli_extracted"
    rc = cli_main(["extract", "--game", "acb", dup_root, source_match, str(out_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Wrote" in out
    xml_path = os.path.join(str(out_dir), os.path.basename(source_match) + ".xml")
    assert os.path.isfile(xml_path)

    rc = cli_main(["replace", "--game", "acb", dup_root, "SmokeBomb_01", xml_path, "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Would replace 2 sub-part(s) across 2 forge(s)" in out


def test_extract_raw_writes_bytes_unchanged(dup_root, tmp_path):
    matches = search_mod.search(dup_root, GAME, "SmokeBomb_01")
    match = next(m for m in matches if "DataPC.forge" in m)

    out_path, sidecars = search_mod.extract_match(
        dup_root, GAME, match, str(tmp_path / "extracted_raw"), raw=True
    )
    assert sidecars == []
    assert os.path.basename(out_path) == os.path.basename(match)
    with open(out_path, "rb") as f:
        assert f.read() == _synthetic_object_payload(1001, 555111, b"payload-A")


def test_replace_raw_propagates_and_preserves_siblings(dup_root, tmp_path):
    matches = search_mod.search(dup_root, GAME, "SmokeBomb_01")
    assert len(matches) == 2
    source_match = next(m for m in matches if "DataPC.forge" in m)

    raw_path, _ = search_mod.extract_match(
        dup_root, GAME, source_match, str(tmp_path / "extracted_raw"), raw=True
    )
    # Same uid/type_hash header, hand-edited tail -- exactly what a raw
    # hex-edit of the extracted file would produce.
    edited = _synthetic_object_payload(1001, 555111, b"RAW-EDITED-CONTENT")
    with open(raw_path, "wb") as f:
        f.write(edited)

    plan = run_replace(dup_root, GAME, "SmokeBomb_01", raw_path, raw=True)
    assert plan.subpart_count == 2
    assert len(plan.forges) == 2

    reindexed_matches = search_mod.search(dup_root, GAME, "SmokeBomb_01", reindex=True)
    assert len(reindexed_matches) == 2
    # Content is replaced everywhere, but each target keeps its *own*
    # original ID (uid 1001 in DataPC.forge, uid 1003 in sub/Extra.forge) --
    # a prior version cloned the source's uid onto every target, which
    # broke other objects' by-ID references to it and crashed the game.
    for match in reindexed_matches:
        out_path, _ = search_mod.extract_match(
            dup_root, GAME, match, str(tmp_path / "check_raw"), raw=True
        )
        with open(out_path, "rb") as f:
            content = f.read()
        expected_uid = 1001 if "DataPC.forge" in match else 1003
        assert content == _synthetic_object_payload(expected_uid, 555111, b"RAW-EDITED-CONTENT")

    decoy_matches = search_mod.search(dup_root, GAME, "DecoySibling", reindex=True)
    assert len(decoy_matches) == 1
    out_path, _ = search_mod.extract_match(
        dup_root, GAME, decoy_matches[0], str(tmp_path / "check_decoy_raw"), raw=True
    )
    with open(out_path, "rb") as f:
        assert f.read() == _synthetic_object_payload(1002, 999888, b"decoy-untouched")


def test_replace_raw_aborts_on_ext_code_mismatch(mismatch_root, tmp_path):
    matches = search_mod.search(mismatch_root, GAME, "SmokeBomb_01")
    assert len(matches) == 2
    source_match = next(m for m in matches if "DataPC.forge" in m)
    raw_path, _ = search_mod.extract_match(
        mismatch_root, GAME, source_match, str(tmp_path / "extracted_raw"), raw=True
    )

    forge_paths = [
        os.path.join(mismatch_root, "DataPC.forge"),
        os.path.join(mismatch_root, "bad", "Bad.forge"),
    ]
    before = {p: os.path.getmtime(p) for p in forge_paths}

    with pytest.raises(ValueError, match="type mismatch"):
        run_replace(mismatch_root, GAME, "SmokeBomb_01", raw_path, raw=True)

    after = {p: os.path.getmtime(p) for p in forge_paths}
    assert before == after


def test_cli_extract_and_replace_raw_dry_run(dup_root, tmp_path, capsys):
    matches = search_mod.search(dup_root, GAME, "SmokeBomb_01")
    source_match = next(m for m in matches if "DataPC.forge" in m)

    out_dir = tmp_path / "cli_extracted_raw"
    rc = cli_main(["extract", "--game", "acb", dup_root, source_match, str(out_dir), "--raw"])
    assert rc == 0
    capsys.readouterr()
    raw_path = os.path.join(str(out_dir), os.path.basename(source_match))
    assert os.path.isfile(raw_path)

    rc = cli_main(
        ["replace", "--game", "acb", dup_root, "SmokeBomb_01", raw_path, "--raw", "--dry-run"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Would replace 2 sub-part(s) across 2 forge(s)" in out


_PRESERVED_ENTRY_FIELDS = (
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


def test_replace_preserves_forge_entry_metadata_for_every_entry(dup_root, tmp_path):
    """Regression test for a real, reported crash: applying `replace` used
    to silently reset umac_hash (and several other per-entry fields --
    forge.py's create_entry() can't recover them from a loose file alone)
    for *every* entry in the affected forge, not just the touched one,
    because forge.repack() rebuilt each ForgeEntry from scratch. This
    happened even with zero content edits."""
    forge_path = os.path.join(dup_root, "DataPC.forge")
    before = {e.id: e for e in unpack(forge_path, str(tmp_path / "before"), GAME)}

    matches = search_mod.search(dup_root, GAME, "SmokeBomb_01")
    source_match = next(m for m in matches if "DataPC.forge" in m)
    xml_path, _ = search_mod.extract_match(dup_root, GAME, source_match, str(tmp_path / "extracted"))

    plan = run_replace(dup_root, GAME, "SmokeBomb_01", str(xml_path))
    assert plan.subpart_count == 2

    after = {e.id: e for e in unpack(forge_path, str(tmp_path / "after"), GAME)}

    assert set(before) == set(after)
    for entry_id, entry_before in before.items():
        entry_after = after[entry_id]
        for field in _PRESERVED_ENTRY_FIELDS:
            assert getattr(entry_before, field) == getattr(entry_after, field), (
                f"entry {entry_id} field {field!r} drifted: "
                f"{getattr(entry_before, field)!r} -> {getattr(entry_after, field)!r}"
            )
