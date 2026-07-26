"""Synthetic, always-on tests for search.py -- unlike test_real_*.py (which
is skip-gated on a machine-local game install), these build small but fully
valid .forge/.data fixtures from scratch via the existing repack()/
repack_datafile() pipeline, so they run in any environment.
"""
from __future__ import annotations

import io
import json
import os

import pytest

from anvilforge import search as search_mod
from anvilforge.binio import write_class_id
from anvilforge.cli import main as cli_main
from anvilforge.datafile import repack_datafile
from anvilforge.forge import repack
from anvilforge.games import Game

GAME = Game.BROTHERHOOD
LEGACY = True


def _synthetic_subpart(uid: int, ext_code: int, payload: bytes = b"hello world") -> bytes:
    """A loose sub-part payload satisfying _derive_uid_and_ext's peek: 2
    pad bytes (first != 1, so no import-table extra), a class-id, a 4-byte
    ext_code, then arbitrary trailing bytes."""
    buf = io.BytesIO()
    buf.write(b"\x00\x00")
    write_class_id(buf, uid, LEGACY)
    buf.write((ext_code & 0xFFFFFFFF).to_bytes(4, "little"))
    buf.write(payload)
    return buf.getvalue()


def _build_data_file(build_dir: str, name: str, subparts: list[tuple[int, str, int, int]]) -> str:
    """subparts: list of (index, name, uid, ext_code). Returns the path to
    the built .data file (as a sibling of build_dir)."""
    in_dir = os.path.join(build_dir, f"loose_{name}")
    os.makedirs(in_dir)
    for index, sub_name, uid, ext_code in subparts:
        path = os.path.join(in_dir, f"{index}_-_{sub_name}.FX")
        with open(path, "wb") as f:
            f.write(_synthetic_subpart(uid, ext_code))
    data_path = os.path.join(build_dir, name)
    repack_datafile(in_dir, data_path, GAME)
    return data_path


def _build_forge(
    build_dir: str,
    forge_path: str,
    data_entries: dict[str, list[tuple[int, str, int, int]]],
) -> None:
    """data_entries: {loose .data filename: subparts spec}. Builds each
    .data file plus a trivial .MetaFile into a loose folder, then repacks
    that folder into a real .forge at forge_path."""
    loose_dir = os.path.join(build_dir, "loose_forge_" + os.path.basename(forge_path))
    os.makedirs(loose_dir)
    for i, (loose_name, subparts) in enumerate(data_entries.items()):
        built = _build_data_file(build_dir, f"{i}_{loose_name}", subparts)
        with open(built, "rb") as src, open(os.path.join(loose_dir, loose_name), "wb") as dst:
            dst.write(src.read())
    with open(os.path.join(loose_dir, "4_-_SomeMeta.MetaFile"), "wb") as f:
        f.write(b"meta-payload")

    os.makedirs(os.path.dirname(forge_path), exist_ok=True)
    repack(loose_dir, forge_path, GAME)


@pytest.fixture
def game_root(tmp_path):
    root = tmp_path / "game_root"
    root.mkdir()
    build = tmp_path / "build"
    build.mkdir()

    _build_forge(
        str(build),
        str(root / "DataPC.forge"),
        {
            "83_-_AC2MP_ID20_MainTemplate_Player.data": [
                (0, "AC2MP_FX_SmokeBomb_01", 1001, 111),
                (1, "AC2MP_FX_SmokeBomb_02", 1002, 222),
            ],
        },
    )
    _build_forge(
        str(build),
        str(root / "sub" / "Extra.forge"),
        {
            "5_-_AC2MP_ID99_Other.data": [
                (0, "AC2MP_FX_SmokeBomb_03", 1003, 333),
            ],
        },
    )
    return str(root)


def test_find_forges_recursive(game_root):
    found = search_mod.find_forges(game_root)
    assert found == sorted(found)
    assert "DataPC.forge" in found
    assert os.path.join("sub", "Extra.forge") in found


def test_build_index_records_entries_and_subparts(game_root):
    index = search_mod.build_index(game_root, GAME)
    assert index["game"] == "acb"
    assert set(index["forges"]) == {"DataPC.forge", os.path.join("sub", "Extra.forge")}

    forge_data = index["forges"]["DataPC.forge"]
    names = {e["loose_name"] for e in forge_data["entries"]}
    assert any(n.endswith("AC2MP_ID20_MainTemplate_Player.data") for n in names)
    assert any(n.endswith(".MetaFile") for n in names)

    data_entry = next(
        e for e in forge_data["entries"] if e["loose_name"].endswith(".data")
    )
    subparts = forge_data["subparts"][str(data_entry["i"])]
    sub_names = {s[1] for s in subparts}
    assert sub_names == {"AC2MP_FX_SmokeBomb_01", "AC2MP_FX_SmokeBomb_02"}


def test_search_index_partial_name_match(game_root):
    index = search_mod.build_index(game_root, GAME)
    forge_data = index["forges"]["DataPC.forge"]
    data_entry = next(e for e in forge_data["entries"] if e["loose_name"].endswith(".data"))

    results = search_mod.search_index(index, "SmokeBomb_01")
    assert results == [
        f"DataPC.forge/{data_entry['loose_name']}/0_-_AC2MP_FX_SmokeBomb_01.111"
    ]


def test_search_index_matches_across_forges_case_insensitive(game_root):
    index = search_mod.build_index(game_root, GAME)
    results = search_mod.search_index(index, "smokebomb")
    assert len(results) == 3


def test_search_top_level_convenience_function(game_root):
    results = search_mod.search(game_root, GAME, "SmokeBomb_03")
    assert len(results) == 1
    assert "Extra.forge" in results[0]
    assert os.path.isfile(os.path.join(game_root, search_mod.DEFAULT_INDEX_NAME))


def test_incremental_build_skips_unchanged_forges(game_root, monkeypatch):
    calls = []
    real_scan = search_mod._scan_forge

    def counting_scan(path, game):
        calls.append(path)
        return real_scan(path, game)

    monkeypatch.setattr(search_mod, "_scan_forge", counting_scan)

    search_mod.build_index(game_root, GAME, force=True)
    assert len(calls) == 2

    calls.clear()
    search_mod.build_index(game_root, GAME)
    assert calls == []

    calls.clear()
    search_mod.build_index(game_root, GAME, force=True)
    assert len(calls) == 2


def test_reindex_rescans_a_touched_forge(game_root, monkeypatch):
    calls = []
    real_scan = search_mod._scan_forge

    def counting_scan(path, game):
        calls.append(path)
        return real_scan(path, game)

    monkeypatch.setattr(search_mod, "_scan_forge", counting_scan)
    search_mod.build_index(game_root, GAME, force=True)

    forge_path = os.path.join(game_root, "DataPC.forge")
    with open(forge_path, "ab") as f:
        f.write(b"\x00")

    calls.clear()
    search_mod.build_index(game_root, GAME)
    assert calls == [forge_path]


def test_deleted_forge_is_pruned_from_index(game_root):
    search_mod.build_index(game_root, GAME, force=True)
    os.remove(os.path.join(game_root, "DataPC.forge"))

    index = search_mod.build_index(game_root, GAME)
    assert "DataPC.forge" not in index["forges"]
    assert os.path.join("sub", "Extra.forge") in index["forges"]


def test_cli_search_prints_matches(game_root, capsys):
    rc = cli_main(["search", "--game", "acb", game_root, "SmokeBomb_02"])
    assert rc == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1
    assert out[0].startswith("DataPC.forge/")
    assert out[0].endswith("AC2MP_FX_SmokeBomb_02.222")


def test_index_rebuilt_when_stored_game_differs(game_root):
    index_path = os.path.join(game_root, search_mod.DEFAULT_INDEX_NAME)
    stale = {
        "game": "ac4",
        "forges": {"stale.forge": {"mtime": 1, "size": 1, "entries": [], "subparts": {}}},
    }
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(stale, f)

    index = search_mod.build_index(game_root, GAME)
    assert index["game"] == "acb"
    assert "stale.forge" not in index["forges"]
    assert "DataPC.forge" in index["forges"]
