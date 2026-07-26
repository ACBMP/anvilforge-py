"""Validates the forge unpack/repack pipeline against real, retail Assassin's
Creed Brotherhood multiplayer .forge files, if present on disk. These are not
shipped with the repo (copyrighted game assets); the tests are skipped if the
directory isn't found.
"""
from __future__ import annotations

import filecmp
import os

import pytest

from anvilforge.create_entry import create_entry
from anvilforge.forge import repack, unpack
from anvilforge.games import Game

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

pytestmark = pytest.mark.skipif(
    not os.path.isdir(SAMPLE_DIR), reason="real ACB MP sample forges not available"
)


@pytest.mark.parametrize("filename", SAMPLES)
def test_unpack_then_content_derived_ids_match_container_ids(tmp_path, filename):
    src = os.path.join(SAMPLE_DIR, filename)
    if not os.path.exists(src):
        pytest.skip(f"{filename} not present")

    out_dir = tmp_path / "unpacked"
    entries = unpack(src, str(out_dir), Game.BROTHERHOOD)
    assert entries, "expected at least one entry"

    names = sorted(os.listdir(out_dir), key=lambda n: int(n.split("_-_")[0]))
    assert len(names) == len(entries)

    mismatches = []
    for ground_truth, name in zip(entries, names):
        derived = create_entry(str(out_dir / name), 0, Game.BROTHERHOOD)
        if derived is None:
            continue  # a handful of retail entries are unparseable placeholders
        if derived.id != ground_truth.id:
            mismatches.append((name, hex(ground_truth.id), hex(derived.id)))

    assert not mismatches


@pytest.mark.parametrize("filename", SAMPLES)
def test_repack_round_trip_preserves_recoverable_entries(tmp_path, filename):
    src = os.path.join(SAMPLE_DIR, filename)
    if not os.path.exists(src):
        pytest.skip(f"{filename} not present")

    out_dir = tmp_path / "unpacked"
    unpack(src, str(out_dir), Game.BROTHERHOOD)

    repacked = tmp_path / "repacked.forge"
    repack(str(out_dir), str(repacked), Game.BROTHERHOOD)

    reunpacked_dir = tmp_path / "reunpacked"
    unpack(str(repacked), str(reunpacked_dir), Game.BROTHERHOOD)

    cmp = filecmp.dircmp(str(out_dir), str(reunpacked_dir))
    assert not cmp.right_only, "repack produced files that weren't in the original"
    assert not cmp.diff_files, "repacked entry content changed"
    # left_only is allowed: entries create_entry() can't parse (degenerate
    # placeholders in retail data) are legitimately dropped on repack.
