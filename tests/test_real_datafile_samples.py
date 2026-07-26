"""Validates the DataFile (loose .data entry) unpack/repack pipeline against
real, retail Assassin's Creed Brotherhood multiplayer .forge files, if
present on disk. Skipped if the sample directory isn't found -- see
test_real_acb_samples.py.
"""
from __future__ import annotations

import filecmp
import os

import pytest

from anvilforge.datafile import repack_datafile, unpack_datafile
from anvilforge.forge import unpack
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


def _non_dependency_names(dir_path: str) -> set[str]:
    return {n for n in os.listdir(dir_path) if not n.lower().endswith(".dependency")}


@pytest.mark.parametrize("filename", SAMPLES)
def test_datafile_round_trip_preserves_subpart_content(tmp_path, filename):
    src = os.path.join(SAMPLE_DIR, filename)
    if not os.path.exists(src):
        pytest.skip(f"{filename} not present")

    forge_out = tmp_path / "forge_out"
    unpack(src, str(forge_out), Game.BROTHERHOOD)

    data_files = [n for n in os.listdir(forge_out) if n.lower().endswith(".data")]
    assert data_files, "expected at least one loose .data entry"

    tested = 0
    for name in data_files:
        data_path = forge_out / name
        if data_path.stat().st_size == 0:
            continue

        unpacked = tmp_path / f"u1_{name}"
        repacked = tmp_path / f"rp_{name}"
        reunpacked = tmp_path / f"u2_{name}"

        unpack_datafile(str(data_path), str(unpacked), Game.BROTHERHOOD)
        repack_datafile(str(unpacked), str(repacked), Game.BROTHERHOOD)
        unpack_datafile(str(repacked), str(reunpacked), Game.BROTHERHOOD)

        names1 = _non_dependency_names(str(unpacked))
        names2 = _non_dependency_names(str(reunpacked))
        assert names1 == names2, f"{name}: sub-part name set changed across round trip"
        for sub in names1:
            assert filecmp.cmp(
                unpacked / sub, reunpacked / sub, shallow=False
            ), f"{name}/{sub}: content changed across round trip"
        tested += 1

    assert tested, "expected at least one non-empty .data entry to test"
