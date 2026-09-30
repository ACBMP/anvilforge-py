"""Validates fastload.py (the engine-rule object codec) against real, retail
multiplayer .forge files, if present on disk: every object it can decode must
consume its payload exactly and re-encode byte-for-byte, and the core world
types (Entity, EntityGroup, CrowdDutyRegion, ...) must all be decodable.
Skipped if the game installs aren't found.
"""
from __future__ import annotations

import io
import os
from collections import Counter

import pytest

from anvilforge.datafile import iter_datafile_subparts
from anvilforge.fastload import Codec, DecodeError, Obj, Ptr, walk
from anvilforge.fileset import iter_fileset_entries, read_entry_payload
from anvilforge.forge import read_header
from anvilforge.games import Game
from anvilforge.schema import Schema

ACB_DIR = "/home/a/vbox/Assassin's Creed Brotherhood/multi"
ACR_DIR = (
    "/home/a/Games/assassins-creed-revelations/drive_c/Program Files (x86)/Ubisoft/"
    "Ubisoft Game Launcher/Games/Assassin's Creed Revelations/multi"
)

# Types that must always decode (they're what a map port has to rewrite).
MUST_DECODE = {"Entity", "EntityGroup", "CrowdDutyRegion", "World", "WorldDataLayerManager",
               "MeshShape", "Skeleton", "Material", "Mesh", "TextureMap", "GridCellDataBlock"}


def _objects(path, game):
    with open(path, "rb") as f:
        for s in range(read_header(f, 25)):
            for entry in list(iter_fileset_entries(f, s, True)):
                raw = read_entry_payload(f, entry, True)
                for _i, ext, _name, payload in iter_datafile_subparts(io.BytesIO(raw), game):
                    yield ext, payload


def _schemas():
    acb = Schema.load_default(Game.BROTHERHOOD)
    acr = Schema.load_default(Game.REVELATIONS)
    return acb.with_placeholders_filled(acr), acr


def _check(path, game, schema):
    codec = Codec(schema)
    seen = Counter()
    for ext, payload in _objects(path, game):
        t = schema.name_of(ext)
        try:
            root = codec.decode(payload)
        except DecodeError:
            # some types (Animation, FX, MaterialTemplate, NavMeshManager,
            # cinematic cameras, ...) use hand-written serializers outside the
            # generic rules; the codec must reject those, never mis-decode them
            assert t not in MUST_DECODE, f"{t} failed to decode"
            continue
        assert codec.encode(root) == payload, f"{t} did not round-trip"
        seen[t] += 1
    return seen


@pytest.mark.skipif(not os.path.isdir(ACB_DIR), reason="ACB install not available")
@pytest.mark.parametrize("filename", ["DataPC_AC2MP_MtStMichel_dlc.forge", "DataPC_AC2MP_Alhambra_dlc.forge"])
def test_acb_maps_round_trip(filename):
    path = os.path.join(ACB_DIR, filename)
    if not os.path.exists(path):
        pytest.skip(f"{filename} not present")
    acb_true, _ = _schemas()
    seen = _check(path, Game.BROTHERHOOD, acb_true)
    assert seen["Entity"] > 100


@pytest.mark.skipif(not os.path.isdir(ACR_DIR), reason="ACR install not available")
def test_acr_dyers_round_trips():
    path = os.path.join(ACR_DIR, "DataPC_ACFE_Dyers_dlc.forge")
    if not os.path.exists(path):
        pytest.skip("Dyers not present")
    _, acr = _schemas()
    seen = _check(path, Game.REVELATIONS, acr)
    assert seen["Entity"] == 3147
    assert seen["CrowdDutyRegion"] == 37
    assert seen["MeshShape"] == 467


def test_encode_relayout_drops_fields_missing_from_target_schema():
    """Encoding against a different schema lays fields out by name, so a
    field the target lacks is simply not written (the port's core move)."""
    acb = Schema.load_default(Game.BROTHERHOOD)
    acr = Schema.load_default(Game.REVELATIONS)
    h = next(k for k, v in acr.names.items() if v == "Skeleton")
    codec_acr = Codec(acr)
    fields = {}
    for _cname, props in codec_acr.levels(h):
        for p in props:
            fields[acr.name_of(p.name_hash)] = None
    assert "Scale" in fields
    assert "Scale" not in {acb.name_of(p.name_hash) for _c, ps in Codec(acb).levels(h) for p in ps}
