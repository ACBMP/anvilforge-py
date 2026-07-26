"""Pushes a hand-edited XML (from `search.extract_match`) into every current
`search` match of a query -- rewriting each matched sub-part's binary
content across however many forges/.data entries hold a copy of it.

Reuses the existing, already-tested pipeline wholesale (forge.unpack/repack,
datafile.unpack_datafile/repack_datafile, objectxml.decode_object) rather
than patching bytes in place -- there is no in-place container-patch
infrastructure in this codebase yet (see the "known caveat" in the plan this
was built from: a full forge repack resets a few per-entry metadata fields
for every entry in that forge, not just the ones touched here).
"""
from __future__ import annotations

import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Iterable

from .datafile import (
    derive_uid_and_ext,
    object_id_bytes,
    repack_datafile,
    splice_object_id,
    unpack_datafile,
)
from .forge import backup as backup_forge, repack, unpack
from .games import Game, is_legacy
from .objectxml import decode_object
from .schema import Schema
from .search import (
    build_index,
    find_subpart_payload,
    parse_match_path,
    read_entry_bytes,
    search_index,
)


@dataclass
class ReplacePlan:
    root: str
    game: Game
    forges: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    skipped_whole_entries: list[str] = field(default_factory=list)

    @property
    def subpart_count(self) -> int:
        return sum(len(subs) for data in self.forges.values() for subs in data.values())


def plan_replace(
    root: str,
    game: Game,
    query: str,
    index_path: str | None = None,
    exclude: Iterable[str] = (),
    case_sensitive: bool = False,
) -> ReplacePlan:
    """Groups every current `search_index` match of `query` (minus
    `exclude`) by forge, then by `.data` entry, so `apply_replace` touches
    each container only once. Matches naming a whole `.data`/`.MetaFile`/
    `.PrefetchInfo` entry (no sub-part) aren't schema-decodable this way
    and are recorded in `skipped_whole_entries` instead."""
    root = os.path.abspath(root)
    index = build_index(root, game, index_path=index_path)
    matches = search_index(index, query, case_sensitive=case_sensitive)
    exclude_set = set(exclude)

    plan = ReplacePlan(root=root, game=game)
    for match_path in matches:
        if match_path in exclude_set:
            continue
        forge_relpath, entry_loose_name, subpart_loose_name = parse_match_path(index, match_path)
        if subpart_loose_name is None:
            plan.skipped_whole_entries.append(match_path)
            continue
        plan.forges.setdefault(forge_relpath, {}).setdefault(entry_loose_name, []).append(
            subpart_loose_name
        )
    return plan


def _edited_type_hash(edited_xml_path: str) -> str | None:
    return ET.parse(edited_xml_path).getroot().get("TypeHash")


def _sidecar_names(edited_xml_path: str) -> set[str]:
    names: set[str] = set()
    for el in ET.parse(edited_xml_path).getroot().iter():
        file_name = el.get("File")
        if file_name:
            names.add(file_name)
    return names


def _edited_ext_code(edited_raw_path: str, legacy: bool) -> int:
    with open(edited_raw_path, "rb") as f:
        raw = f.read()
    _uid, ext_code = derive_uid_and_ext(raw, legacy)
    return ext_code


def verify_plan(plan: ReplacePlan, edited_path: str, *, raw: bool = False) -> None:
    """Raises ValueError (naming the offending match) if any planned
    sub-part's *current* type doesn't match edited_path's -- checked up
    front so a mismatch anywhere aborts before `apply_replace` has written
    anything.

    Non-`raw` mode compares decoded TypeHash (objectxml.decode_object, the
    same schema-driven decode `extract_match` uses), via the offset-based,
    no-unpack read `extract_match` also uses. `raw` mode instead compares
    the raw ext_code embedded in each sub-part's own container framing
    (datafile.derive_uid_and_ext), straight from the index -- no schema
    decode at all, so it's exactly as safe for types whose XML conversion
    isn't reliable (e.g. FX) as the raw extract/apply path itself."""
    index = build_index(plan.root, plan.game)
    legacy = is_legacy(plan.game)

    if raw:
        expected = _edited_ext_code(edited_path, legacy)
        for forge_relpath, data_entries in plan.forges.items():
            forge_data = index["forges"][forge_relpath]
            for entry_loose_name, subpart_loose_names in data_entries.items():
                entry = next(
                    e for e in forge_data["entries"] if e["loose_name"] == entry_loose_name
                )
                ext_codes_by_name = {
                    f"{i}_-_{name}.{ext}": ext_code
                    for i, name, ext_code, ext in forge_data["subparts"][str(entry["i"])]
                }
                for subpart_loose_name in subpart_loose_names:
                    actual = ext_codes_by_name[subpart_loose_name]
                    if actual != expected:
                        raise ValueError(
                            f"type mismatch at {forge_relpath}/{entry_loose_name}/"
                            f"{subpart_loose_name}: expected ext_code {expected!r}, found "
                            f"{actual!r} -- aborting, nothing has been written"
                        )
        return

    schema = Schema.load_default(plan.game)
    expected = _edited_type_hash(edited_path)
    for forge_relpath, data_entries in plan.forges.items():
        for entry_loose_name, subpart_loose_names in data_entries.items():
            data = read_entry_bytes(plan.root, index, plan.game, forge_relpath, entry_loose_name)
            for subpart_loose_name in subpart_loose_names:
                payload = find_subpart_payload(data, plan.game, schema, subpart_loose_name)
                actual = decode_object(payload, schema, legacy).get("TypeHash")
                if actual != expected:
                    raise ValueError(
                        f"type mismatch at {forge_relpath}/{entry_loose_name}/"
                        f"{subpart_loose_name}: expected TypeHash {expected!r}, found "
                        f"{actual!r} -- aborting, nothing has been written"
                    )


def _apply_raw(target_path: str, edited_bytes: bytes, legacy: bool) -> None:
    """Overwrites target_path with edited_bytes, except its own ID field is
    kept as-is: other objects in the same .data may reference this sub-part
    by that ID, and blindly adopting a donor instance's ID would silently
    break those references (real, previously-shipped bug -- see
    datafile.object_id_bytes)."""
    with open(target_path, "rb") as f:
        original_id = object_id_bytes(f.read(), legacy)
    with open(target_path, "wb") as f:
        f.write(splice_object_id(edited_bytes, legacy, original_id))


def _apply_xml(target_xml_path: str, edited_root: ET.Element) -> None:
    """Writes a copy of edited_root to target_xml_path, except its own ID
    (and PreHeader, if any) are kept as target_xml_path's current values --
    same identity-preservation rationale as _apply_raw, at the XML level."""
    original_root = ET.parse(target_xml_path).getroot()
    original_id = original_root.get("ID")
    original_pre_header = original_root.get("PreHeader")

    patched = ET.fromstring(ET.tostring(edited_root))
    if original_id is not None:
        patched.set("ID", original_id)
    if original_pre_header is not None:
        patched.set("PreHeader", original_pre_header)
    else:
        patched.attrib.pop("PreHeader", None)

    ET.ElementTree(patched).write(target_xml_path, encoding="utf-8", xml_declaration=True)


def apply_replace(
    plan: ReplacePlan, edited_path: str, *, make_backup: bool = False, raw: bool = False
) -> None:
    """Executes `plan` (see plan_replace) -- call `verify_plan` first. For
    every forge involved: unpacks it (forge.unpack), overwrites the planned
    sub-parts inside each touched `.data` with edited_path's content (a raw
    byte copy when `raw`, otherwise an `.xml` sibling -- repack_datafile's
    existing `_sync_xml_siblings` then re-encodes it -- plus any sidecars
    edited_path references), *except* each target's own ID/PreHeader is
    preserved rather than adopted from edited_path (see _apply_raw/
    _apply_xml) -- every untouched sub-part stays byte-identical -- then
    repacks the whole forge to a sibling temp file and atomically
    `os.replace`s the original.

    `raw` also skips objectxml entirely while re-unpacking each `.data`
    (`unpack_datafile`'s `write_xml=False`) -- not just for the touched
    sub-part, but for every sub-part sharing that `.data`, so a type whose
    XML decode is unreliable can't crash this path via some *other*,
    untouched sub-part in the same container."""
    game = plan.game
    legacy = is_legacy(game)
    schema = Schema.load_default(game)

    if raw:
        with open(edited_path, "rb") as f:
            edited_bytes = f.read()
    else:
        edited_root = ET.parse(edited_path).getroot()
        sidecar_names = _sidecar_names(edited_path)
        xml_dir = os.path.dirname(os.path.abspath(edited_path))

    for forge_relpath, data_entries in plan.forges.items():
        forge_path = os.path.join(plan.root, forge_relpath)
        if make_backup:
            backup_forge(forge_path)

        with tempfile.TemporaryDirectory(prefix="anvilforge_replace_forge_") as tmp_forge_dir:
            original_entries = unpack(forge_path, tmp_forge_dir, game)

            for entry_loose_name, subpart_loose_names in data_entries.items():
                data_path = os.path.join(tmp_forge_dir, entry_loose_name)
                with tempfile.TemporaryDirectory(prefix="anvilforge_replace_data_") as tmp_data_dir:
                    unpack_datafile(data_path, tmp_data_dir, game, schema=schema, write_xml=not raw)
                    for subpart_loose_name in subpart_loose_names:
                        if raw:
                            target = os.path.join(tmp_data_dir, subpart_loose_name)
                            _apply_raw(target, edited_bytes, legacy)
                        else:
                            target_xml = os.path.join(tmp_data_dir, subpart_loose_name + ".xml")
                            _apply_xml(target_xml, edited_root)
                            for sidecar_name in sidecar_names:
                                shutil.copyfile(
                                    os.path.join(xml_dir, sidecar_name),
                                    os.path.join(tmp_data_dir, sidecar_name),
                                )
                    repack_datafile(tmp_data_dir, data_path, game)

            new_forge_path = forge_path + ".new"
            repack(tmp_forge_dir, new_forge_path, game, original_entries=original_entries)
            os.replace(new_forge_path, forge_path)


def replace(
    root: str,
    game: Game,
    query: str,
    edited_path: str,
    index_path: str | None = None,
    exclude: Iterable[str] = (),
    case_sensitive: bool = False,
    make_backup: bool = False,
    dry_run: bool = False,
    raw: bool = False,
) -> ReplacePlan:
    """Top-level entry point: plans, type-checks, and (unless `dry_run`)
    applies replacing every current match of `query` with `edited_path`'s
    content (raw bytes if `raw`, else XML). Returns the plan either way,
    for reporting."""
    plan = plan_replace(
        root, game, query, index_path=index_path, exclude=exclude, case_sensitive=case_sensitive
    )
    if plan.subpart_count:
        verify_plan(plan, edited_path, raw=raw)
        if not dry_run:
            apply_replace(plan, edited_path, make_backup=make_backup, raw=raw)
    return plan
