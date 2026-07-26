"""Finds named assets across a whole game install without unpacking anything
to disk.

A `.forge`'s own entries and a `.data`'s own sub-parts both carry a real,
human-readable name directly in their binary framing (see fileset.py's
`_read_entry_info`/`iter_fileset_entries` and datafile.py's
`iter_content_records`/`iter_datafile_subparts`) -- this module walks that
framing only (no schema-driven object decode, no loose-file writes) to build
a lightweight index mapping each such name to the same virtual path
components `unpack()`/`unpack_datafile()` would produce on disk, so a match
can be immediately acted on with the existing unpack commands.

Extension names (e.g. ".FX") do need a schema, but `Schema.load_default`
loads the one bundled for the game with no extra cost per sub-part beyond a
cached dict lookup (`resolve_extension`).
"""
from __future__ import annotations

import io
import json
import os
import tempfile

from .datafile import iter_datafile_subparts, resolve_extension
from .fileset import LEGACY_DATA_HEADER_SIZE, iter_fileset_entries, loose_file_name, read_entry_payload
from .forge import read_header
from .games import ENTRIES_PER_FILESET, GAME_CODES, Game, forge_version, is_legacy
from .objectxml import write_object_xml
from .schema import Schema

DEFAULT_INDEX_NAME = ".anvilforge_index.json"


def find_forges(root: str) -> list[str]:
    """Every `*.forge` file under root (recursive, case-insensitive),
    returned as paths relative to root."""
    matches = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".forge"):
                full = os.path.join(dirpath, name)
                matches.append(os.path.relpath(full, root))
    return sorted(matches)


def _scan_forge(path: str, game: Game) -> tuple[list[dict], dict[str, list]]:
    """Scans one .forge file's structure (entries + any .data sub-parts)
    with zero disk writes and zero schema-driven object decode -- only
    `resolve_extension`'s cheap hash->name lookup against the game's bundled
    default schema."""
    legacy = is_legacy(game)
    version = forge_version(game)
    schema = Schema.load_default(game)

    entries: list[dict] = []
    subparts: dict[str, list] = {}

    with open(path, "rb") as f:
        fileset_count = read_header(f, version)
        for set_index in range(fileset_count):
            for local_index, entry in enumerate(iter_fileset_entries(f, set_index, legacy)):
                global_index = set_index * ENTRIES_PER_FILESET + local_index
                name = loose_file_name(global_index, entry)
                kind = os.path.splitext(name)[1]
                entries.append(
                    {
                        "i": global_index,
                        "loose_name": name,
                        "offset": entry.offset,
                        "length": entry.length_on_disk,
                        "id": entry.id,
                    }
                )
                if kind != ".data":
                    continue
                # read_entry_payload seeks/reads f during the pause between
                # this yield and the generator's next resume -- safe, see
                # iter_fileset_entries's own docstring.
                try:
                    payload = read_entry_payload(f, entry, legacy)
                    sub_list = []
                    with io.BytesIO(payload) as buf:
                        for sub_index, ext_code, sub_name, _payload in iter_datafile_subparts(
                            buf, game
                        ):
                            ext = resolve_extension(ext_code, schema)
                            sub_list.append([sub_index, sub_name, ext_code, ext])
                    subparts[str(global_index)] = sub_list
                except Exception:
                    subparts[str(global_index)] = None

    return entries, subparts


def _load_index(index_path: str) -> dict:
    if not os.path.isfile(index_path):
        return {"game": None, "forges": {}}
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"game": None, "forges": {}}


def _save_index(index_path: str, index: dict) -> None:
    directory = os.path.dirname(index_path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".anvilforge_index_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(index, f)
        os.replace(tmp_path, index_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def build_index(
    root: str, game: Game, index_path: str | None = None, force: bool = False
) -> dict:
    """Builds (or incrementally updates) the search index rooted at `root`.

    Forges whose stored mtime+size still match the file on disk are reused
    as-is unless `force` is set; forges that no longer exist under `root`
    are pruned. Written to `index_path` (default: `<root>/.anvilforge_index.json`).
    """
    root = os.path.abspath(root)
    if index_path is None:
        index_path = os.path.join(root, DEFAULT_INDEX_NAME)

    index = _load_index(index_path)
    game_code = GAME_CODES[game]
    if index.get("game") != game_code:
        index = {"game": game_code, "forges": {}}

    forges = index.setdefault("forges", {})
    current = find_forges(root)
    current_set = set(current)

    for relpath in list(forges):
        if relpath not in current_set:
            del forges[relpath]

    for relpath in current:
        full_path = os.path.join(root, relpath)
        st = os.stat(full_path)
        cached = forges.get(relpath)
        if (
            not force
            and cached is not None
            and cached.get("mtime") == st.st_mtime
            and cached.get("size") == st.st_size
        ):
            continue
        entries, subparts = _scan_forge(full_path, game)
        forges[relpath] = {
            "mtime": st.st_mtime,
            "size": st.st_size,
            "entries": entries,
            "subparts": subparts,
        }

    _save_index(index_path, index)
    return index


def search_index(index: dict, query: str, case_sensitive: bool = False) -> list[str]:
    """Substring-matches `query` against every sub-part's resolved name
    (falling back to a `.data` entry's own name when none of its sub-parts
    match, and matching non-`.data` entries -- .MetaFile/.PrefetchInfo --
    directly) and returns composed virtual paths using exactly the naming
    convention `unpack()`/`unpack_datafile()` already write to disk."""
    needle = query if case_sensitive else query.lower()

    def matches(name: str) -> bool:
        return needle in (name if case_sensitive else name.lower())

    results: list[str] = []
    for forge_relpath, forge_data in sorted(index.get("forges", {}).items()):
        for entry in forge_data.get("entries", []):
            entry_loose_name = entry["loose_name"]
            sub_list = forge_data.get("subparts", {}).get(str(entry["i"])) or []
            matched_sub = False
            for sub_index, sub_name, _ext_code, ext in sub_list:
                sub_fname = f"{sub_index}_-_{sub_name}.{ext}"
                if matches(sub_fname):
                    results.append(f"{forge_relpath}/{entry_loose_name}/{sub_fname}")
                    matched_sub = True
            if not matched_sub and matches(entry_loose_name):
                results.append(f"{forge_relpath}/{entry_loose_name}")
    return results


def search(
    root: str,
    game: Game,
    query: str,
    index_path: str | None = None,
    reindex: bool = False,
    case_sensitive: bool = False,
) -> list[str]:
    """Builds/updates the index rooted at `root` (incremental unless
    `reindex`) and returns every matching virtual path for `query`."""
    index = build_index(root, game, index_path=index_path, force=reindex)
    return search_index(index, query, case_sensitive=case_sensitive)


def parse_match_path(index: dict, match_path: str) -> tuple[str, str, str | None]:
    """Splits a virtual path returned by `search_index` into
    (forge_relpath, entry_loose_name, subpart_loose_name) -- the latter is
    None for a match naming a whole top-level entry (.data/.MetaFile/
    .PrefetchInfo) rather than a sub-part inside one. Resolved against the
    index's own forge_relpath keys (rather than guessing from slash counts)
    since a forge_relpath can itself contain subdirectories."""
    for forge_relpath in index.get("forges", {}):
        prefix = forge_relpath + "/"
        if match_path.startswith(prefix):
            remainder = match_path[len(prefix) :]
            parts = remainder.split("/", 1)
            entry_loose_name = parts[0]
            subpart_loose_name = parts[1] if len(parts) > 1 else None
            return forge_relpath, entry_loose_name, subpart_loose_name
    raise ValueError(f"{match_path!r} does not match any forge in the index")


def read_entry_bytes(
    root: str, index: dict, game: Game, forge_relpath: str, entry_loose_name: str
) -> bytes:
    """Reads exactly one forge entry's raw bytes directly via its stored
    offset/length (search.build_index records these), without touching any
    other entry in the forge."""
    forge_data = index.get("forges", {}).get(forge_relpath)
    if forge_data is None:
        raise ValueError(f"{forge_relpath!r} not in the index; run search (or --reindex) first")
    entry = next(
        (e for e in forge_data["entries"] if e["loose_name"] == entry_loose_name), None
    )
    if entry is None:
        raise ValueError(
            f"{entry_loose_name!r} not found in {forge_relpath!r}; run search (or --reindex) first"
        )
    legacy = is_legacy(game)
    with open(os.path.join(root, forge_relpath), "rb") as f:
        f.seek(entry["offset"] + (LEGACY_DATA_HEADER_SIZE if legacy else 0))
        return f.read(entry["length"])


def find_subpart_payload(
    data: bytes, game: Game, schema: Schema | None, subpart_loose_name: str
) -> bytes:
    """Locates one sub-part's raw payload within a .data entry's raw bytes
    by its composed loose filename (`{index}_-_{name}.{ext}`, matching
    `search_index`'s own convention)."""
    with io.BytesIO(data) as buf:
        for sub_index, ext_code, name, payload in iter_datafile_subparts(buf, game):
            ext = resolve_extension(ext_code, schema)
            if f"{sub_index}_-_{name}.{ext}" == subpart_loose_name:
                return payload
    raise ValueError(f"sub-part {subpart_loose_name!r} not found in this .data entry")


def extract_match(
    root: str,
    game: Game,
    match_path: str,
    out_dir: str,
    index_path: str | None = None,
    raw: bool = False,
) -> tuple[str, list[str]]:
    """Extracts one search match into out_dir. By default, decodes it to XML
    (+ any image sidecars) reusing objectxml.write_object_xml -- no new
    decode logic. `raw=True` instead writes the sub-part's raw bytes
    unchanged (no schema decode at all) for types whose XML conversion
    isn't reliable yet (e.g. FX -- see objectxml.py's module docstring on
    unconfirmed constructs); pair with `replace(..., raw=True)`. Only
    sub-part-level matches are extractable this way; a match naming a whole
    `.data`/`.MetaFile`/`.PrefetchInfo` entry raises (use `unpack`/
    `unpack-data` for those instead). Returns (out_path, sidecar_paths)
    (sidecar_paths is always empty when raw=True)."""
    root = os.path.abspath(root)
    index = build_index(root, game, index_path=index_path)
    forge_relpath, entry_loose_name, subpart_loose_name = parse_match_path(index, match_path)
    if subpart_loose_name is None:
        raise ValueError(
            f"{match_path!r} names a whole entry, not a sub-part -- use `unpack`/"
            "`unpack-data` to extract it instead"
        )

    data = read_entry_bytes(root, index, game, forge_relpath, entry_loose_name)
    schema = Schema.load_default(game)
    payload = find_subpart_payload(data, game, schema, subpart_loose_name)

    os.makedirs(out_dir, exist_ok=True)
    if raw:
        out_path = os.path.join(out_dir, subpart_loose_name)
        with open(out_path, "wb") as f:
            f.write(payload)
        return out_path, []

    out_path = os.path.join(out_dir, subpart_loose_name + ".xml")
    sidecars = write_object_xml(payload, schema, is_legacy(game), out_path)
    return out_path, sidecars
