from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .datafile import repack_datafile, unpack_datafile
from .forge import backup, repack, unpack
from .games import GAME_CODES, GAME_EXECUTABLES, Game, is_legacy
from .objectxml import read_object_xml, write_object_xml
from .schema import Schema
from .replace import replace as run_replace
from .search import extract_match, search as run_search

_GAME_CHOICES = {code: game for game, code in GAME_CODES.items()}


def _detect_game(game_path: str) -> Game:
    """Auto-detects --game for `search` by looking for a known game
    executable (games.GAME_EXECUTABLES) anywhere under game_path."""
    found: set[Game] = set()
    for _dirpath, _dirnames, filenames in os.walk(game_path):
        for name in filenames:
            game = GAME_EXECUTABLES.get(name.lower())
            if game is not None:
                found.add(game)
    if len(found) == 1:
        return found.pop()
    if not found:
        raise SystemExit(
            f"could not auto-detect game under {game_path!r} (no known game executable "
            "found nearby); pass --game explicitly"
        )
    raise SystemExit(f"multiple games detected under {game_path!r}; pass --game explicitly")


def _add_game_arg(p: argparse.ArgumentParser, required: bool = True) -> None:
    p.add_argument(
        "--game",
        required=required,
        choices=sorted(_GAME_CHOICES),
        help="acb=Brotherhood, acr=Revelations, ac3=Assassin's Creed III, ac4=Black Flag",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="anvilforge", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_unpack = sub.add_parser("unpack", help="extract a .forge file into a folder of loose files")
    p_unpack.add_argument("forge_file")
    p_unpack.add_argument("out_dir")
    _add_game_arg(p_unpack)
    p_unpack.add_argument(
        "--backup", action="store_true", help="copy the source .forge into Backups/ first"
    )

    p_repack = sub.add_parser("repack", help="build a .forge file from a folder of loose files")
    p_repack.add_argument("in_dir")
    p_repack.add_argument("forge_file")
    _add_game_arg(p_repack)

    p_unpack_data = sub.add_parser(
        "unpack-data", help="extract a loose .data entry's named sub-parts into a folder"
    )
    p_unpack_data.add_argument("data_file")
    p_unpack_data.add_argument("out_dir")
    _add_game_arg(p_unpack_data)
    p_unpack_data.add_argument(
        "--schema",
        metavar="SCHEMA_FILE",
        help="a .schema file to use instead of the one bundled for --game -- sub-parts get "
        "real extension names plus a best-effort XML conversion (with image sidecars for "
        "recognized textures) alongside each raw sub-part; see objectxml.py",
    )
    p_unpack_data.add_argument(
        "--raw",
        action="store_true",
        help="skip schema-based extension/XML resolution and dump sub-parts as-is",
    )

    p_repack_data = sub.add_parser(
        "repack-data", help="build a loose .data entry from a folder of its sub-parts"
    )
    p_repack_data.add_argument("in_dir")
    p_repack_data.add_argument("data_file")
    _add_game_arg(p_repack_data)

    p_obj_to_xml = sub.add_parser(
        "object-to-xml", help="convert one schema-described sub-part (from unpack-data) to XML"
    )
    p_obj_to_xml.add_argument("object_file")
    p_obj_to_xml.add_argument("xml_file")
    p_obj_to_xml.add_argument(
        "--schema",
        metavar="SCHEMA_FILE",
        help="a .schema file to use instead of the one bundled for --game",
    )
    _add_game_arg(p_obj_to_xml)

    p_xml_to_obj = sub.add_parser(
        "xml-to-object", help="convert an object-to-xml (or unpack-data --schema) XML file back to binary"
    )
    p_xml_to_obj.add_argument("xml_file")
    p_xml_to_obj.add_argument("object_file")
    _add_game_arg(p_xml_to_obj)

    p_search = sub.add_parser(
        "search",
        help="find named assets across a whole game install (builds/reuses an index)",
    )
    p_search.add_argument("game_path")
    p_search.add_argument("query")
    _add_game_arg(p_search, required=False)
    p_search.add_argument(
        "--reindex", action="store_true", help="force a full index rebuild instead of reusing it"
    )
    p_search.add_argument(
        "--index-file",
        metavar="PATH",
        help="index file location (default: <GAME_PATH>/.anvilforge_index.json)",
    )

    p_extract = sub.add_parser(
        "extract", help="pull one `search` match out as an editable XML file"
    )
    p_extract.add_argument("game_path")
    p_extract.add_argument("match_path", help="a virtual path copied from `search` output")
    p_extract.add_argument("out_dir")
    _add_game_arg(p_extract, required=False)
    p_extract.add_argument(
        "--index-file",
        metavar="PATH",
        help="index file location (default: <GAME_PATH>/.anvilforge_index.json)",
    )
    p_extract.add_argument(
        "--raw",
        action="store_true",
        help="write the sub-part's raw bytes unchanged instead of decoding to XML -- use this "
        "for types whose XML conversion isn't reliable yet (e.g. FX); pair with `replace --raw`",
    )

    p_replace = sub.add_parser(
        "replace",
        help="push an edited `extract` file into every current `search` match of a query",
    )
    p_replace.add_argument("game_path")
    p_replace.add_argument("query")
    p_replace.add_argument("edited_file")
    _add_game_arg(p_replace, required=False)
    p_replace.add_argument(
        "--index-file",
        metavar="PATH",
        help="index file location (default: <GAME_PATH>/.anvilforge_index.json)",
    )
    p_replace.add_argument(
        "--exclude",
        metavar="MATCH_PATH",
        action="append",
        default=[],
        help="skip this match (repeatable)",
    )
    p_replace.add_argument(
        "--backup", action="store_true", help="copy each affected .forge into Backups/ first"
    )
    p_replace.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would be touched (incl. the type-compatibility check) without writing anything",
    )
    p_replace.add_argument(
        "--raw",
        action="store_true",
        help="edited_file is the sub-part's raw bytes, not XML -- skips schema decode entirely "
        "(both to check and to apply), for types whose XML conversion isn't reliable yet (e.g. FX)",
    )

    args = parser.parse_args(argv)

    if args.command == "search":
        game = _GAME_CHOICES[args.game] if args.game else _detect_game(args.game_path)
        matches = run_search(
            args.game_path, game, args.query, index_path=args.index_file, reindex=args.reindex
        )
        for match in matches:
            print(match)
        return 0
    elif args.command == "extract":
        game = _GAME_CHOICES[args.game] if args.game else _detect_game(args.game_path)
        out_path, sidecars = extract_match(
            args.game_path,
            game,
            args.match_path,
            args.out_dir,
            index_path=args.index_file,
            raw=args.raw,
        )
        print(f"Wrote {out_path}" + (f" (+ {len(sidecars)} sidecar file(s))" if sidecars else ""))
        return 0
    elif args.command == "replace":
        game = _GAME_CHOICES[args.game] if args.game else _detect_game(args.game_path)
        plan = run_replace(
            args.game_path,
            game,
            args.query,
            args.edited_file,
            index_path=args.index_file,
            exclude=args.exclude,
            make_backup=args.backup,
            dry_run=args.dry_run,
            raw=args.raw,
        )
        for path in plan.skipped_whole_entries:
            print(f"Skipped (whole entry, not a sub-part): {path}")
        verb = "Would replace" if args.dry_run else "Replaced"
        print(f"{verb} {plan.subpart_count} sub-part(s) across {len(plan.forges)} forge(s)")
        if args.dry_run:
            for forge_relpath, data_entries in plan.forges.items():
                for entry_loose_name, subparts in data_entries.items():
                    for subpart_loose_name in subparts:
                        print(f"  {forge_relpath}/{entry_loose_name}/{subpart_loose_name}")
        return 0

    game = _GAME_CHOICES[args.game]

    if args.command == "unpack":
        if args.backup:
            dest = backup(args.forge_file)
            print(f"Backed up to {dest}")
        out_dir = Path(args.out_dir) / Path(args.forge_file).stem
        entries = unpack(args.forge_file, str(out_dir), game)
        print(f"Unpacked {len(entries)} entries to {out_dir}")
    elif args.command == "repack":
        repack(args.in_dir, args.forge_file, game)
        print(f"Wrote {args.forge_file}")
    elif args.command == "unpack-data":
        if args.raw:
            schema = None
        elif args.schema:
            schema = Schema.load(args.schema)
        else:
            schema = Schema.load_default(game)
        out_dir = Path(args.out_dir) / Path(args.data_file).stem
        unpack_datafile(args.data_file, str(out_dir), game, schema=schema)
        print(f"Unpacked {args.data_file} to {out_dir}")
    elif args.command == "repack-data":
        repack_datafile(args.in_dir, args.data_file, game)
        print(f"Wrote {args.data_file}")
    elif args.command == "object-to-xml":
        with open(args.object_file, "rb") as f:
            data = f.read()
        schema = Schema.load(args.schema) if args.schema else Schema.load_default(game)
        written = write_object_xml(data, schema, is_legacy(game), args.xml_file)
        print(f"Wrote {args.xml_file}" + (f" (+ {len(written)} sidecar file(s))" if written else ""))
    elif args.command == "xml-to-object":
        data = read_object_xml(args.xml_file, is_legacy(game))
        with open(args.object_file, "wb") as f:
            f.write(data)
        print(f"Wrote {args.object_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
