from __future__ import annotations

import argparse
import sys

from .datafile import repack_datafile, unpack_datafile
from .forge import backup, repack, unpack
from .games import Game, is_legacy
from .objectxml import read_object_xml, write_object_xml
from .schema import Schema

_GAME_CHOICES = {
    "acb": Game.BROTHERHOOD,
    "acr": Game.REVELATIONS,
    "ac3": Game.AC3,
    "ac4": Game.BLACK_FLAG,
}


def _add_game_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--game",
        required=True,
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
        help="a matching .schema file (see ./schema/) -- when given, sub-parts get real "
        "extension names plus a best-effort XML conversion (with image sidecars for "
        "recognized textures) alongside each raw sub-part; see objectxml.py",
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
    p_obj_to_xml.add_argument("--schema", metavar="SCHEMA_FILE", required=True)
    _add_game_arg(p_obj_to_xml)

    p_xml_to_obj = sub.add_parser(
        "xml-to-object", help="convert an object-to-xml (or unpack-data --schema) XML file back to binary"
    )
    p_xml_to_obj.add_argument("xml_file")
    p_xml_to_obj.add_argument("object_file")
    _add_game_arg(p_xml_to_obj)

    args = parser.parse_args(argv)
    game = _GAME_CHOICES[args.game]

    if args.command == "unpack":
        if args.backup:
            dest = backup(args.forge_file)
            print(f"Backed up to {dest}")
        entries = unpack(args.forge_file, args.out_dir, game)
        print(f"Unpacked {len(entries)} entries to {args.out_dir}")
    elif args.command == "repack":
        repack(args.in_dir, args.forge_file, game)
        print(f"Wrote {args.forge_file}")
    elif args.command == "unpack-data":
        schema = Schema.load(args.schema) if args.schema else None
        unpack_datafile(args.data_file, args.out_dir, game, schema=schema)
        print(f"Unpacked {args.data_file} to {args.out_dir}")
    elif args.command == "repack-data":
        repack_datafile(args.in_dir, args.data_file, game)
        print(f"Wrote {args.data_file}")
    elif args.command == "object-to-xml":
        with open(args.object_file, "rb") as f:
            data = f.read()
        schema = Schema.load(args.schema)
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
