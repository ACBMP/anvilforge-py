from enum import IntEnum


class Game(IntEnum):
    """Subset of AnvilToolkit's Utils/Game.cs enum, restricted to the titles this
    rewrite supports. Numeric values are kept identical to the original enum in
    case they ever need to line up with data recovered from real files."""

    BLACK_FLAG = 0
    BROTHERHOOD = 3
    REVELATIONS = 4
    AC3 = 5


# Forge container version ("scimitar" version field) used by each game.
FORGE_VERSION: dict[Game, int] = {
    Game.BROTHERHOOD: 25,
    Game.REVELATIONS: 25,
    Game.AC3: 27,
    Game.BLACK_FLAG: 27,
}

# Games using the "legacy" per-entry layout: 32-bit IDs/class-ids, 188-byte
# entry info records, and a 440-byte "FILEDATA" header preceding each entry's
# raw payload. Everything else (format 27) uses 64-bit IDs, 192-byte info
# records, and no per-entry header.
LEGACY_GAMES: frozenset[Game] = frozenset({Game.BROTHERHOOD, Game.REVELATIONS})

# Per-entry-set chunk size: ForgeFile splits entries into FileSets of at most
# this many, mirroring the original tool's hardcoded 5000.
ENTRIES_PER_FILESET = 5000

GAME_EXECUTABLES: dict[str, Game] = {
    "acbmp.exe": Game.BROTHERHOOD,
    "acbsp.exe": Game.BROTHERHOOD,
    "assassinscreedbrotherhood.exe": Game.BROTHERHOOD,
    "acrmp.exe": Game.REVELATIONS,
    "acrsp.exe": Game.REVELATIONS,
    "acrpr.exe": Game.REVELATIONS,
    "assassinscreedrevelations.exe": Game.REVELATIONS,
    "ac3sp.exe": Game.AC3,
    "ac3mp.exe": Game.AC3,
    "ac4bfsp.exe": Game.BLACK_FLAG,
    "ac4bfmp.exe": Game.BLACK_FLAG,
}


def forge_version(game: Game) -> int:
    try:
        return FORGE_VERSION[game]
    except KeyError:
        raise NotImplementedError(f"{game!r} is not supported by this rewrite yet") from None


def is_legacy(game: Game) -> bool:
    return game in LEGACY_GAMES
