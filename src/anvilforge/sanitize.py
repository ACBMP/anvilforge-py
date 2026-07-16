"""Port of the entry-name cleanup in ForgeEntry.ReadInfo (DataStorage.IllegalCharacters)."""
from __future__ import annotations

ILLEGAL_CHARACTERS = frozenset(
    [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 42, 47, 58, 60, 62, 63,
        92, 124,
    ]
)


def sanitize_entry_name(name: str) -> str:
    if not name:
        name = "Unnamed"
    name = name.replace("\x00", "")
    for code in ILLEGAL_CHARACTERS:
        name = name.replace(chr(code), "*")
    name = name.replace("\x0b", "*").replace("\x15", "*")
    name = name.replace("*", "")
    return name
