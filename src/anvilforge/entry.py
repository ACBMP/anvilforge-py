"""Port of AnvilToolkit.FileTypes.AnvilNext.Containers.ForgeEntry."""
from __future__ import annotations

from dataclasses import dataclass


def entry_size(legacy: bool) -> int:
    """Size in bytes of one entry's "info" record: 188 for the legacy
    (Brotherhood/Revelations) layout, 192 for the modern (AC3/BlackFlag) one."""
    return 188 if legacy else 192


@dataclass
class ForgeEntry:
    id: int = 0
    engine_version: int = 0
    offset: int = 0
    length_on_disk: int = 0
    umac_hash: int = 0
    extension: int = 0
    revision_number_data: int = 0
    revision_number_attributes: int = 0
    parent: int = 0
    timestamp: int = 0
    name: str = ""
    scc_status_data: int = 0
    metafile_key: int = 0
    scc_status_attributes: int = 0
    is_hidden: int = 0
