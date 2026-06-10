"""内置声纹包 — 索引↔名称映射。

映射：
    =====  ============
    Index  Name
    =====  ============
    0      John
    1      Frank
    2      Michael
    3      Qingqing
    4      Xixi
    5      Zhong
    6      Angle
    7      Albert
    =====  ============
"""

from importlib import resources

_PEOPLE: list[str] = [
    "john",
    "frank",
    "michael",
    "qingqing",
    "xixi",
    "zhong",
    "angle",
    "albert",
]


def get_voiceprint_path(index: int) -> str:
    if index < 0 or index >= len(_PEOPLE):
        raise IndexError(
            f"package_pk_index {index} out of range (0-{len(_PEOPLE) - 1})"
        )
    return ""


def get_voiceprint_name(index: int) -> str:
    if index < 0 or index >= len(_PEOPLE):
        raise IndexError(
            f"package_pk_index {index} out of range (0-{len(_PEOPLE) - 1})"
        )
    return _PEOPLE[index]
