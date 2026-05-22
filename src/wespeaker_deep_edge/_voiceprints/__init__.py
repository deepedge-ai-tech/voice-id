"""内置声纹包 — 打包到 whl 中的预注册声纹文件。

索引映射：

    =====  ============
    Index  Name
    =====  ============
    0      John
    1      Frank
    2      Michael
    3      Qingqing
    4      Xixi
    5      Zhong
    6      John_usb_yun
    7      Angle
    8      Albert
    =====  ============
"""

from importlib import resources

_PEOPLE: list[str] = ["john", "frank", "michael", "qingqing", "xixi", "zhong", "john_usb_yun", "angle", "albert"]


def get_voiceprint_path(index: int) -> str:
    """返回第 index 个人的内置声纹 .pkl 文件绝对路径。

    Args:
        index: 声纹索引 (0-8).

    Returns:
        内置声纹文件路径。

    Raises:
        IndexError: index 超出范围。
    """
    if index < 0 or index >= len(_PEOPLE):
        raise IndexError(
            f"package_pk_index {index} 超出范围 (0-{len(_PEOPLE) - 1})"
        )
    name = _PEOPLE[index]
    return str(
        resources.files("wespeaker_deep_edge._voiceprints") / f"voice_{name}.pkl"
    )


def get_voiceprint_name(index: int) -> str:
    """返回第 index 个人的名字。

    Args:
        index: 声纹索引 (0-8).

    Returns:
        名字字符串。
    """
    if index < 0 or index >= len(_PEOPLE):
        raise IndexError(
            f"package_pk_index {index} 超出范围 (0-{len(_PEOPLE) - 1})"
        )
    return _PEOPLE[index]
