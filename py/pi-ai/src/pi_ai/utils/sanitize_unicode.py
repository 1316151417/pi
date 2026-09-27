"""Remove unpaired Unicode surrogates without changing valid pairs."""

import re

__all__ = ["sanitize_surrogates"]

_UNPAIRED_SURROGATES = re.compile(r"[\ud800-\udbff](?![\udc00-\udfff])|(?<![\ud800-\udbff])[\udc00-\udfff]")


def sanitize_surrogates(text: str) -> str:
    return _UNPAIRED_SURROGATES.sub("", text)
