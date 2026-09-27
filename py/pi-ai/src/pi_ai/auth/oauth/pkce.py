"""S256 PKCE verifier/challenge generation using the standard library."""

import base64
import hashlib
import secrets
from dataclasses import dataclass

__all__ = ["PKCE", "generate_pkce"]


@dataclass(frozen=True)
class PKCE:
    verifier: str
    challenge: str


async def generate_pkce() -> PKCE:
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii").rstrip("=")
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("utf-8")).digest()).decode("ascii").rstrip("=")
    return PKCE(verifier=verifier, challenge=challenge)
