"""Small shared primitives for artifact identities and publication.

Hashing and revision validation are correctness rules used by score, endpoint, confidence, and
predictor artifacts. Atomic JSON publication prevents independently scheduled stages from seeing
partial records. Keeping one implementation prevents the stages from accepting different bytes as
the same input.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile


_PINNED_REVISION = re.compile(r"^[0-9a-f]{40}$")


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def atomic_json_dump(value, path):
    """Atomically replace one finite JSON artifact after its bytes reach disk."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".", suffix=".partial", dir=directory
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def identified(payload):
    """Add a SHA-256 identity over every other field in an immutable record."""
    if "identity" in payload:
        raise ValueError("payload already has an identity")
    return dict(payload, identity=hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest())


def validate_identity(payload, label):
    core = dict(payload)
    identity = core.pop("identity", None)
    expected = hashlib.sha256(canonical_json(core).encode("utf-8")).hexdigest()
    if identity != expected:
        raise ValueError("%s identity does not match its contents" % label)
    return payload


def require_pinned_revision(revision, label="revision"):
    revision = str(revision or "")
    if not _PINNED_REVISION.fullmatch(revision):
        raise ValueError("%s %r is not a pinned 40-hex commit" % (label, revision))
    return revision
