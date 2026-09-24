"""Reconcile releasable commits against the release notes pending for them.

Text in, commits out: check_release_notes.py does the git and file access.
"""

import re
from collections.abc import Iterable
from typing import NamedTuple

# The types release-please-config.json gives a changelog section, and any type
# marked breaking, which release-please renders even when the type is hidden.
# Lenient past the type on purpose: a header its grammar rejects, such as
# `fix : x` or the `Revert "fix: x"` GitHub's revert button writes, is a loss too.
_RELEASABLE = re.compile(
    r'(?:revert\s+")?(?:feat|fix|perf|revert)\b\s*[(!:]|[^\s(:!]+(?:\([^()]*\))?!:',
    re.IGNORECASE,
)
_HEADING = re.compile(r"^## ", re.MULTILINE)


class Commit(NamedTuple):
    """A commit as `git log --format='%H %s'` reports it."""

    sha: str
    subject: str


def is_releasable(subject: str) -> bool:
    """Whether the subject's conventional type earns an entry in the notes."""
    return _RELEASABLE.match(subject) is not None


def pending_section(changelog: str, version: str) -> str:
    """The text of `version`'s section, up to the next release heading.

    Raises:
        ValueError: if the changelog has no section for `version`.
    """
    heading = re.search(rf"^## \[{re.escape(version)}\]", changelog, re.MULTILINE)
    if heading is None:
        raise ValueError(f"CHANGELOG.md has no section for {version}")
    following = _HEADING.search(changelog, heading.end())
    return changelog[heading.start() : following.start() if following else None]


def missing_entries(commits: Iterable[Commit], section: str) -> list[Commit]:
    """The releasable commits whose short sha the section never cites."""
    # release-please cites seven hex digits, as link text and as the full
    # sha's prefix, so match at the start of a token and never inside one.
    return [
        commit
        for commit in commits
        if is_releasable(commit.subject)
        and not re.search(rf"\b{re.escape(commit.sha[:7])}", section)
    ]
