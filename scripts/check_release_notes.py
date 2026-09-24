"""Fail when a releasable commit is missing from the pending release notes.

release-please discards a commit its grammar cannot parse, logs that only at
debug level and still exits success, and the loss is permanent once the
release is cut. Run on main after release-please has written the release
branch, this checks every commit no release tag contains yet against the
section release-please wrote for the next version, whatever the cause.

It works per commit, from the subject alone. A commit counts as listed once
its short sha appears, so a second entry from the same commit can mask a
dropped first one, and a breaking change declared only in a footer is not
required. A commit whose PR retypes it to a hidden type with
BEGIN_COMMIT_OVERRIDE is reported until the release is cut, because git still
records the original type.
"""

import argparse
import json
import subprocess
import sys

from release_notes import Commit, is_releasable, missing_entries, pending_section

MANIFEST = ".release-please-manifest.json"
# release-please-config.json: component "langchain-litellm", tag-separator "==".
TAG_PREFIX = "langchain-litellm=="
RELEASE_BRANCH = "origin/release-please--branches--main--components--langchain-litellm"
REMEDY = (
    "To restore the entry for a squash-merged PR, put a message release-please "
    "can parse between BEGIN_COMMIT_OVERRIDE and END_COMMIT_OVERRIDE in its "
    "description, then push to main or re-run the latest Release run. For a "
    "rebase-merged or directly pushed commit, add the entry by hand when the "
    "release is cut."
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], stdout=subprocess.PIPE, text=True, check=True
    ).stdout


def _exists(rev: str) -> bool:
    return (
        subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}"],
            stdout=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def _version(rev: str) -> str:
    """The root package's version in the release-please manifest at `rev`."""
    version: str = json.loads(_git("show", f"{rev}:{MANIFEST}"))["."]
    return version


def _commits(head: str) -> list[Commit]:
    """The commits behind `head` that no release tag contains yet.

    Excluding every tag, not only the one `head`'s manifest names, keeps a
    release cut after `head` from counting its own commits as unreleased.
    """
    commits = []
    log = _git("log", "--format=%H %s", head, "--not", f"--tags={TAG_PREFIX}*", "--")
    for line in log.splitlines():
        sha, _, subject = line.partition(" ")
        commits.append(Commit(sha, subject))
    return commits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main",
        default="HEAD",
        help="main, whose manifest names the last release (default: %(default)s)",
    )
    parser.add_argument(
        "--release",
        default=RELEASE_BRANCH,
        help="the branch holding the pending notes (default: %(default)s)",
    )
    args = parser.parse_args()

    released = _version(args.main)
    if not _exists(TAG_PREFIX + released):
        print(
            f"main's manifest names {released}, but {TAG_PREFIX}{released} does "
            "not exist: its release PR was merged and never tagged. Look for "
            "'untagged, merged release PRs outstanding' in the release-please log."
        )
        return 1
    releasable = [c for c in _commits(args.main) if is_releasable(c.subject)]

    # No release branch, or one whose version is already tagged, means
    # release-please has rendered none of them.
    pending = _version(args.release) if _exists(args.release) else released
    if _exists(TAG_PREFIX + pending):
        print("No release notes pending")
        missing = releasable
    else:
        changelog = _git("show", f"{args.release}:CHANGELOG.md")
        missing = missing_entries(releasable, pending_section(changelog, pending))
        print(
            f"CHANGELOG.md {pending} lists {len(releasable) - len(missing)} of "
            f"{len(releasable)} releasable commits since the last release"
        )

    for commit in missing:
        print(f"missing: {commit.sha[:7]} {commit.subject}")
    if missing:
        print(REMEDY)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
